"""KEGG module rule parser + per-genome completeness scorer.

A KEGG module DEFINITION is a boolean rule over KO IDs:

    K00844 (K01810,K06859,K13810) K00850 (K01623,K11645)

Grammar (we implement the common subset that covers ~all metabolic modules):
  - whitespace between tokens   → AND
  - commas inside parens        → OR
  - parens                      → grouping
  - "+", "-" appear in some modules → treated as AND/optional (we ignore "-")

Completeness convention: we evaluate the rule with the per-genome KO set
substituted as 1/0, where AND = product, OR = max. The result is a 0.0-1.0
fractional score: 1.0 = pathway complete, 0.0 = no genes present, intermediate
values reflect partial coverage of the AND chain.

Example:
    rule = "K00001 (K00002,K00003) K00004"
    ko_set = {"K00001", "K00003"}      # has step 1, has alt for step 2, missing step 3
    completeness = 1 * max(0,1) * 0     # = 0  (chain broken at step 3)

For partial-credit grading, AND is replaced by a *fraction-present* aggregator
(`fractional=True` mode): the score becomes the average of the step scores,
weighted equally. This is what most KEGG completeness tools report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Tokenizer: KO IDs, parens, commas, plus, minus, whitespace.
TOKEN_RE = re.compile(r"K\d{5}|[()+,-]|\s+")


@dataclass
class Node:
    """AST node. type: 'KO' | 'AND' | 'OR'."""
    type: str
    ko: str | None = None
    children: list["Node"] | None = None


def tokenize(rule: str) -> list[str]:
    out: list[str] = []
    for m in TOKEN_RE.finditer(rule):
        tok = m.group(0)
        if tok.isspace():
            out.append(" ")
        else:
            out.append(tok)
    # Collapse runs of spaces
    cleaned: list[str] = []
    for tok in out:
        if tok == " ":
            if cleaned and cleaned[-1] != " ":
                cleaned.append(" ")
        else:
            cleaned.append(tok)
    if cleaned and cleaned[0] == " ":
        cleaned = cleaned[1:]
    if cleaned and cleaned[-1] == " ":
        cleaned = cleaned[:-1]
    return cleaned


class _Parser:
    """Recursive-descent parser over the token stream."""

    def __init__(self, tokens: list[str]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def consume(self) -> str:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        return self._and_seq(end_tokens={None, ")"})

    def _and_seq(self, end_tokens: set[str | None]) -> Node:
        children: list[Node] = []
        while self.peek() not in end_tokens:
            tok = self.peek()
            if tok == " ":
                self.consume()
                continue
            if tok == "+" or tok == "-":  # treat as AND-like glue
                self.consume()
                continue
            children.append(self._or_seq())
        if len(children) == 1:
            return children[0]
        return Node(type="AND", children=children)

    def _or_seq(self) -> Node:
        # An OR-sequence is a comma-separated list of factors *only inside parens*.
        # At the top level, commas don't appear (KEGG normalizes that).
        first = self._factor()
        children: list[Node] = [first]
        while self.peek() == ",":
            self.consume()
            children.append(self._factor())
        if len(children) == 1:
            return first
        return Node(type="OR", children=children)

    def _factor(self) -> Node:
        tok = self.peek()
        if tok == "(":
            self.consume()
            inner = self._and_seq(end_tokens={")"})
            if self.peek() == ")":
                self.consume()
            return inner
        if tok and tok.startswith("K") and len(tok) == 6 and tok[1:].isdigit():
            self.consume()
            return Node(type="KO", ko=tok)
        # Skip stray tokens defensively
        if tok is not None:
            self.consume()
        return Node(type="AND", children=[])


def parse_definition(definition: str) -> Node:
    return _Parser(tokenize(definition)).parse()


def evaluate(node: Node, ko_set: set[str], fractional: bool = True) -> float:
    """Return 0.0-1.0 completeness for this AST under the given ko_set.

    fractional=True  → AND = mean of children   (KEGG-style partial credit)
    fractional=False → AND = min of children    (strict; pathway must be intact)
    """
    if node.type == "KO":
        return 1.0 if node.ko in ko_set else 0.0
    if not node.children:
        return 0.0
    scores = [evaluate(c, ko_set, fractional) for c in node.children]
    if node.type == "AND":
        return float(sum(scores) / len(scores)) if fractional else float(min(scores))
    if node.type == "OR":
        return float(max(scores))
    return 0.0


def module_completeness(definition: str, ko_set: set[str], fractional: bool = True) -> float:
    return evaluate(parse_definition(definition), ko_set, fractional=fractional)
