"""Filter language for selecting visits.

------------------------------------------------------------------------------
Configuration this module uses: NONE.
Filters are constructed from CLI arguments by cli.py; this module just defines
the matching logic and operator parsing.
------------------------------------------------------------------------------

CLI surface (all repeatable):

  --gender female               (OR on repeats)
  --age ">20"                   (AND on repeats, so a range works: '>=18' '<65')
  --age "<=40"
  --name "Ivan KERQY"           (OR on repeats; matches person_name exactly)
  --visit-id <uuid>             (OR on repeats)
  --person-id <uuid>            (OR on repeats)
  --wears-glasses               (boolean flag: only include glasses=True)
  --no-glasses                  (boolean flag: only include glasses=False)
  --dominant-hand right         (OR on repeats; right | left | ambidextrous)

Repeat semantics:
  - Numeric comparators (age) combine with AND so ranges work.
  - Categorical fields (gender, name, dominant_hand, visit_id, person_id)
    combine with OR so sets work.
  - Booleans are mutually exclusive flags, not repeatable.
"""

from __future__ import annotations

import operator
import re
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .metadata import Visit


_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">":  operator.gt,
    "<":  operator.lt,
    "=":  operator.eq,
}

# Two-char operators must come before single-char in the alternation so
# ">=20" doesn't get misread as ">" + "=20".
_COMPARATOR_PATTERN = re.compile(
    r"^\s*(>=|<=|==|!=|>|<|=)?\s*(-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class NumericConstraint:
    """A single comparator + value pair, e.g. (op.gt, 20)."""

    op: Callable[[float, float], bool]
    value: float
    raw: str

    def matches(self, x: float) -> bool:
        return self.op(x, self.value)


def parse_numeric_constraint(text: str) -> NumericConstraint:
    """Parse strings like '>20', '<=40', '==30', or bare '25' (treated as ==)."""
    match = _COMPARATOR_PATTERN.match(text)
    if not match:
        raise ValueError(
            f"Cannot parse numeric constraint {text!r}. "
            f"Expected forms: '>20', '>=18', '<=40', '==30', or just '30'."
        )
    op_str, value_str = match.group(1), match.group(2)
    op_str = op_str if op_str else "=="
    return NumericConstraint(
        op=_COMPARATORS[op_str],
        value=float(value_str),
        raw=text.strip(),
    )


@dataclass
class VisitFilter:
    """A composed predicate over Visit. Empty filter accepts everything."""

    genders: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    visit_ids: set[uuid.UUID] = field(default_factory=set)
    person_ids: set[uuid.UUID] = field(default_factory=set)
    dominant_hands: set[str] = field(default_factory=set)
    wears_glasses: bool | None = None       # None = don't care
    age_constraints: list[NumericConstraint] = field(default_factory=list)

    def matches(self, visit: Visit) -> bool:
        if self.genders and visit.gender.lower() not in self.genders:
            return False
        if self.names and visit.person_name not in self.names:
            return False
        if self.visit_ids and visit.visit_id not in self.visit_ids:
            return False
        if self.person_ids and visit.person_id not in self.person_ids:
            return False
        if self.dominant_hands and visit.dominant_hand.lower() not in self.dominant_hands:
            return False
        if self.wears_glasses is not None and visit.wears_glasses != self.wears_glasses:
            return False
        for constraint in self.age_constraints:
            if not constraint.matches(visit.age):
                return False
        return True

    def apply(self, visits: list[Visit]) -> list[Visit]:
        return [v for v in visits if self.matches(v)]

    def describe(self) -> str:
        parts: list[str] = []
        if self.genders:
            parts.append(f"gender in {sorted(self.genders)}")
        if self.age_constraints:
            parts.append("age " + " AND ".join(c.raw for c in self.age_constraints))
        if self.wears_glasses is not None:
            parts.append(f"wears_glasses={self.wears_glasses}")
        if self.dominant_hands:
            parts.append(f"dominant_hand in {sorted(self.dominant_hands)}")
        if self.names:
            parts.append(f"name in {sorted(self.names)}")
        if self.visit_ids:
            parts.append(f"{len(self.visit_ids)} visit_id(s)")
        if self.person_ids:
            parts.append(f"{len(self.person_ids)} person_id(s)")
        return ", ".join(parts) if parts else "no filters"


def build_filter(
    *,
    genders: list[str] | None = None,
    ages: list[str] | None = None,
    names: list[str] | None = None,
    visit_ids: list[str] | None = None,
    person_ids: list[str] | None = None,
    dominant_hands: list[str] | None = None,
    wears_glasses: bool | None = None,
) -> VisitFilter:
    """Construct a VisitFilter from raw CLI arguments."""
    f = VisitFilter()
    if genders:
        f.genders = {g.lower() for g in genders}
    if names:
        f.names = set(names)
    if visit_ids:
        f.visit_ids = {uuid.UUID(v) for v in visit_ids}
    if person_ids:
        f.person_ids = {uuid.UUID(v) for v in person_ids}
    if dominant_hands:
        f.dominant_hands = {h.lower() for h in dominant_hands}
    if wears_glasses is not None:
        f.wears_glasses = wears_glasses
    if ages:
        f.age_constraints = [parse_numeric_constraint(a) for a in ages]
    return f
