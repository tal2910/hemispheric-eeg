"""Tests for filter parsing and application."""

import uuid
from pathlib import Path

import pytest

from hemispheric.filters import (
    build_filter,
    parse_numeric_constraint,
)
from hemispheric.metadata import Visit


def make_visit(
    *, name="X", gender="female", age=30, vid=None, pid=None,
    wears_glasses=False, dominant_hand="right",
) -> Visit:
    return Visit(
        visit_id=vid or uuid.uuid4(),
        person_id=pid or uuid.uuid4(),
        person_name=name,
        age=age,
        gender=gender,
        wears_glasses=wears_glasses,
        date_of_visit="2024-01-01",
        dominant_hand=dominant_hand,
        npy_path=Path("/tmp/x.npy"),
        metadata_path=Path("/tmp/x.json"),
        extra={},
    )


class TestParseNumericConstraint:
    @pytest.mark.parametrize("text,expected_op_name,expected_value", [
        (">20",   "gt", 20.0),
        (">=18",  "ge", 18.0),
        ("<40",   "lt", 40.0),
        ("<=65",  "le", 65.0),
        ("==30",  "eq", 30.0),
        ("!=25",  "ne", 25.0),
        ("=42",   "eq", 42.0),
        ("99",    "eq", 99.0),
        (" >= 21 ", "ge", 21.0),
        ("3.5",   "eq", 3.5),
    ])
    def test_valid(self, text, expected_op_name, expected_value):
        c = parse_numeric_constraint(text)
        assert c.value == expected_value
        assert c.op.__name__ == expected_op_name

    @pytest.mark.parametrize("text", ["", "abc", "> twenty", ">>10", "10..5"])
    def test_invalid(self, text):
        with pytest.raises(ValueError):
            parse_numeric_constraint(text)


class TestVisitFilter:
    def test_empty_filter_accepts_all(self):
        f = build_filter()
        assert f.matches(make_visit())

    def test_gender_match(self):
        f = build_filter(genders=["female"])
        assert f.matches(make_visit(gender="female"))
        assert not f.matches(make_visit(gender="male"))

    def test_gender_case_insensitive(self):
        f = build_filter(genders=["Female"])
        assert f.matches(make_visit(gender="FEMALE"))

    def test_gender_or(self):
        f = build_filter(genders=["female", "other"])
        assert f.matches(make_visit(gender="female"))
        assert f.matches(make_visit(gender="other"))
        assert not f.matches(make_visit(gender="male"))

    def test_age_single_constraint(self):
        f = build_filter(ages=[">20"])
        assert f.matches(make_visit(age=21))
        assert not f.matches(make_visit(age=20))

    def test_age_range_combines_with_AND(self):
        f = build_filter(ages=[">=18", "<65"])
        assert f.matches(make_visit(age=18))
        assert f.matches(make_visit(age=64))
        assert not f.matches(make_visit(age=17))
        assert not f.matches(make_visit(age=65))

    def test_combined_filters(self):
        f = build_filter(genders=["female"], ages=[">20"])
        assert f.matches(make_visit(gender="female", age=25))
        assert not f.matches(make_visit(gender="female", age=18))
        assert not f.matches(make_visit(gender="male", age=25))

    def test_visit_id_filter(self):
        target = uuid.uuid4()
        other = uuid.uuid4()
        f = build_filter(visit_ids=[str(target)])
        assert f.matches(make_visit(vid=target))
        assert not f.matches(make_visit(vid=other))

    def test_person_id_filter(self):
        target = uuid.uuid4()
        other = uuid.uuid4()
        f = build_filter(person_ids=[str(target)])
        assert f.matches(make_visit(pid=target))
        assert not f.matches(make_visit(pid=other))

    def test_wears_glasses_true(self):
        f = build_filter(wears_glasses=True)
        assert f.matches(make_visit(wears_glasses=True))
        assert not f.matches(make_visit(wears_glasses=False))

    def test_wears_glasses_false(self):
        f = build_filter(wears_glasses=False)
        assert f.matches(make_visit(wears_glasses=False))
        assert not f.matches(make_visit(wears_glasses=True))

    def test_wears_glasses_none_accepts_both(self):
        f = build_filter()  # don't care
        assert f.matches(make_visit(wears_glasses=True))
        assert f.matches(make_visit(wears_glasses=False))

    def test_dominant_hand_or(self):
        f = build_filter(dominant_hands=["right", "ambidextrous"])
        assert f.matches(make_visit(dominant_hand="right"))
        assert f.matches(make_visit(dominant_hand="ambidextrous"))
        assert not f.matches(make_visit(dominant_hand="left"))
