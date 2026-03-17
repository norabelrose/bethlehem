"""
Tests for HebrewCalendarResult save/load roundtrip.
"""

import json
import pytest

from bethlehem import HebrewCalendarResult, CalendarEntry

NISAN = CalendarEntry(
    mi=0,
    hname="Nisan",
    am_yr=3758,
    evening_jd=1720692.5,
    greg_d=14,
    greg_mo=4,
    greg_yr=-2,
    greg_str="14 Apr 3 BC",
    cat="A",
    q=0.45,
    arcl=12.3,
    arcv=8.7,
    W=0.21,
    moon_alt=10.5,
    moon_age_h=27.3,
    lag_min=42.0,
    uncertain=False,
    note="",
    days=30,
    fm_hday=15,
    fm_local_h=20.5,
)

IYAR = CalendarEntry(
    mi=1,
    hname="Iyar",
    am_yr=3758,
    evening_jd=1720722.5,
    greg_d=14,
    greg_mo=5,
    greg_yr=-2,
    greg_str="14 May 3 BC",
    cat="B",
    q=0.1,
    arcl=10.1,
    arcv=6.2,
    W=0.18,
    moon_alt=9.0,
    moon_age_h=26.0,
    lag_min=38.0,
    uncertain=True,
    note="±1 day (borderline)",
    days=None,
    fm_hday=None,
    fm_local_h=None,
)


def make_result(calendar: list[CalendarEntry] | None = None) -> HebrewCalendarResult:
    """Build a minimal HebrewCalendarResult for testing."""
    return HebrewCalendarResult(
        calendar=[NISAN, IYAR] if calendar is None else calendar,
        loc_name="Jerusalem",
        loc_lat=31.7683,
        loc_lon=35.2137,
        start_year=-2,
        end_year=0,
    )


class TestHebrewCalendarResultSaveLoad:
    def test_roundtrip_fields(self, tmp_path):
        """Saved then loaded result has identical top-level fields."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        assert loaded.loc_name == original.loc_name
        assert loaded.loc_lat == original.loc_lat
        assert loaded.loc_lon == original.loc_lon
        assert loaded.start_year == original.start_year
        assert loaded.end_year == original.end_year

    def test_roundtrip_calendar_length(self, tmp_path):
        """Loaded calendar has the same number of months."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        assert len(loaded.calendar) == len(original.calendar)

    def test_roundtrip_calendar_entries(self, tmp_path):
        """Each calendar entry survives the roundtrip intact."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        for orig_row, load_row in zip(original.calendar, loaded.calendar):
            assert load_row == orig_row

    def test_roundtrip_entries_are_dataclasses(self, tmp_path):
        """Loaded calendar entries are CalendarEntry instances, not dicts."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        for entry in loaded.calendar:
            assert isinstance(entry, CalendarEntry)

    def test_json_file_is_valid(self, tmp_path):
        """Saved file is valid JSON with expected top-level keys."""
        result = make_result()
        path = tmp_path / "calendar.json"
        result.save(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data.keys()) == {
            "location",
            "lat",
            "lon",
            "start_year",
            "end_year",
            "calendar",
        }

    def test_roundtrip_preserves_none_values(self, tmp_path):
        """None values in calendar entries (days, fm_hday, fm_local_h) survive roundtrip."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        last = loaded.calendar[-1]
        assert last.days is None
        assert last.fm_hday is None
        assert last.fm_local_h is None

    def test_roundtrip_uncertain_flag(self, tmp_path):
        """Boolean uncertain flag is preserved exactly."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        assert loaded.calendar[0].uncertain is False
        assert loaded.calendar[1].uncertain is True

    def test_roundtrip_numeric_precision(self, tmp_path):
        """Float fields (q, arcl, lat/lon etc.) survive roundtrip without loss."""
        original = make_result()
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        assert loaded.loc_lat == pytest.approx(original.loc_lat)
        assert loaded.loc_lon == pytest.approx(original.loc_lon)
        assert loaded.calendar[0].q == pytest.approx(original.calendar[0].q)

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Loading a missing file raises an exception."""
        with pytest.raises(Exception):
            HebrewCalendarResult.from_file(tmp_path / "missing.json")

    def test_empty_calendar_roundtrip(self, tmp_path):
        """An empty calendar list survives save/load."""
        original = make_result(calendar=[])
        path = tmp_path / "calendar.json"
        original.save(path)
        loaded = HebrewCalendarResult.from_file(path)

        assert loaded.calendar == []
