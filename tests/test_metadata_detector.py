"""
tests/test_metadata_detector.py
---------------------------------
Unit tests for borehole-level metadata extraction from raw page text.
"""

import pytest
from borehole_parser.metadata_detector import extract_metadata


SAMPLE_TEXT_FORMAT_1 = """
RIYADH GEOTECHNIQUE & FOUNDATIONS
Test Boring Log
BH NO.   BH-01
SHEET  1  OF  2
PROJECT: Bachelor's Village, Ma'aden Phosphate Project, Ras Az Zawr, Geo. Inv.
PROJECT NO.: 06-K-3055
CLIENT: Saudi Arabian Mining Co. (Ma'aden)
ELEVATION:  3.62
DATE STARTED:  27-12-06
DATE COMPLETED:  27-12-06
GROUND WATER
DATE:  29-12-06    TIME: 06:30    WATER DEPTH: 2.00
DRILLING METHOD: Straight Rotary
DRILLER: Hadi
INSPECTOR: Glasuddin
"""

SAMPLE_TEXT_FORMAT_2 = """
JOB NO.        RYSM-25-01158
HOLE NO.       BH-07
SHEET   1   of   5
DATE from  11.01.26  to  14.01.26
PROJECT NAME KAFD-AREA 07
LOCATION: KAFD, RIYADH, KINGDOM OF SAUDI ARABIA
CLIENT: KAFD
METHOD RC
CO-ORDINATES
E 665812.11
N 2738920.20
G.W.D 26.22
Borehole Depth (M): 50.00
GROUND-LEVEL  +637.29  M
"""


class TestFormat1Metadata:
    def setup_method(self):
        self.meta = extract_metadata(SAMPLE_TEXT_FORMAT_1)

    def test_borehole_id(self):
        assert self.meta.get("borehole_id") == "BH-01"

    def test_date_started(self):
        assert self.meta.get("date_started") == "27-12-06"

    def test_gw_depth(self):
        assert self.meta.get("ground_water_depth_m") == "2.00"


class TestFormat2Metadata:
    def setup_method(self):
        self.meta = extract_metadata(SAMPLE_TEXT_FORMAT_2)

    def test_borehole_id(self):
        assert self.meta.get("borehole_id") in ("BH-07", "BH07")

    def test_gw_depth(self):
        assert self.meta.get("ground_water_depth_m") == "26.22"

    def test_total_depth(self):
        assert self.meta.get("total_depth_m") == "50.00"

    def test_elevation(self):
        assert self.meta.get("elevation_m") == "+637.29"

    def test_coordinates_north(self):
        assert self.meta.get("coordinates_north") == "2738920.20"

    def test_coordinates_east(self):
        assert self.meta.get("coordinates_east") == "665812.11"


class TestEmptyText:
    def test_empty_returns_dict(self):
        meta = extract_metadata("")
        assert isinstance(meta, dict)

    def test_none_returns_dict(self):
        meta = extract_metadata(None)
        assert isinstance(meta, dict)
