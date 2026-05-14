import numpy as np
import pandas as pd

from lay_0x1.lay0x1_core import normalize_team_name


def test_normalize_team_name_nulls():
    assert normalize_team_name(pd.NA) == ""
    assert normalize_team_name(np.nan) == ""
    assert normalize_team_name(None) == ""


def test_normalize_team_name_standard():
    assert normalize_team_name("Manchester United") == "manchesterunited"
    assert normalize_team_name("REAL MADRID") == "realmadrid"
    assert normalize_team_name("Arsenal") == "arsenal"


def test_normalize_team_name_accents():
    assert normalize_team_name("São Paulo") == "saopaulo"
    assert normalize_team_name("Grêmio") == "gremio"
    assert normalize_team_name("1. FC Köln") == "1fckoln"
    assert normalize_team_name("VfL Osnabrück") == "vflosnabruck"
    assert normalize_team_name("Fenerbahçe") == "fenerbahce"


def test_normalize_team_name_punctuation():
    assert normalize_team_name("A.C. Milan") == "acmilan"
    assert normalize_team_name("Paris St-Germain") == "parisstgermain"
    assert normalize_team_name("Boca Juniors / A") == "bocajuniorsa"
    assert normalize_team_name("Team_A_Reserves") == "teamareserves"
    assert normalize_team_name("O'Higgins") == "ohiggins"


def test_normalize_team_name_whitespace():
    assert normalize_team_name("  Chelsea  ") == "chelsea"
    assert normalize_team_name("Inter   Milan") == "intermilan"
    assert normalize_team_name("\tJuventus\n") == "juventus"


def test_normalize_team_name_numeric():
    assert normalize_team_name(1860) == "1860"
    assert normalize_team_name("1860 München") == "1860munchen"
    assert normalize_team_name("Schalke 04") == "schalke04"
