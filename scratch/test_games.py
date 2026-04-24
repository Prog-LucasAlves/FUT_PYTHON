import sys
from pathlib import Path

import pandas as pd

# Add the project dir to sys.path to import modules
sys.path.append(str(Path(r"d:\Sports\FUT_PYTHON\lay_0x1")))

import data_utils


def test():
    df_today = data_utils.load_today_games()
    print(f"Total today games: {len(df_today)}")

    date_selected = pd.to_datetime("2026-04-23").date()
    df_day_filtered = df_today[df_today["Date"].dt.date == date_selected]
    print(f"Games for {date_selected}: {len(df_day_filtered)}")
    print(df_day_filtered[["Match", "League"]])


if __name__ == "__main__":
    test()
