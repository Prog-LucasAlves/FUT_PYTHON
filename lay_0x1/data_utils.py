import ast
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from app_config import PROJECT_ROOT, UNKNOWN_TEAMS_LOG_FILE, UNKNOWN_TEAMS_RESOLVED_FILE
from lay0x1_core import normalize_team_name
from team_map import is_known_team_name, map_team_name

REQUIRED_TODAY_COLUMNS = {"Date", "Home", "Away"}


def _read_csv_file(file_path, sep=";"):
    try:
        return pd.read_csv(file_path, sep=sep)
    except Exception as exc:
        st.sidebar.warning(f"Falha ao ler {file_path.name}: {exc}")
        return None


def _validate_columns(df, required_columns, source_label):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.sidebar.warning(f"{source_label}: colunas ausentes {', '.join(missing)}")
        return False
    return True


def log_unknown_team_names(df, source_label):
    if df.empty:
        return

    unknown_names = set()
    for col in ["Home", "Away"]:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            unknown_names.update(name for name in values if name and not is_known_team_name(name))

    if not unknown_names:
        return

    UNKNOWN_TEAMS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(UNKNOWN_TEAMS_LOG_FILE, "a", encoding="utf-8") as f:
        for name in sorted(unknown_names):
            f.write(f"{stamp} | {source_label} | {name}\n")


def load_unknown_team_names(limit=50):
    if not UNKNOWN_TEAMS_LOG_FILE.exists():
        return pd.DataFrame(columns=["timestamp", "source", "name"])

    rows = []
    with open(UNKNOWN_TEAMS_LOG_FILE, encoding="utf-8") as f:
        for line in f:
            parts = [part.strip() for part in line.strip().split("|")]
            if len(parts) != 3:
                continue
            rows.append({"timestamp": parts[0], "source": parts[1], "name": parts[2]})

    if not rows:
        return pd.DataFrame(columns=["timestamp", "source", "name"])

    resolved = set(load_resolved_unknown_team_names())
    df = pd.DataFrame(rows).drop_duplicates()
    if resolved:
        df = df[~df["name"].isin(resolved)]
    df = df.tail(limit).reset_index(drop=True)
    return df


def load_resolved_unknown_team_names():
    if not UNKNOWN_TEAMS_RESOLVED_FILE.exists():
        return []
    try:
        data = json.loads(UNKNOWN_TEAMS_RESOLVED_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        return []
    return []


def save_resolved_unknown_team_names(names):
    cleaned = sorted({str(name).strip() for name in names if str(name).strip()})
    UNKNOWN_TEAMS_RESOLVED_FILE.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def load_historical_data():
    merged_path = PROJECT_ROOT / "data_total" / "dados_historico.csv"
    hist_path = PROJECT_ROOT / "data_total" / "dados_betfair.csv"
    footy_path = PROJECT_ROOT / "data_total" / "dados_footystats.csv"

    source_path = merged_path if merged_path.exists() else hist_path
    if not source_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(source_path, sep=";")
    if not _validate_columns(df, {"Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"}, "dados históricos"):
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])
    df["Norm_Home"] = df["Home"].apply(normalize_team_name)
    df["Norm_Away"] = df["Away"].apply(normalize_team_name)

    if footy_path.exists():
        try:
            df_footy = pd.read_csv(footy_path, sep=";")
            if not _validate_columns(df_footy, {"Date", "Home", "Away"}, "dados FootyStats"):
                return df
            log_unknown_team_names(df_footy, "historical_footystats")
            df_footy["Date"] = pd.to_datetime(df_footy["Date"])
            df_footy["Home"] = df_footy["Home"].apply(map_team_name)
            df_footy["Away"] = df_footy["Away"].apply(map_team_name)

            cols_to_merge = [
                "Date",
                "Home",
                "Away",
                "xG_H",
                "xG_A",
                "PPG_H_Pre",
                "PPG_A_Pre",
                "Possession_H",
                "Possession_A",
                "DangerousAttacks_H",
                "DangerousAttacks_A",
                "Shots_H",
                "Shots_A",
                "ShotsOnTarget_H",
                "ShotsOnTarget_A",
                "Corners_H",
                "Corners_A",
            ]
            cols_to_merge = [c for c in cols_to_merge if c in df_footy.columns]
            df_footy["Norm_Home"] = df_footy["Home"].apply(normalize_team_name)
            df_footy["Norm_Away"] = df_footy["Away"].apply(normalize_team_name)
            cols_to_merge_filtered = [c for c in cols_to_merge if c not in ["Home", "Away"]]
            footy_subset = df_footy[cols_to_merge_filtered + ["Norm_Home", "Norm_Away"]].copy()
            footy_subset = footy_subset.drop_duplicates(subset=["Date", "Norm_Home", "Norm_Away"], keep="last")
            missing_footy_cols = [c for c in cols_to_merge_filtered if c not in df.columns]
            if source_path == hist_path or missing_footy_cols:
                df = pd.merge(df, footy_subset, on=["Date", "Norm_Home", "Norm_Away"], how="left")
        except Exception as e:
            st.sidebar.warning(f"Erro ao mesclar FootyStats: {e}")

    def parse_minutes(x):
        try:
            if pd.isna(x) or x == "" or x == "[]":
                return []
            if isinstance(x, list):
                return x
            return ast.literal_eval(x)
        except Exception:
            return []

    df["Min_Goals_H"] = df["Min_Goals_H"].apply(parse_minutes)
    df["Min_Goals_A"] = df["Min_Goals_A"].apply(parse_minutes)
    return df


def load_today_games():
    data_day_dir = PROJECT_ROOT / "data_day"
    if not data_day_dir.exists():
        return pd.DataFrame()

    files = [f for f in data_day_dir.iterdir() if f.name.endswith(".csv") and f.name.startswith("dados_day_betfair")]
    if not files:
        return pd.DataFrame()

    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    df_list = []
    for file in files:
        df_temp = _read_csv_file(file, sep=";")
        if df_temp is None or df_temp.empty:
            continue
        if not _validate_columns(df_temp, REQUIRED_TODAY_COLUMNS, file.name):
            continue
        df_list.append(df_temp)

    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    if not _validate_columns(df, REQUIRED_TODAY_COLUMNS, "dados do dia concatenados"):
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Home"] = df["Home"].apply(map_team_name)
    df["Away"] = df["Away"].apply(map_team_name)
    log_unknown_team_names(df, "data_day")
    odds_cols = [c for c in df.columns if "Odd_" in c]
    df["non_zero_count"] = (df[odds_cols] > 0).sum(axis=1)
    df = df.sort_values("non_zero_count", ascending=False)
    df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
    df = df.drop(columns=["non_zero_count"])
    df["Norm_Home"] = df["Home"].apply(normalize_team_name)
    df["Norm_Away"] = df["Away"].apply(normalize_team_name)
    df["Match"] = df["Home"].astype(str) + " vs " + df["Away"].astype(str)
    return df
