import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import poisson


def normalize_team_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    name = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    name = re.sub(r"[\.\-_/]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"[^a-z0-9]", "", name)


def normalize_goal_minute(minute):
    if pd.isna(minute):
        return None
    if isinstance(minute, (int, float)):
        return float(minute)
    minute_str = str(minute).strip().replace("'", "")
    if not minute_str:
        return None
    try:
        if "+" in minute_str:
            base, extra = minute_str.split("+", 1)
            return float(base) + float(extra)
        return float(minute_str)
    except Exception:
        return None


def count_goals_until(mins_list, minute_limit):
    valid_minutes = [m for m in (normalize_goal_minute(m) for m in mins_list) if m is not None]
    return sum(1 for m in valid_minutes if m <= minute_limit)


def count_goals_after(mins_list, minute_limit):
    valid_minutes = [m for m in (normalize_goal_minute(m) for m in mins_list) if m is not None]
    return sum(1 for m in valid_minutes if m > minute_limit)


def format_minutes(decimal_min):
    if pd.isna(decimal_min) or decimal_min == 0:
        return "N/A"
    minutes = int(decimal_min)
    seconds = int((decimal_min - minutes) * 60)
    return f"{minutes}'{seconds:02d}\""


def get_last_10_team_summary(
    df_games,
    team_name,
    target_score,
    normalize_team_name_fn=normalize_team_name,
):
    team_norm = normalize_team_name_fn(team_name)
    team_games = df_games[(df_games["Norm_Home"] == team_norm) | (df_games["Norm_Away"] == team_norm)].copy()
    if team_games.empty:
        return {
            "form_sequence": "Sem jogos",
            "record": "0V 0E 0D",
            "points": 0,
            "max_points": 0,
            "win_rate": 0.0,
            "target_score_count": 0,
            "games_analyzed": 0,
        }

    team_games = team_games.sort_values("Date", ascending=False).head(10).copy()

    def summarize_match(row):
        if row["Norm_Home"] == team_norm:
            goals_for = int(row["Goals_H_FT"])
            goals_against = int(row["Goals_A_FT"])
        else:
            goals_for = int(row["Goals_A_FT"])
            goals_against = int(row["Goals_H_FT"])

        if goals_for > goals_against:
            result = "V"
            points = 3
        elif goals_for == goals_against:
            result = "E"
            points = 1
        else:
            result = "D"
            points = 0

        return goals_for, goals_against, result, points

    summaries = team_games.apply(summarize_match, axis=1)
    goals_for = summaries.apply(lambda x: x[0])
    goals_against = summaries.apply(lambda x: x[1])
    results = summaries.apply(lambda x: x[2])
    points = summaries.apply(lambda x: x[3])

    wins = int((results == "V").sum())
    draws = int((results == "E").sum())
    losses = int((results == "D").sum())
    total_points = int(points.sum())
    total_games = len(team_games)
    max_points = total_games * 3

    def build_ht_scenario_summary(df, ht_score):
        def match_ht_score(row):
            if row["Norm_Home"] == team_norm:
                return row["Goals_H_HT"] == ht_score[0] and row["Goals_A_HT"] == ht_score[1]
            return row["Goals_A_HT"] == ht_score[0] and row["Goals_H_HT"] == ht_score[1]

        scenario_games = df[df.apply(match_ht_score, axis=1)].copy()
        if scenario_games.empty:
            return {
                "total": 0,
                "home_goal_to_75": 0,
                "away_goal_to_75": 0,
                "stayed_score_to_75": 0,
                "changed_score_to_75": 0,
            }

        def score_until_75(row):
            if row["Norm_Home"] == team_norm:
                return count_goals_until(row["Min_Goals_H"], 75), count_goals_until(
                    row["Min_Goals_A"],
                    75,
                )
            return count_goals_until(row["Min_Goals_A"], 75), count_goals_until(
                row["Min_Goals_H"],
                75,
            )

        scores_75 = scenario_games.apply(score_until_75, axis=1)
        team_scores_75 = scores_75.apply(lambda x: x[0])
        opp_scores_75 = scores_75.apply(lambda x: x[1])
        stayed_mask = (team_scores_75 == ht_score[0]) & (opp_scores_75 == ht_score[1])

        return {
            "total": int(len(scenario_games)),
            "home_goal_to_75": int((team_scores_75 > ht_score[0]).sum()),
            "away_goal_to_75": int((opp_scores_75 > ht_score[1]).sum()),
            "stayed_score_to_75": int(stayed_mask.sum()),
            "changed_score_to_75": int((~stayed_mask).sum()),
        }

    return {
        "form_sequence": " | ".join(results.tolist()),
        "record": f"{wins}V {draws}E {losses}D",
        "points": total_points,
        "max_points": max_points,
        "win_rate": (wins / total_games) * 100 if total_games > 0 else 0.0,
        "target_score_count": int(((goals_for == target_score[0]) & (goals_against == target_score[1])).sum()),
        "games_analyzed": total_games,
        "ht_00": build_ht_scenario_summary(team_games, (0, 0)),
        "ht_01": build_ht_scenario_summary(team_games, (0, 1)),
    }


def build_risk_plan(avg_match_2h, exit_minute, sample_quality, score, recommendation):
    if pd.isna(avg_match_2h):
        base_exit = exit_minute
    else:
        base_exit = min(int(avg_match_2h + 5), 80)
    if sample_quality == "Boa":
        confidence = "Maior"
        buffer_minutes = 0
    elif sample_quality == "Moderada":
        confidence = "Média"
        buffer_minutes = 2
    else:
        confidence = "Baixa"
        buffer_minutes = 4
    adjusted_exit = min(base_exit + buffer_minutes, 80)
    if score >= 11:
        risk_band = "Agressivo"
        hedge_note = "Aceita manter até o limite, mas evita esticar além do minuto alvo."
        stop_note = "Se o placar abrir contra e a odd alongar muito, sair sem esperar confirmação extra."
    elif score >= 7:
        risk_band = "Moderado"
        hedge_note = "Buscar hedge ao redor do minuto alvo e reduzir exposição se o jogo travar."
        stop_note = "Se o 0x1 ficar cada vez mais provável, não insistir."
    else:
        risk_band = "Conservador"
        hedge_note = "Preferir redução de risco mais cedo, sobretudo em jogos com baixa amostra."
        stop_note = "Se não houver pressão clara, priorizar saída antecipada."
    return {
        "base_exit": base_exit,
        "adjusted_exit": adjusted_exit,
        "confidence": confidence,
        "risk_band": risk_band,
        "hedge_note": hedge_note,
        "stop_note": stop_note,
        "recommendation": recommendation,
    }


def get_h2h_stats(df_games, home_team, away_team, normalize_team_name_fn=normalize_team_name):
    norm_h = normalize_team_name_fn(home_team)
    norm_a = normalize_team_name_fn(away_team)
    h2h = df_games[((df_games["Norm_Home"] == norm_h) & (df_games["Norm_Away"] == norm_a)) | ((df_games["Norm_Home"] == norm_a) & (df_games["Norm_Away"] == norm_h))].copy()
    if h2h.empty:
        return None
    h2h = h2h.sort_values("Date", ascending=False)
    total = len(h2h)
    score_0x1 = len(h2h[(h2h["Goals_H_FT"] == 0) & (h2h["Goals_A_FT"] == 1)])
    score_1x0 = len(h2h[(h2h["Goals_H_FT"] == 1) & (h2h["Goals_A_FT"] == 0)])
    score_0x0 = len(h2h[(h2h["Goals_H_FT"] == 0) & (h2h["Goals_A_FT"] == 0)])
    ft_scores = h2h["Goals_H_FT"].astype(int).astype(str) + "x" + h2h["Goals_A_FT"].astype(int).astype(str)
    top_scores = (ft_scores.value_counts(normalize=True) * 100).head(5).to_dict()
    return {"total": total, "score_0x1": score_0x1, "score_0x1_pct": (score_0x1 / total) * 100, "score_1x0": score_1x0, "score_0x0": score_0x0, "top_scores": top_scores, "games": h2h}


def get_goal_interval_stats(df_games, home_team, away_team, normalize_team_name_fn=normalize_team_name, normalize_goal_minute_fn=normalize_goal_minute):
    norm_h = normalize_team_name_fn(home_team)
    norm_a = normalize_team_name_fn(away_team)
    h_games = df_games[(df_games["Norm_Home"] == norm_h) | (df_games["Norm_Away"] == norm_h)].copy()
    a_games = df_games[(df_games["Norm_Home"] == norm_a) | (df_games["Norm_Away"] == norm_a)].copy()
    intervals = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]

    def ensure_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, (set, dict)):
            return list(x)
        return []

    def calc_interval_pct(games, team_norm):
        result = {}
        total = len(games)
        if total == 0:
            return {f"{a}-{b}'": 0.0 for a, b in intervals}
        for start, end in intervals:
            label = f"{start}-{end}'"
            count = games.apply(
                lambda row: any((m := normalize_goal_minute_fn(x)) is not None and start < m <= end for x in ensure_list(row["Min_Goals_H"] if row["Norm_Home"] == team_norm else row["Min_Goals_A"])),
                axis=1,
            ).sum()
            result[label] = (count / total) * 100
        return result

    def calc_combined(games):
        result = {}
        total = len(games)
        if total == 0:
            return {f"{a}-{b}'": 0.0 for a, b in intervals}
        for start, end in intervals:
            label = f"{start}-{end}'"
            count = games.apply(lambda row: any((m := normalize_goal_minute_fn(x)) is not None and start < m <= end for x in (ensure_list(row["Min_Goals_H"]) + ensure_list(row["Min_Goals_A"]))), axis=1).sum()
            result[label] = (count / total) * 100
        return result

    return {"home_attack": calc_interval_pct(h_games, norm_h), "away_attack": calc_interval_pct(a_games, norm_a), "home_combined": calc_combined(h_games), "away_combined": calc_combined(a_games), "home_sample": len(h_games), "away_sample": len(a_games)}


def build_poisson_timing_scenario(df_games, team_name, role, scenario_score, cutoff_minute=75, end_minute=90, normalize_team_name_fn=normalize_team_name, count_goals_until_fn=count_goals_until, count_goals_after_fn=count_goals_after):
    team_norm = normalize_team_name_fn(team_name)
    role_col = "Norm_Home" if role == "home" else "Norm_Away"
    team_games = df_games[df_games[role_col] == team_norm].copy()
    if team_games.empty:
        return None
    h_goals = team_games["Min_Goals_H"].apply(lambda mins: count_goals_until_fn(mins, cutoff_minute))
    a_goals = team_games["Min_Goals_A"].apply(lambda mins: count_goals_until_fn(mins, cutoff_minute))
    mask = (h_goals == scenario_score[0]) & (a_goals == scenario_score[1])
    scenario_df = team_games[mask]
    if scenario_df.empty:
        return None
    future_home_goals = scenario_df["Min_Goals_H"].apply(lambda mins: count_goals_after_fn(mins, cutoff_minute))
    future_away_goals = scenario_df["Min_Goals_A"].apply(lambda mins: count_goals_after_fn(mins, cutoff_minute))
    future_total_goals = future_home_goals + future_away_goals
    lambda_home = future_home_goals.mean()
    lambda_away = future_away_goals.mean()
    lambda_total = future_total_goals.mean()
    minute_axis = list(range(cutoff_minute + 1, end_minute + 1))
    remaining_window = max(end_minute - cutoff_minute, 1)
    timeline = []
    for minute in minute_axis:
        elapsed = minute - cutoff_minute
        fraction = elapsed / remaining_window
        mu_home = lambda_home * fraction
        mu_away = lambda_away * fraction
        mu_total = lambda_total * fraction
        timeline.append({"Minute": minute, "Home": (1 - poisson.pmf(0, mu_home)) * 100, "Away": (1 - poisson.pmf(0, mu_away)) * 100, "Match": (1 - poisson.pmf(0, mu_total)) * 100})
    return {
        "sample_size": len(scenario_df),
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "lambda_total": lambda_total,
        "prob_home_goal": (1 - poisson.pmf(0, lambda_home)) * 100,
        "prob_away_goal": (1 - poisson.pmf(0, lambda_away)) * 100,
        "prob_match_goal": (1 - poisson.pmf(0, lambda_total)) * 100,
        "timeline": pd.DataFrame(timeline),
        "scenario_label": f"{scenario_score[0]}x{scenario_score[1]} aos {cutoff_minute}'",
    }


def analyze_goal_timing(df_games, home_team, away_team, normalize_team_name_fn=normalize_team_name, normalize_goal_minute_fn=normalize_goal_minute):
    norm_h = normalize_team_name_fn(home_team)
    norm_a = normalize_team_name_fn(away_team)
    h_games = df_games[(df_games["Norm_Home"] == norm_h) | (df_games["Norm_Away"] == norm_h)].copy()
    a_games = df_games[(df_games["Norm_Home"] == norm_a) | (df_games["Norm_Away"] == norm_a)].copy()

    def get_frequent_scores(games, team_norm):
        def process_scores(df, suffix):
            def get_score_view(row):
                gh = row[f"Goals_H_{suffix}"]
                ga = row[f"Goals_A_{suffix}"]
                if pd.isnull(gh) or pd.isnull(ga):
                    return "N/A"
                if row["Norm_Home"] == team_norm:
                    return f"{int(gh)}x{int(ga)}"
                return f"{int(ga)}x{int(gh)}"

            scores = df.apply(get_score_view, axis=1)
            return (scores.value_counts(normalize=True) * 100).head(5).to_dict()

        return {"HT": process_scores(games, "HT"), "FT": process_scores(games, "FT")}

    def get_timing_stats(games, team_norm):
        def ensure_list(x):
            if isinstance(x, list):
                return x
            if isinstance(x, (set, dict)):
                return list(x)
            return []

        def normalize_mins_list(mins_list):
            return [m for m in (normalize_goal_minute_fn(m) for m in mins_list) if m is not None]

        def get_team_mins(row):
            mins = row["Min_Goals_H"] if row["Norm_Home"] == team_norm else row["Min_Goals_A"]
            return normalize_mins_list(ensure_list(mins))

        team_mins = games.apply(get_team_mins, axis=1)
        first_goal_team = team_mins.apply(lambda x: min(x) if len(x) > 0 else None).dropna()

        def get_match_mins(row):
            combined = ensure_list(row["Min_Goals_H"]) + ensure_list(row["Min_Goals_A"])
            return normalize_mins_list(combined)

        match_mins = games.apply(get_match_mins, axis=1)
        first_goal_match = match_mins.apply(lambda x: min(x) if len(x) > 0 else None).dropna()

        games_00_ht = games[(games["Goals_H_HT"] == 0) & (games["Goals_A_HT"] == 0)]

        def first_in_2h(mins_list):
            m2h = [m for m in mins_list if m > 45]
            return min(m2h) if len(m2h) > 0 else None

        first_team_2h = games_00_ht.apply(get_team_mins, axis=1).apply(first_in_2h).dropna()
        first_match_2h = games_00_ht.apply(get_match_mins, axis=1).apply(first_in_2h).dropna()

        return {
            "avg_first_team": float(first_goal_team.mean()) if not first_goal_team.empty else None,
            "avg_first_match": float(first_goal_match.mean()) if not first_goal_match.empty else None,
            "avg_team_2h_00ht": float(first_team_2h.mean()) if not first_team_2h.empty else None,
            "avg_match_2h_00ht": float(first_match_2h.mean()) if not first_match_2h.empty else None,
            "sample_size": len(games),
            "sample_00ht": len(games_00_ht),
            "raw_first_team": [int(x) for x in first_goal_team.tolist()],
        }

    if len(h_games) == 0 or len(a_games) == 0:
        return None, None, None
    stats_h = get_timing_stats(h_games, norm_h)
    stats_a = get_timing_stats(a_games, norm_a)
    stats_h["frequent_scores"] = get_frequent_scores(h_games, norm_h)
    stats_a["frequent_scores"] = get_frequent_scores(a_games, norm_a)

    def get_combined_score_stats(h_df, a_df, suffix):
        combined = pd.concat([h_df, a_df])
        valid = combined.dropna(subset=[f"Goals_H_{suffix}", f"Goals_A_{suffix}"])
        if valid.empty:
            return {}
        scores = valid[f"Goals_H_{suffix}"].astype(int).astype(str) + "x" + valid[f"Goals_A_{suffix}"].astype(int).astype(str)
        return (scores.value_counts(normalize=True).head(5) * 100).to_dict()

    stats_combined_scores = {
        "HT": get_combined_score_stats(h_games, a_games, "HT"),
        "FT": get_combined_score_stats(h_games, a_games, "FT"),
    }

    return stats_h, stats_a, stats_combined_scores


def build_team_role_profile(df_games, team_name, role, normalize_team_name_fn=normalize_team_name):
    team_norm = normalize_team_name_fn(team_name)
    role_col = "Norm_Home" if role == "home" else "Norm_Away"
    games = df_games[df_games[role_col] == team_norm].copy()
    if games.empty:
        return {
            "sample_size": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "ppg_season": 0.0,
            "ppg_last_10": 0.0,
            "ppg_last_5": 0.0,
            "goals_for": 0,
            "goals_against": 0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "first_goal": {},
        }

    games = games.sort_values("Date", ascending=False).copy()

    def get_team_scores(row):
        if role == "home":
            gf = int(row["Goals_H_FT"])
            ga = int(row["Goals_A_FT"])
            mins_for = row["Min_Goals_H"] if isinstance(row["Min_Goals_H"], list) else []
            mins_against = row["Min_Goals_A"] if isinstance(row["Min_Goals_A"], list) else []
        else:
            gf = int(row["Goals_A_FT"])
            ga = int(row["Goals_H_FT"])
            mins_for = row["Min_Goals_A"] if isinstance(row["Min_Goals_A"], list) else []
            mins_against = row["Min_Goals_H"] if isinstance(row["Min_Goals_H"], list) else []
        return gf, ga, mins_for, mins_against

    score_data = games.apply(get_team_scores, axis=1)
    goals_for = score_data.apply(lambda x: x[0])
    goals_against = score_data.apply(lambda x: x[1])

    result_series = goals_for.combine(goals_against, lambda gf, ga: "W" if gf > ga else "D" if gf == ga else "L")
    points_series = result_series.map({"W": 3, "D": 1, "L": 0})

    def ppg(subset):
        return float(points_series.loc[subset.index].mean()) if not subset.empty else 0.0

    def first_goal_flags(row):
        gf, ga, mins_for, mins_against = get_team_scores(row)
        team_first = None
        opp_first = None
        if mins_for:
            team_minutes = [m for m in (normalize_goal_minute(x) for x in mins_for) if m is not None]
            if team_minutes:
                team_first = min(team_minutes)
        if mins_against:
            opp_minutes = [m for m in (normalize_goal_minute(x) for x in mins_against) if m is not None]
            if opp_minutes:
                opp_first = min(opp_minutes)
        if team_first is None and opp_first is None:
            return {
                "scored_first": False,
                "suffered_first": False,
                "scored_first_won": False,
                "scored_first_draw": False,
                "scored_first_lost": False,
                "suffered_first_won": False,
                "suffered_first_draw": False,
                "suffered_first_lost": False,
                "scored_first_ht": False,
                "scored_first_ht_won": False,
            }
        scored_first = team_first is not None and (opp_first is None or team_first < opp_first)
        suffered_first = opp_first is not None and (team_first is None or opp_first < team_first)
        won = gf > ga
        draw = gf == ga
        lost = gf < ga
        scored_first_ht = scored_first and team_first <= 45
        return {
            "scored_first": scored_first,
            "suffered_first": suffered_first,
            "scored_first_won": scored_first and won,
            "scored_first_draw": scored_first and draw,
            "scored_first_lost": scored_first and lost,
            "suffered_first_won": suffered_first and won,
            "suffered_first_draw": suffered_first and draw,
            "suffered_first_lost": suffered_first and lost,
            "scored_first_ht": scored_first_ht,
            "scored_first_ht_won": scored_first_ht and won,
        }

    first_goal_df = games.apply(first_goal_flags, axis=1, result_type="expand")
    total_games = len(games)
    wins = int((result_series == "W").sum())
    draws = int((result_series == "D").sum())
    losses = int((result_series == "L").sum())
    points = int(points_series.sum())

    recent_10 = games.head(10)
    recent_5 = games.head(5)
    first_goal_scored = first_goal_df["scored_first"]
    first_goal_suffered = first_goal_df["suffered_first"]
    first_goal_scored_count = int(first_goal_scored.sum())
    first_goal_suffered_count = int(first_goal_suffered.sum())
    first_goal_ht = first_goal_df["scored_first_ht"]

    def pct(num, den=total_games):
        return (float(num) / den * 100) if den else 0.0

    return {
        "sample_size": total_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "ppg_season": (points / total_games) if total_games else 0.0,
        "ppg_last_10": ppg(recent_10),
        "ppg_last_5": ppg(recent_5),
        "goals_for": int(goals_for.sum()),
        "goals_against": int(goals_against.sum()),
        "avg_goals_for": float(goals_for.mean()),
        "avg_goals_against": float(goals_against.mean()),
        "first_goal": {
            "scored_first_pct": pct(first_goal_scored_count),
            "scored_first_won_pct": pct(first_goal_df["scored_first_won"].sum()),
            "scored_first_draw_pct": pct(first_goal_df["scored_first_draw"].sum()),
            "scored_first_lost_pct": pct(first_goal_df["scored_first_lost"].sum()),
            "suffered_first_pct": pct(first_goal_suffered_count),
            "suffered_first_won_pct": pct(first_goal_df["suffered_first_won"].sum()),
            "suffered_first_draw_pct": pct(first_goal_df["suffered_first_draw"].sum()),
            "suffered_first_lost_pct": pct(first_goal_df["suffered_first_lost"].sum()),
            "first_goal_first_half_pct": pct(first_goal_ht.sum()),
            "first_goal_first_half_won_pct": pct(first_goal_df["scored_first_ht_won"].sum()),
        },
    }


def calculate_pro_metrics(df_games, home_team, away_team, current_match_data, normalize_team_name_fn=normalize_team_name):
    norm_h = normalize_team_name_fn(home_team)
    norm_a = normalize_team_name_fn(away_team)
    home_h = df_games[(df_games["Norm_Home"] == norm_h) | (df_games["Norm_Away"] == norm_h)].copy()
    away_a = df_games[(df_games["Norm_Home"] == norm_a) | (df_games["Norm_Away"] == norm_a)].copy()
    if len(home_h) < 1 or len(away_a) < 1:
        return None

    def get_stats(games, team_norm):
        def get_team_val(row, col_h, col_a):
            col = col_h if row["Norm_Home"] == team_norm else col_a
            return row[col] if col in row.index else np.nan

        def get_team_series(col_h, col_a):
            return games.apply(lambda r: get_team_val(r, col_h, col_a), axis=1)

        goals = games.apply(lambda r: get_team_val(r, "Goals_H_FT", "Goals_A_FT"), axis=1)
        goals_ht = games.apply(lambda r: get_team_val(r, "Goals_H_HT", "Goals_A_HT"), axis=1)
        mins = games.apply(lambda r: get_team_val(r, "Min_Goals_H", "Min_Goals_A"), axis=1)
        mean_goals = goals.mean()
        mean_goals_ht = goals_ht.mean()
        variance = goals.var()
        first_goal_mins = mins.apply(lambda x: normalize_goal_minute(list(x)[0]) if (isinstance(x, (list, set, tuple)) and len(x) > 0) else None).dropna()

        avg_first_goal = first_goal_mins.mean() if not first_goal_mins.empty else 0
        cost_of_goal = (variance / (mean_goals + 0.001)) if mean_goals > 0 else 0
        avg_xg = games.apply(lambda r: get_team_val(r, "xG_H", "xG_A"), axis=1).mean()
        avg_ppg = games.apply(lambda r: get_team_val(r, "PPG_H_Pre", "PPG_A_Pre"), axis=1).mean()
        avg_da = games.apply(lambda r: get_team_val(r, "DangerousAttacks_H", "DangerousAttacks_A"), axis=1).mean()
        shots_on_target = pd.to_numeric(get_team_series("ShotsOnTarget_H", "ShotsOnTarget_A"), errors="coerce")
        total_shots = pd.to_numeric(get_team_series("Shots_H", "Shots_A"), errors="coerce")
        total_goals = goals.sum()
        valid_sot = shots_on_target.dropna()
        valid_total_shots = total_shots.dropna()
        if not valid_sot.empty and valid_sot.sum() > 0:
            finishing_value = valid_sot.sum() / total_goals if total_goals > 0 else np.nan
            finishing_label = "Chutes no Gol/Gol"
            finishing_desc = "Média de chutes no gol necessários para marcar 1 gol"
        elif not valid_total_shots.empty and valid_total_shots.sum() > 0:
            finishing_value = valid_total_shots.sum() / total_goals if total_goals > 0 else np.nan
            finishing_label = "Chutes/Gol"
            finishing_desc = "Média de chutes necessários para marcar 1 gol"
        else:
            finishing_value = np.nan
            finishing_label = "Chutes no Gol/Gol"
            finishing_desc = "Dados de finalização indisponíveis no histórico"
        return {
            "mean": mean_goals,
            "mean_ht": mean_goals_ht,
            "variance": variance,
            "cost": cost_of_goal,
            "zeros": (len(games[goals == 0]) / len(games)) * 100,
            "over15": (len(games[goals > 1.5]) / len(games)) * 100,
            "avg_first_goal": avg_first_goal,
            "total_games": len(games),
            "avg_xg": np.nan_to_num(avg_xg),
            "avg_ppg": np.nan_to_num(avg_ppg),
            "avg_da": np.nan_to_num(avg_da),
            "shots_per_goal": finishing_value,
            "shots_per_goal_label": finishing_label,
            "shots_per_goal_desc": finishing_desc,
            "clean_sheet_pct": (games.apply(lambda r: (r["Goals_A_FT"] == 0) if r["Norm_Home"] == team_norm else (r["Goals_H_FT"] == 0), axis=1).sum() / len(games)) * 100,
        }

    def capped_score(value, max_value, invert=False, default=50):
        if pd.isna(value):
            return default
        score = max(0, min((float(value) / max_value) * 100, 100))
        return 100 - score if invert else score

    def calculate_lay_strength_index(stats):
        shots_value = stats["shots_per_goal"]
        finishing_score = 100 - min((shots_value / 6) * 100, 100) if pd.notna(shots_value) else 50
        components = {"xg": capped_score(stats["avg_xg"], 2.5), "ppg": capped_score(stats["avg_ppg"], 3.0), "da": capped_score(stats["avg_da"], 80), "zero_avoid": max(0, 100 - stats["zeros"]), "over15": max(0, min(stats["over15"], 100)), "finishing": finishing_score}
        index = components["xg"] * 0.30 + components["zero_avoid"] * 0.20 + components["over15"] * 0.20 + components["da"] * 0.15 + components["ppg"] * 0.10 + components["finishing"] * 0.05
        values = list(components.values())
        mean_comp = sum(values) / len(values)
        variance = sum((v - mean_comp) ** 2 for v in values) / len(values)
        return round(index, 1), round(variance, 1)

    def classify_strength(index):
        if index >= 75:
            return "Elite"
        if index >= 60:
            return "Forte"
        if index >= 45:
            return "Moderado"
        return "Baixo"

    def classify_variance(variance):
        if variance < 200:
            return "Consistente"
        if variance < 500:
            return "Irregular"
        return "Volátil"

    def describe_variance(variance_label, team_role):
        descriptions = {
            "Consistente": (f"Os componentes do {team_role} estão alinhados — xG, pressão, finalizações e capacidade de marcar apontam na mesma direção. Sinal confiável: o índice reflete um perfil real e coerente."),
            "Irregular": (f"O {team_role} apresenta componentes mistos — algumas métricas são fortes, outras fracas. O índice deve ser interpretado com cautela e confirmado por outros indicadores."),
            "Volátil": (f"Componentes muito discrepantes no {team_role}: métricas opostas coexistem. O índice pode ser enganoso — exige atenção redobrada antes de apostar no Lay 0x1."),
        }
        return descriptions.get(variance_label, "")

    stats_h = get_stats(home_h, norm_h)
    stats_a = get_stats(away_a, norm_a)
    last10_home = get_last_10_team_summary(df_games, home_team, (0, 1), normalize_team_name_fn)
    last10_away = get_last_10_team_summary(df_games, away_team, (0, 1), normalize_team_name_fn)
    role_profile_home = build_team_role_profile(df_games, home_team, "home", normalize_team_name_fn)
    role_profile_away = build_team_role_profile(df_games, away_team, "away", normalize_team_name_fn)
    lay01_index_home, lay01_var_home = calculate_lay_strength_index(stats_h)
    lay01_index_away, lay01_var_away = calculate_lay_strength_index(stats_a)
    prob_h0_ht = poisson.pmf(0, stats_h["mean_ht"])
    prob_a1_ht = poisson.pmf(1, stats_a["mean_ht"])
    poisson_0x1_ht = (prob_h0_ht * prob_a1_ht) * 100

    def _normalize_goals(games, team_norm, suffix):
        """Return (team_goals, opp_goals) series for the given suffix (HT or FT), normalized to team perspective."""

        def get_int(val):
            try:
                if pd.isna(val):
                    return 0
                return int(float(val))
            except:
                return 0

        team_g = games.apply(lambda r: get_int(r[f"Goals_H_{suffix}"]) if r["Norm_Home"] == team_norm else get_int(r[f"Goals_A_{suffix}"]), axis=1)
        opp_g = games.apply(lambda r: get_int(r[f"Goals_A_{suffix}"]) if r["Norm_Home"] == team_norm else get_int(r[f"Goals_H_{suffix}"]), axis=1)
        return team_g, opp_g

    def analyze_ht_scenarios(games_h, games_a, norm_h_val, norm_a_val):
        # Normalize perspective: 0x1 means team_scored=0, opponent_scored=1
        h_team_ht, h_opp_ht = _normalize_goals(games_h, norm_h_val, "HT")
        h_team_ft, h_opp_ft = _normalize_goals(games_h, norm_h_val, "FT")
        a_team_ht, a_opp_ht = _normalize_goals(games_a, norm_a_val, "HT")
        a_team_ft, a_opp_ft = _normalize_goals(games_a, norm_a_val, "FT")

        # 0x0 HT scenarios (from team perspective)
        h_00_mask = (h_team_ht == 0) & (h_opp_ht == 0)
        a_00_mask = (a_team_ht == 0) & (a_opp_ht == 0)
        total_00_ht = int(h_00_mask.sum()) + int(a_00_mask.sum())
        # Red = FT ended 0x1 (team 0, opponent 1)
        h_red_00 = int(((h_team_ft == 0) & (h_opp_ft == 1) & h_00_mask).sum())
        a_red_00 = int(((a_team_ft == 0) & (a_opp_ft == 1) & a_00_mask).sum())
        prob_red_from_00 = ((h_red_00 + a_red_00) / total_00_ht * 100) if total_00_ht > 0 else 0

        # 0x1 HT scenarios
        h_01_mask = (h_team_ht == 0) & (h_opp_ht == 1)
        a_01_mask = (a_team_ht == 0) & (a_opp_ht == 1)
        total_01_ht = int(h_01_mask.sum()) + int(a_01_mask.sum())
        h_red_01 = int(((h_team_ft == 0) & (h_opp_ft == 1) & h_01_mask).sum())
        a_red_01 = int(((a_team_ft == 0) & (a_opp_ft == 1) & a_01_mask).sum())
        prob_red_from_01 = ((h_red_01 + a_red_01) / total_01_ht * 100) if total_01_ht > 0 else 0
        return prob_red_from_00, prob_red_from_01

    h2h_stats = get_h2h_stats(df_games, home_team, away_team, normalize_team_name_fn)
    red_00, red_01 = analyze_ht_scenarios(home_h, away_a, norm_h, norm_a)

    # Normalize combined perspective for HT/FT percentages
    def _build_normalized_scores(games, team_norm, suffix):
        team_g, opp_g = _normalize_goals(games, team_norm, suffix)
        return team_g.astype(str) + "x" + opp_g.astype(str)

    h_ht_scores = _build_normalized_scores(home_h, norm_h, "HT")
    a_ht_scores = _build_normalized_scores(away_a, norm_a, "HT")
    h_ft_scores = _build_normalized_scores(home_h, norm_h, "FT")
    a_ft_scores = _build_normalized_scores(away_a, norm_a, "FT")
    all_ht_scores = pd.concat([h_ht_scores, a_ht_scores], ignore_index=True)
    all_ft_scores = pd.concat([h_ft_scores, a_ft_scores], ignore_index=True)
    total_combined = len(all_ht_scores)
    if total_combined > 0:
        pct_00_ht = float((all_ht_scores == "0x0").mean() * 100)
        pct_01_ht = float((all_ht_scores == "0x1").mean() * 100)
        pct_00_ft = float((all_ft_scores == "0x0").mean() * 100)
        pct_01_ft = float((all_ft_scores == "0x1").mean() * 100)
    else:
        pct_00_ht = pct_01_ht = pct_00_ft = pct_01_ft = 0.0
    pct_other_ht = max(0.0, 100.0 - pct_00_ht - pct_01_ht)
    pct_other_ft = max(0.0, 100.0 - pct_00_ft - pct_01_ft)
    sample_home = len(home_h)
    sample_away = len(away_a)
    min_sample = min(sample_home, sample_away)
    sample_quality = "Boa" if min_sample >= 50 else "Moderada" if min_sample >= 25 else "Pequena"
    score = 0
    reasons = []
    odd_h_back = current_match_data.get("Odd_H_Back", 0)
    odd_lay_0x1 = current_match_data.get("Odd_CS_0x1_Lay", 0)
    odd_btts = current_match_data.get("Odd_BTTS_Yes_Back", 0)
    odd_over25 = current_match_data.get("Odd_Over25_FT_Back", 0)

    # --- LÓGICA DE BINS (BASEADO NO TEBF005.PY) ---
    def get_bin(val, bins, labels):
        for i in range(len(bins) - 1):
            if bins[i] < val <= bins[i + 1]:
                return labels[i]
        return labels[-1]

    # Média de gols do mandante APENAS em casa (para o Bin_Avg_H)
    home_only_games = home_h[home_h["Norm_Home"] == norm_h]
    home_avg_at_home = home_only_games["Goals_H_FT"].mean() if not home_only_games.empty else stats_h["mean"]

    bin_h = get_bin(odd_h_back, [1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 100], ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.1-2.5", "2.6-3.0", "3.0+"])
    bin_over = get_bin(odd_over25, [0, 1.6, 1.8, 2.0, 100], ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
    bin_btts = get_bin(odd_btts, [0, 1.6, 1.8, 2.0, 100], ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
    bin_lay = get_bin(odd_lay_0x1, [0, 10, 15, 20, 30, 100], ["<10", "10-15", "15-20", "20-30", "30+"])
    bin_avg_h = get_bin(home_avg_at_home, [0, 1.2, 1.5, 1.8, 5.0], ["<1.2", "1.2-1.5", "1.5-1.8", "1.8+"])

    winning_brackets = [
        ("2.1-2.5", "2.0+", "1.8-2.0", "10-15", "1.2-1.5"),
        ("1.7-2.0", "2.0+", "2.0+", "15-20", "1.2-1.5"),
        ("2.1-2.5", "1.6-1.8", "1.6-1.8", "15-20", "1.8+"),
        ("2.1-2.5", "1.8-2.0", "1.6-1.8", "10-15", "1.2-1.5"),
        ("<1.3", "<1.6", "1.8-2.0", "30+", "1.8+"),
        ("1.5-1.7", "2.0+", "2.0+", "15-20", "1.5-1.8"),
        ("1.3-1.5", "1.8-2.0", "2.0+", "20-30", "1.5-1.8"),
        ("3.0+", "1.8-2.0", "1.6-1.8", "10-15", "1.8+"),
        ("1.3-1.5", "<1.6", "1.8-2.0", "30+", "1.8+"),
        ("3.0+", "2.0+", "2.0+", "<10", "1.5-1.8"),
    ]

    is_portfolio_match = (bin_h, bin_over, bin_btts, bin_lay, bin_avg_h) in winning_brackets

    # 1. Padrão de Portfolio - Golden Branch (Peso 8)
    # Reflete o match completo dos 5 Bins (Odd H, Over, BTTS, Lay, Avg Goals H)
    if is_portfolio_match:
        score += 8
        reasons.append("Match de Portfolio Detectado (Golden Branch): DNA lucrativo confirmado (Bins)")

    # 2. Heurística Poisson HT (Peso 2)
    if poisson_0x1_ht < 7:
        score += 2
        reasons.append(f"Baixa probabilidade Poisson no HT ({poisson_0x1_ht:.1f}%)")
    elif poisson_0x1_ht < 12:
        score += 1
        reasons.append(f"Heurística Poisson Moderada no HT ({poisson_0x1_ht:.1f}%)")

    # 3. H2H (Peso 1)
    if h2h_stats and h2h_stats.get("score_0x1_pct", 100) <= 10:
        score += 1
        reasons.append(f"Histórico Direto Favorável (0x1 em apenas {h2h_stats['score_0x1_pct']:.1f}% dos jogos)")

    # 4. Clean Sheet Mandante (Peso 1)
    if stats_h["clean_sheet_pct"] >= 25:
        score += 1
        reasons.append(f"Clean Sheet Mandante Robusto ({stats_h['clean_sheet_pct']:.1f}%): Dificuldade do visitante marcar")

    # 5. Força do Mandante (Peso 1+1+1)
    if stats_h["avg_ppg"] >= 1.6:
        score += 1
        reasons.append(f"PPG Mandante sólido ({stats_h['avg_ppg']:.2f}): Superioridade técnica")
    if stats_h["avg_xg"] > 1.5:
        score += 1
        reasons.append(f"xG Mandante alto ({stats_h['avg_xg']:.2f}): Forte produção ofensiva")
    if stats_h["cost"] > 1.2:
        score += 1
        reasons.append(f"Custo do Gol Mandante Alto ({stats_h['cost']:.2f}): Eficiência em evitar placares magros adversários")

    # 6. Qualidade da Amostra (Peso 1)
    if sample_quality == "Boa":
        score += 1
        reasons.append("Amostra histórica boa: leitura mais confiável")
    elif sample_quality == "Pequena":
        score -= 1
        reasons.append("Amostra histórica pequena: reduzir confiança do sinal")
    recommendation = "NÃO INDICADO"
    max_score = 16
    if score >= 11:
        recommendation = "FORTE INDICAÇÃO"
    elif score >= 7:
        recommendation = "INDICAÇÃO MODERADA"
    return {
        "home": stats_h,
        "away": stats_a,
        "last10_home": last10_home,
        "last10_away": last10_away,
        "role_profile_home": role_profile_home,
        "role_profile_away": role_profile_away,
        "lay_strength_home": lay01_index_home,
        "lay_strength_away": lay01_index_away,
        "lay_strength_home_label": classify_strength(lay01_index_home),
        "lay_strength_away_label": classify_strength(lay01_index_away),
        "lay_var_home": lay01_var_home,
        "lay_var_away": lay01_var_away,
        "lay_var_home_label": classify_variance(lay01_var_home),
        "lay_var_away_label": classify_variance(lay01_var_away),
        "lay_var_home_desc": describe_variance(classify_variance(lay01_var_home), "mandante"),
        "lay_var_away_desc": describe_variance(classify_variance(lay01_var_away), "visitante"),
        "poisson_0x1": poisson_0x1_ht,
        "poisson_0x1_label": "Heurística Poisson HT",
        "heuristic_success": poisson_0x1_ht,
        "heuristic_success_label": "Heurística Poisson HT",
        "h_games": home_h,
        "a_games": away_a,
        "sample_home": sample_home,
        "sample_away": sample_away,
        "sample_quality": sample_quality,
        "sample_warning": min_sample < 25,
        "red_from_00": red_00,
        "red_from_01": red_01,
        "pct_00_ht": pct_00_ht,
        "pct_01_ht": pct_01_ht,
        "pct_00_ft": pct_00_ft,
        "pct_01_ft": pct_01_ft,
        "pct_other_ht": pct_other_ht,
        "pct_other_ft": pct_other_ft,
        "recommendation": recommendation,
        "score": score,
        "max_score": max_score,
        "reasons": reasons,
    }
