import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import poisson


def normalize_team_name(name):
    if pd.isna(name):
        return ""
    name = str(name).lower()
    name = "".join(c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn")
    name = name.replace("se ", " ").replace("sc ", " ").replace("ec ", " ").replace("cr ", " ").replace("fc ", " ").replace("as ", " ").replace("us ", " ").replace("afc ", " ").replace("rcd ", " ").replace("rc ", " ").replace("cf ", " ").replace("cd ", " ")
    name = name.replace(" fc", " ").replace(" cf", " ").replace(" afc", " ").replace(" cfc", " ").replace(" sc", " ").replace(" ac", " ").replace(" ud", " ")
    name = name.replace(" rj", " ").replace(" sp", " ").replace(" mg", " ").replace(" pr", " ").replace(" go", " ").replace(" ba", " ").replace(" rs", " ")
    name = name.replace("atl. ", "atletico ").replace("atl ", "atletico ")
    name = name.replace("ath. ", "athletic ").replace("ath ", "athletic ")
    name = name.replace("int. ", "inter ").replace("int ", "inter ")
    name = name.replace("st. ", "saint ").replace("st ", "saint ")
    name = name.replace("man city", "mancity")
    name = name.replace("man utd", "manunited")
    name = name.replace("utd", "united")
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

    def calc_interval_pct(games, mins_col):
        result = {}
        total = len(games)
        if total == 0:
            return {f"{a}-{b}'": 0.0 for a, b in intervals}
        for start, end in intervals:
            label = f"{start}-{end}'"
            count = games[mins_col].apply(lambda mins: any((m := normalize_goal_minute_fn(x)) is not None and start < m <= end for x in mins)).sum()
            result[label] = (count / total) * 100
        return result

    def calc_combined(games):
        result = {}
        total = len(games)
        if total == 0:
            return {f"{a}-{b}'": 0.0 for a, b in intervals}
        for start, end in intervals:
            label = f"{start}-{end}'"
            count = games.apply(lambda row: any((m := normalize_goal_minute_fn(x)) is not None and start < m <= end for x in (row["Min_Goals_H"] + row["Min_Goals_A"])), axis=1).sum()
            result[label] = (count / total) * 100
        return result

    return {"home_attack": calc_interval_pct(h_games, "Min_Goals_H"), "away_attack": calc_interval_pct(a_games, "Min_Goals_A"), "home_combined": calc_combined(h_games), "away_combined": calc_combined(a_games), "home_sample": len(h_games), "away_sample": len(a_games)}


def build_poisson_timing_scenario(df_games, team_name, role, scenario_score, cutoff_minute=75, end_minute=90, normalize_team_name_fn=normalize_team_name, count_goals_until_fn=count_goals_until, count_goals_after_fn=count_goals_after):
    team_norm = normalize_team_name_fn(team_name)
    role_col = "Norm_Home" if role == "home" else "Norm_Away"
    team_games = df_games[df_games[role_col] == team_norm].copy()
    if team_games.empty:
        return None
    selected_games = []
    for _, row in team_games.iterrows():
        score_at_cutoff = (count_goals_until_fn(row["Min_Goals_H"], cutoff_minute), count_goals_until_fn(row["Min_Goals_A"], cutoff_minute))
        if score_at_cutoff == scenario_score:
            selected_games.append(row)
    if not selected_games:
        return None
    scenario_df = pd.DataFrame(selected_games)
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
                if row["Norm_Home"] == team_norm:
                    return f"{int(row[f'Goals_H_{suffix}'])}x{int(row[f'Goals_A_{suffix}'])}"
                return f"{int(row[f'Goals_A_{suffix}'])}x{int(row[f'Goals_H_{suffix}'])}"

            scores = df.apply(get_score_view, axis=1)
            return (scores.value_counts(normalize=True) * 100).head(5).to_dict()

        return {"HT": process_scores(games, "HT"), "FT": process_scores(games, "FT")}

    def get_timing_stats(games, team_norm):
        def get_team_mins(row):
            return row["Min_Goals_H"] if row["Norm_Home"] == team_norm else row["Min_Goals_A"]

        team_mins = games.apply(get_team_mins, axis=1)
        first_goal_team = team_mins.apply(lambda x: min(x) if len(x) > 0 else None).dropna()
        first_goal_match = games.apply(lambda row: min(row["Min_Goals_H"] + row["Min_Goals_A"]) if len(row["Min_Goals_H"] + row["Min_Goals_A"]) > 0 else None, axis=1).dropna()
        games_00_ht = games[(games["Goals_H_HT"] == 0) & (games["Goals_A_HT"] == 0)]

        def first_in_2h(mins_list):
            m2h = [m for m in (normalize_goal_minute_fn(m) for m in mins_list) if m is not None and m > 45]
            return min(m2h) if len(m2h) > 0 else None

        first_team_2h = games_00_ht.apply(get_team_mins, axis=1).apply(first_in_2h).dropna()
        first_match_2h = games_00_ht.apply(
            lambda row: min([m for m in (normalize_goal_minute_fn(m) for m in (row["Min_Goals_H"] + row["Min_Goals_A"])) if m is not None and m > 45]) if len([m for m in (normalize_goal_minute_fn(m) for m in (row["Min_Goals_H"] + row["Min_Goals_A"])) if m is not None and m > 45]) > 0 else None,
            axis=1,
        ).dropna()
        return {"avg_first_team": first_goal_team.mean(), "avg_first_match": first_goal_match.mean(), "avg_team_2h_00ht": first_team_2h.mean(), "avg_match_2h_00ht": first_match_2h.mean(), "sample_size": len(games), "sample_00ht": len(games_00_ht), "raw_first_team": first_goal_team.tolist()}

    if len(h_games) == 0 or len(a_games) == 0:
        return None, None, None
    stats_h = get_timing_stats(h_games, norm_h)
    stats_a = get_timing_stats(a_games, norm_a)
    stats_h["frequent_scores"] = get_frequent_scores(h_games, norm_h)
    stats_a["frequent_scores"] = get_frequent_scores(a_games, norm_a)
    stats_combined_scores = {
        "HT": (pd.concat([h_games, a_games])["Goals_H_HT"].astype(int).astype(str) + "x" + pd.concat([h_games, a_games])["Goals_A_HT"].astype(int).astype(str)).value_counts(normalize=True).head(5).mul(100).to_dict(),
        "FT": (pd.concat([h_games, a_games])["Goals_H_FT"].astype(int).astype(str) + "x" + pd.concat([h_games, a_games])["Goals_A_FT"].astype(int).astype(str)).value_counts(normalize=True).head(5).mul(100).to_dict(),
    }
    return stats_h, stats_a, stats_combined_scores


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
        mins = games.apply(lambda r: get_team_val(r, "Min_Goals_H", "Min_Goals_A"), axis=1)
        mean_goals = goals.mean()
        variance = goals.var()
        first_goal_mins = mins.apply(lambda x: x[0] if len(x) > 0 else None).dropna()
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
    lay01_index_home, lay01_var_home = calculate_lay_strength_index(stats_h)
    lay01_index_away, lay01_var_away = calculate_lay_strength_index(stats_a)
    prob_h0 = poisson.pmf(0, stats_h["mean"])
    prob_a1 = poisson.pmf(1, stats_a["mean"])
    poisson_0x1 = (prob_h0 * prob_a1) * 100

    def analyze_ht_scenarios(games_h, games_a):
        h_00_ht = games_h[(games_h["Goals_H_HT"] == 0) & (games_h["Goals_A_HT"] == 0)]
        a_00_ht = games_a[(games_a["Goals_H_HT"] == 0) & (games_a["Goals_A_HT"] == 0)]
        total_00_ht = len(h_00_ht) + len(a_00_ht)
        prob_red_from_00 = ((len(h_00_ht[(h_00_ht["Goals_H_FT"] == 0) & (h_00_ht["Goals_A_FT"] == 1)]) + len(a_00_ht[(a_00_ht["Goals_H_FT"] == 0) & (a_00_ht["Goals_A_FT"] == 1)])) / total_00_ht * 100) if total_00_ht > 0 else 0
        h_01_ht = games_h[(games_h["Goals_H_HT"] == 0) & (games_h["Goals_A_HT"] == 1)]
        a_01_ht = games_a[(games_a["Goals_H_HT"] == 0) & (games_a["Goals_A_HT"] == 1)]
        total_01_ht = len(h_01_ht) + len(a_01_ht)
        prob_red_from_01 = ((len(h_01_ht[(h_01_ht["Goals_H_FT"] == 0) & (h_01_ht["Goals_A_FT"] == 1)]) + len(a_01_ht[(a_01_ht["Goals_H_FT"] == 0) & (a_01_ht["Goals_A_FT"] == 1)])) / total_01_ht * 100) if total_01_ht > 0 else 0
        return prob_red_from_00, prob_red_from_01

    red_00, red_01 = analyze_ht_scenarios(home_h, away_a)
    sample_home = len(home_h)
    sample_away = len(away_a)
    min_sample = min(sample_home, sample_away)
    sample_quality = "Boa" if min_sample >= 50 else "Moderada" if min_sample >= 25 else "Pequena"
    score = 0
    reasons = []
    odd_h_back = current_match_data.get("Odd_H_Back", 0)
    odd_a_back = current_match_data.get("Odd_A_Back", 0)
    odd_lay_0x1 = current_match_data.get("Odd_CS_0x1_Lay", 0)
    odd_btts = current_match_data.get("Odd_BTTS_Yes_Back", 0)
    odd_over25 = current_match_data.get("Odd_Over25_FT_Back", 0)
    odds_rules = [
        {"name": "Faixa 1", "match": (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.99) and (odd_lay_0x1 >= 20.00)},
        {"name": "Faixa 2", "match": (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.99) and (13.00 <= odd_lay_0x1 <= 13.90)},
        {"name": "Faixa 3", "match": (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.90) and (12.00 <= odd_lay_0x1 <= 12.90)},
        {"name": "Faixa 4", "match": (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.99) and (18.00 <= odd_lay_0x1 <= 19.90)},
        {"name": "Faixa 5", "match": (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.99) and (15.00 <= odd_lay_0x1 <= 15.90)},
        {"name": "Faixa 6", "match": (2.50 <= odd_h_back <= 2.99) and (2.50 <= odd_a_back <= 2.99) and (11.00 <= odd_lay_0x1 <= 11.90)},
        {"name": "Faixa 7", "match": (1.80 <= odd_h_back <= 2.09) and (odd_a_back >= 5.00) and (15.00 <= odd_lay_0x1 <= 15.90)},
        {"name": "Faixa 8", "match": (1.80 <= odd_h_back <= 2.09) and (odd_a_back >= 5.00) and (14.00 <= odd_lay_0x1 <= 14.90)},
        {"name": "Faixa 9", "match": (2.10 <= odd_h_back <= 2.49) and (4.00 <= odd_a_back <= 4.99) and (11.00 <= odd_lay_0x1 <= 11.90)},
        {"name": "Faixa 10", "match": (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.99) and (16.00 <= odd_lay_0x1 <= 17.90)},
    ]
    matching_odds_rules = [rule["name"] for rule in odds_rules if rule["match"]]
    if matching_odds_rules:
        score += 5
        reasons.append(f"Padrão de Odds Detectado (Match Odds + Lay 0x1): {', '.join(matching_odds_rules)}")
    if 0 < odd_btts < 1.90:
        score += 2
        reasons.append(f"Odd BTTS baixa ({odd_btts:.2f}): Tendência de ambos marcarem")
    if 0 < odd_over25 < 2.10:
        score += 1
        reasons.append(f"Odd Over 2.5 baixa ({odd_over25:.2f}): Expectativa de gols")
    if stats_h["variance"] > 1.0:
        score += 1
        reasons.append(f"Variância Mandante Alta ({stats_h['variance']:.2f}): Time inconsistente (Bom para Lay)")
    if stats_h["cost"] > 1.2:
        score += 1
        reasons.append(f"Custo do Gol Mandante Alto ({stats_h['cost']:.2f}): Dificuldade em manter placares magros")
    if stats_h["avg_xg"] > 1.5:
        score += 1
        reasons.append(f"xG Mandante alto ({stats_h['avg_xg']:.2f}): Forte produção ofensiva")
    if poisson_0x1 < 7:
        score += 2
        reasons.append(f"Baixa probabilidade Poisson ({poisson_0x1:.1f}%)")
    heuristic_success = 100 - poisson_0x1
    if heuristic_success > 92:
        score += 2
        reasons.append(f"Sinal heurístico forte ({heuristic_success:.1f}%)")
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
        "poisson_0x1": poisson_0x1,
        "poisson_0x1_label": "Heurística Poisson",
        "heuristic_success": heuristic_success,
        "heuristic_success_label": "Complemento heurístico",
        "h_games": home_h,
        "a_games": away_a,
        "sample_home": sample_home,
        "sample_away": sample_away,
        "sample_quality": sample_quality,
        "sample_warning": min_sample < 25,
        "red_from_00": red_00,
        "red_from_01": red_01,
        "recommendation": recommendation,
        "score": score,
        "max_score": max_score,
        "reasons": reasons,
    }
