import ast
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

# Configuração da Página
st.set_page_config(page_title="Lay 0x1 PRO - FutStats", page_icon="📈", layout="wide")

# Estilo Customizado Profissional
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0d11; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(145deg, #1e2130, #161924);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
    }
    .strategy-card {
        background-color: #161924;
        border-left: 5px solid #00ff88;
        padding: 15px;
        margin: 10px 0;
    }
    .highlight-green { color: #00ff88; font-weight: bold; }
    .highlight-red { color: #ff4b4b; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Funções de Carregamento de Dados
@st.cache_data(ttl=3600)
def load_data():
    hist_path = "data_total/dados_betfair.csv"
    footy_path = "data_total/dados_footystats.csv"

    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path, sep=";")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])

        # Carregar dados adicionais do FootyStats para métricas avançadas (xG, PPG, Ataques)
        if os.path.exists(footy_path):
            try:
                df_footy = pd.read_csv(footy_path, sep=";")
                df_footy["Date"] = pd.to_datetime(df_footy["Date"])

                # Selecionar colunas relevantes do FootyStats
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
                # Filtrar colunas que realmente existem
                cols_to_merge = [c for c in cols_to_merge if c in df_footy.columns]

                # Merge com dados da Betfair (usando data e times como chave)
                df = pd.merge(
                    df,
                    df_footy[cols_to_merge],
                    on=["Date", "Home", "Away"],
                    how="left",
                )
            except Exception as e:
                st.sidebar.warning(f"Erro ao mesclar FootyStats: {e}")

        # Converter colunas de minutos para listas reais
        def parse_minutes(x):
            try:
                if pd.isna(x) or x == "" or x == "[]":
                    return []
                if isinstance(x, list):
                    return x
                return ast.literal_eval(x)
            except:
                return []

        df["Min_Goals_H"] = df["Min_Goals_H"].apply(parse_minutes)
        df["Min_Goals_A"] = df["Min_Goals_A"].apply(parse_minutes)

        return df
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_today_games():
    data_day_dir = "data_day/"
    if os.path.exists(data_day_dir):
        files = [f for f in os.listdir(data_day_dir) if f.endswith(".csv")]
        if files:
            df_list = []
            for file in files:
                try:
                    df_temp = pd.read_csv(os.path.join(data_day_dir, file), sep=";")
                    df_list.append(df_temp)
                except:
                    pass
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.drop_duplicates(subset=["Date", "Home", "Away"])
                return df
    return pd.DataFrame()


# Funções Utilitárias de Formatação
def format_minutes(decimal_min):
    if pd.isna(decimal_min) or decimal_min == 0:
        return "N/A"
    minutes = int(decimal_min)
    seconds = int((decimal_min - minutes) * 60)
    return f"{minutes}'{seconds:02d}\""


# Motor de Análise de Timing de Gols
def analyze_goal_timing(df_games, home_team, away_team):
    h_games = df_games[df_games["Home"] == home_team].copy()
    a_games = df_games[df_games["Away"] == away_team].copy()

    def get_frequent_scores(games, prefix_h="Goals_H", prefix_a="Goals_A"):
        def process_scores(df, suffix):
            scores = df[f"{prefix_h}_{suffix}"].astype(int).astype(str) + "x" + df[f"{prefix_a}_{suffix}"].astype(int).astype(str)
            counts = scores.value_counts(normalize=True) * 100
            return counts.head(5).to_dict()

        return {"HT": process_scores(games, "HT"), "FT": process_scores(games, "FT")}

    def get_timing_stats(games, is_home):
        col_mins = "Min_Goals_H" if is_home else "Min_Goals_A"
        # opp_col_mins = "Min_Goals_A" if is_home else "Min_Goals_H"

        # 1. Primeiro gol marcado pelo time (Individual)
        first_goal_team = games[col_mins].apply(lambda x: min(x) if len(x) > 0 else None).dropna()

        # 2. Primeiro gol da partida (Ambos os times)
        def match_first(row):
            h_mins = row["Min_Goals_H"]
            a_mins = row["Min_Goals_A"]
            all_mins = h_mins + a_mins
            return min(all_mins) if len(all_mins) > 0 else None

        first_goal_match = games.apply(match_first, axis=1).dropna()

        # 3. Cenário 0x0 HT
        games_00_ht = games[(games["Goals_H_HT"] == 0) & (games["Goals_A_HT"] == 0)]

        def first_in_2h(mins_list):
            m2h = [m for m in mins_list if m > 45]
            return min(m2h) if len(m2h) > 0 else None

        first_team_2h = games_00_ht[col_mins].apply(first_in_2h).dropna()

        def match_first_2h(row):
            all_2h = [m for m in (row["Min_Goals_H"] + row["Min_Goals_A"]) if m > 45]
            return min(all_2h) if len(all_2h) > 0 else None

        first_match_2h = games_00_ht.apply(match_first_2h, axis=1).dropna()

        return {
            "avg_first_team": first_goal_team.mean(),
            "avg_first_match": first_goal_match.mean(),
            "avg_team_2h_00ht": first_team_2h.mean(),
            "avg_match_2h_00ht": first_match_2h.mean(),
            "sample_size": len(games),
            "sample_00ht": len(games_00_ht),
            "raw_first_team": first_goal_team.tolist(),
        }

    stats_h = get_timing_stats(h_games, True)
    stats_a = get_timing_stats(a_games, False)

    # Adicionar placares frequentes
    stats_h["frequent_scores"] = get_frequent_scores(h_games)
    stats_a["frequent_scores"] = get_frequent_scores(a_games)

    combined_games = pd.concat([h_games, a_games]).drop_duplicates(subset=["Date", "Home", "Away"])
    stats_combined_scores = get_frequent_scores(combined_games)

    return stats_h, stats_a, stats_combined_scores


# Motor de Cálculo Estatístico PRO
def calculate_pro_metrics(df_games, home_team, away_team, current_match_data):
    home_h = df_games[df_games["Home"] == home_team].copy()
    away_a = df_games[df_games["Away"] == away_team].copy()

    if len(home_h) < 3 or len(away_a) < 3:
        return None

    def get_stats(games, col_goals, col_mins, prefix=""):
        goals = games[col_goals]
        mean_goals = goals.mean()
        variance = goals.var()

        # Minuto do primeiro gol
        first_goal_mins = games[col_mins].apply(lambda x: x[0] if len(x) > 0 else None).dropna()
        avg_first_goal = first_goal_mins.mean() if not first_goal_mins.empty else 0

        cost_of_goal = (variance / (mean_goals + 0.001)) if mean_goals > 0 else 0

        # Métricas FootyStats (xG e PPG)
        xg_col = f"xG_{prefix}" if f"xG_{prefix}" in games.columns else None
        ppg_col = f"PPG_{prefix}_Pre" if f"PPG_{prefix}_Pre" in games.columns else None
        da_col = f"DangerousAttacks_{prefix}" if f"DangerousAttacks_{prefix}" in games.columns else None

        avg_xg = games[xg_col].mean() if xg_col and not games[xg_col].dropna().empty else 0
        avg_ppg = games[ppg_col].mean() if ppg_col and not games[ppg_col].dropna().empty else 0
        avg_da = games[da_col].mean() if da_col and not games[da_col].dropna().empty else 0

        # Métrica de Chutes por Gol
        shots_col = f"Shots_{prefix}"
        total_shots = games[shots_col].sum() if shots_col in games.columns else 0
        total_goals = games[col_goals].sum()
        shots_per_goal = total_shots / total_goals if total_goals > 0 else 0

        return {
            "mean": mean_goals,
            "variance": variance,
            "cost": cost_of_goal,
            "zeros": (len(games[goals == 0]) / len(games)) * 100,
            "over15": (len(games[goals > 1.5]) / len(games)) * 100,
            "avg_first_goal": avg_first_goal,
            "total_games": len(games),
            "avg_xg": avg_xg,
            "avg_ppg": avg_ppg,
            "avg_da": avg_da,
            "shots_per_goal": shots_per_goal,
        }

    stats_h = get_stats(home_h, "Goals_H_FT", "Min_Goals_H", "H")
    stats_a = get_stats(away_a, "Goals_A_FT", "Min_Goals_A", "A")

    prob_h0 = poisson.pmf(0, stats_h["mean"])
    prob_a1 = poisson.pmf(1, stats_a["mean"])
    poisson_0x1 = (prob_h0 * prob_a1) * 100

    # Análise HT/FT
    def analyze_ht_scenarios(games_h, games_a):
        h_00_ht = games_h[(games_h["Goals_H_HT"] == 0) & (games_h["Goals_A_HT"] == 0)]
        a_00_ht = games_a[(games_a["Goals_H_HT"] == 0) & (games_a["Goals_A_HT"] == 0)]
        total_00_ht = len(h_00_ht) + len(a_00_ht)
        prob_red_from_00 = (
            (
                (
                    len(
                        h_00_ht[(h_00_ht["Goals_H_FT"] == 0) & (h_00_ht["Goals_A_FT"] == 1)],
                    )
                    + len(
                        a_00_ht[(a_00_ht["Goals_H_FT"] == 0) & (a_00_ht["Goals_A_FT"] == 1)],
                    )
                )
                / total_00_ht
                * 100
            )
            if total_00_ht > 0
            else 0
        )

        h_01_ht = games_h[(games_h["Goals_H_HT"] == 0) & (games_h["Goals_A_HT"] == 1)]
        a_01_ht = games_a[(games_a["Goals_H_HT"] == 0) & (games_a["Goals_A_HT"] == 1)]
        total_01_ht = len(h_01_ht) + len(a_01_ht)
        prob_red_from_01 = (
            (
                (
                    len(
                        h_01_ht[(h_01_ht["Goals_H_FT"] == 0) & (h_01_ht["Goals_A_FT"] == 1)],
                    )
                    + len(
                        a_01_ht[(a_01_ht["Goals_H_FT"] == 0) & (a_01_ht["Goals_A_FT"] == 1)],
                    )
                )
                / total_01_ht
                * 100
            )
            if total_01_ht > 0
            else 0
        )

        return prob_red_from_00, prob_red_from_01

    red_00, red_01 = analyze_ht_scenarios(home_h, away_a)

    # Lógica de Recomendação Baseada em Odds e Estatísticas (FootyStats incluído)
    score = 0
    reasons = []

    # Critérios de Odds
    odd_h_back = current_match_data.get("Odd_H_Back", 0)
    odd_a_back = current_match_data.get("Odd_A_Back", 0)
    odd_lay_0x1 = current_match_data.get("Odd_CS_0x1_Lay", 0)
    odd_btts = current_match_data.get("Odd_BTTS_Yes_Back", 0)
    odd_over25 = current_match_data.get("Odd_Over25_FT_Back", 0)

    # 1º Condição 1.80-2.09 | 4.00-4.99 | 20.0+
    cond1 = (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.99) and (odd_lay_0x1 >= 20.00)
    # 2º Condição 1.80-2.09 | 4.00-4.99 | 13.0-13.9
    cond2 = (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.90) and (13.00 <= odd_lay_0x1 <= 19.90)
    # 3º Condição 2.10-2.49 | 3.50-3.99 | 12.0-12.9
    cond3 = (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.90) and (12.00 <= odd_lay_0x1 <= 12.90)
    # 4º Condição 1.80-2.09 | 4.00-4.99 | 18.0-19.9
    cond4 = (1.80 <= odd_h_back <= 2.09) and (4.00 <= odd_a_back <= 4.99) and (18.00 <= odd_lay_0x1 <= 19.90)
    # 5° Condição 2.10-2.49 | 3.50-3.99 | 15.0-15.9
    cond5 = (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.99) and (15.00 <= odd_lay_0x1 <= 15.90)
    # 6º Condição 2.50-2.99 | 2.50-2.99 | 11.0-11.9
    cond6 = (2.50 <= odd_h_back <= 2.99) and (2.50 <= odd_a_back <= 2.99) and (11.00 <= odd_lay_0x1 <= 11.90)
    # 7º Condição 1.80-2.09 | 5.00+ | 15.0-15.9
    cond7 = (1.80 <= odd_h_back <= 2.09) and (odd_a_back >= 5.00) and (15.00 <= odd_lay_0x1 <= 15.90)
    # 8º Condição 1.80-2.09 | 5.00+ | 14.0-14.9
    cond8 = (1.80 <= odd_h_back <= 2.09) and (odd_a_back >= 5.00) and (14.00 <= odd_lay_0x1 <= 14.90)
    # 9° Condição 2.10-2.49 | 4.00-4.99 | 11.0-11.9
    cond9 = (2.10 <= odd_h_back <= 2.49) and (4.00 <= odd_a_back <= 4.99) and (11.00 <= odd_lay_0x1 <= 11.90)
    # 10º Condição 2.10-2.49 | 3.50-3.99 | 16.0-17.9
    cond10 = (2.10 <= odd_h_back <= 2.49) and (3.50 <= odd_a_back <= 3.99) and (16.00 <= odd_lay_0x1 <= 17.90)

    if any([cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond10]):
        score += 5
        reasons.append("Padrão de Odds Detectado (Match Odds + Lay 0x1)")

    # Critérios Adicionais de Odds
    if 0 < odd_btts < 1.90:
        score += 2
        reasons.append(f"Odd BTTS baixa ({odd_btts:.2f}): Tendência de ambos marcarem")

    if 0 < odd_over25 < 2.10:
        score += 1
        reasons.append(f"Odd Over 2.5 baixa ({odd_over25:.2f}): Expectativa de gols")

    # Filtros de Variância e Custo do Gol
    if stats_h["variance"] > 1.0:
        score += 1
        reasons.append(f"Variância Mandante Alta ({stats_h['variance']:.2f}): Time inconsistente (Bom para Lay)")

    if stats_h["cost"] > 1.2:
        score += 1
        reasons.append(f"Custo do Gol Mandante Alto ({stats_h['cost']:.2f}): Dificuldade em manter placares magros")

    # Critérios Estatísticos FootyStats
    if stats_h["avg_xg"] > 1.5:
        score += 1
        reasons.append(f"xG Mandante alto ({stats_h['avg_xg']:.2f}): Forte produção ofensiva")

    if poisson_0x1 < 7:
        score += 2
        reasons.append(f"Baixa probabilidade Poisson ({poisson_0x1:.1f}%)")

    combined_success = 100 - poisson_0x1
    if combined_success > 92:
        score += 2
        reasons.append(f"Sucesso histórico excelente ({combined_success:.1f}%)")

    recommendation = "NÃO INDICADO"
    if score >= 10:
        recommendation = "FORTE INDICAÇÃO"
    elif score >= 6:
        recommendation = "INDICAÇÃO MODERADA"

    return {
        "home": stats_h,
        "away": stats_a,
        "poisson_0x1": poisson_0x1,
        "combined_success": combined_success,
        "h_games": home_h,
        "a_games": away_a,
        "red_from_00": red_00,
        "red_from_01": red_01,
        "recommendation": recommendation,
        "score": score,
        "reasons": reasons,
    }


# Interface
st.title("🛡️ Lay 0x1 Ultimate - Professional Trading Tool")

df_hist = load_data()
df_today = load_today_games()

if df_hist.empty:
    st.error("Não foi possível carregar os dados históricos.")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Busca & Filtros")
date_selected = st.sidebar.date_input(
    "Filtrar Jogos do Dia",
    value=pd.to_datetime("2024-03-16"),
)  # Data de exemplo com dados

leagues = sorted(df_hist["League"].unique().tolist())
selected_leagues = st.sidebar.multiselect("Ligas", leagues, default=leagues[:5])

if not df_today.empty:
    df_day_filtered = df_today[(df_today["Date"].dt.date == date_selected)]
    if selected_leagues:
        df_day_filtered = df_day_filtered[df_day_filtered["League"].isin(selected_leagues)]

    if not df_day_filtered.empty:
        st.subheader(f"📅 Jogos Encontrados em {date_selected}")

        df_day_filtered["Match"] = df_day_filtered["Home"] + " vs " + df_day_filtered["Away"]
        selected_match = st.selectbox(
            "Selecione o Jogo para Análise Profunda",
            df_day_filtered["Match"].tolist(),
        )

        m_data = df_day_filtered[df_day_filtered["Match"] == selected_match].iloc[0]
        results = calculate_pro_metrics(df_hist, m_data["Home"], m_data["Away"], m_data)

        if results:
            # DASHBOARD DE ODDS
            st.markdown("---")
            st.subheader("💰 Monitoramento de Odds de Mercado")
            o1, o2, o3, o4, o5 = st.columns(5)
            with o1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Match Odds (H/D/A)**")
                st.write(
                    f"H: {m_data.get('Odd_H_Back', 0):.2f} / {m_data.get('Odd_H_Lay', 0):.2f}",
                )
                st.write(
                    f"D: {m_data.get('Odd_D_Back', 0):.2f} / {m_data.get('Odd_D_Lay', 0):.2f}",
                )
                st.write(
                    f"A: {m_data.get('Odd_A_Back', 0):.2f} / {m_data.get('Odd_A_Lay', 0):.2f}",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with o2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Over 2.5 Goals**")
                st.write(f"Back: {m_data.get('Odd_Over25_FT_Back', 0):.2f}")
                st.write(f"Lay: {m_data.get('Odd_Over25_FT_Lay', 0):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with o3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**BTTS (Yes)**")
                st.write(f"Back: {m_data.get('Odd_BTTS_Yes_Back', 0):.2f}")
                st.write(f"Lay: {m_data.get('Odd_BTTS_Yes_Lay', 0):.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with o4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Correct Score 0x1**")
                st.write(f"Back: {m_data.get('Odd_CS_0x1_Back', 0):.2f}")
                st.write(
                    f"Lay: <span class='highlight-red'>{m_data.get('Odd_CS_0x1_Lay', 0):.2f}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with o5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Análise de Valor (EV)**")
                odd_lay = m_data.get("Odd_CS_0x1_Lay", 0)
                if odd_lay > 0:
                    ev = (results["combined_success"] / 100) * 1 - (1 - results["combined_success"] / 100) * (odd_lay - 1)
                    st.write(
                        f"EV: <span class='{'highlight-green' if ev > 0 else 'highlight-red'}'>{ev:+.2f}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("EV: N/A")
                st.markdown("</div>", unsafe_allow_html=True)

            # ESTRATÉGIA VENCEDORA
            st.markdown("---")
            st.subheader("🎯 Recomendação de Estratégia Lay 0x1")

            res_col, reasons_col = st.columns([1, 2])

            with res_col:
                color = "#00ff88" if "FORTE" in results["recommendation"] else "#ffcc00" if "MODERADA" in results["recommendation"] else "#ff4b4b"
                st.markdown(
                    f"""
                    <div style="background-color: {color}; color: black; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="margin:0;">{results["recommendation"]}</h2>
                        <p style="margin:0; font-weight: bold;">Score de Confiança: {results["score"]}/15</p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            with reasons_col:
                st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
                st.write("**Por que esta indicação?**")
                for reason in results["reasons"]:
                    st.write(f"✅ {reason}")
                st.markdown("</div>", unsafe_allow_html=True)

            # CRITÉRIOS DE SAÍDA
            st.markdown("---")
            st.subheader("🚪 Critérios de Saída (Gestão de Risco)")
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.success("🏁 **SAÍDA COM GREEN (LUCRO TOTAL)**")
                st.write("- Gol do Mandante (1-0, 2-0, etc.)")
                st.write("- Segundo gol do Visitante (0-2, 1-2, etc.)")
                st.write("- O jogo termina empatado (0-0, 1-1, etc.)")
            with ex2:
                st.warning("🏁 **SAÍDA ESTRATÉGICA (HEDGE/PROTEÇÃO)**")
                st.write(
                    "- **Minuto 75'**: Se o placar for 0-0 ou 0-1, sair para proteger capital.",
                )
                st.write(
                    "- **Pressão Extrema**: Se o visitante estiver com > 70% de posse e muitos ataques perigosos no 2º tempo.",
                )
            with ex3:
                st.error("🏁 **SAÍDA COM RED (PREJUÍZO)**")
                st.write("- O placar termina exatamente em 0-1.")
                st.write("- Se você decidir aceitar o red total no apito final.")

            # DASHBOARD PRINCIPAL (POISSON E VOLATILIDADE)
            st.markdown("---")
            st.subheader("📈 Análise Quantitativa e In-Play")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Prob. Poisson 0x1", f"{results['poisson_0x1']:.2f}%")
                st.write(
                    f"Confiança Probabilística: {results['combined_success']:.1f}%",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Cenário: 0x0 no HT**")
                st.write(f"Risco de Red: {results['red_from_00']:.1f}%")
                st.info("Sobrevivência do 0-0 no 2º tempo.")
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Médias Ofensivas (xG)**")
                st.write(f"Home xG: {results['home']['avg_xg']:.2f}")
                st.write(f"Away xG: {results['away']['avg_xg']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write("**Domínio de Campo (PPG)**")
                st.write(f"Home PPG: {results['home']['avg_ppg']:.2f}")
                st.write(f"Away PPG: {results['away']['avg_ppg']:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

            # Métrica de Chutes por Gol
            st.markdown("---")
            st.subheader("🎯 Eficiência de Finalização (Chutes por Gol)")
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(f"Chutes/Gol - {m_data['Home']}", f"{results['home']['shots_per_goal']:.1f}")
                st.write("Média de chutes necessários para marcar 1 gol")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_sh2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(f"Chutes/Gol - {m_data['Away']}", f"{results['away']['shots_per_goal']:.1f}")
                st.write("Média de chutes necessários para marcar 1 gol")
                st.markdown("</div>", unsafe_allow_html=True)

            # --- NOVA SEÇÃO: ANÁLISE QUANTITATIVA DE TIMING ---
            st.markdown("---")
            st.subheader("⏱️ Análise de Timing e Explosão (First Goal)")

            t_stats_h, t_stats_a, t_combined_scores = analyze_goal_timing(df_hist, m_data["Home"], m_data["Away"])

            if t_stats_h and t_stats_a:
                col_t1, col_t2 = st.columns(2)

                with col_t1:
                    st.markdown(f"#### 🏠 {m_data['Home']} (Timing)")
                    data_t_h = {
                        "Métrica": ["Primeiro Gol (Individual)", "Primeiro Gol (Partida)", "Primeiro Gol 2T (se 0x0 HT)", "Primeiro Gol Partida 2T (se 0x0 HT)"],
                        "Média": [
                            format_minutes(t_stats_h["avg_first_team"]),
                            format_minutes(t_stats_h["avg_first_match"]),
                            format_minutes(t_stats_h["avg_team_2h_00ht"]),
                            format_minutes(t_stats_h["avg_match_2h_00ht"]),
                        ],
                        "Amostra": [f"{t_stats_h['sample_size']} jogos", f"{t_stats_h['sample_size']} jogos", f"{t_stats_h['sample_00ht']} jogos", f"{t_stats_h['sample_00ht']} jogos"],
                    }
                    st.table(pd.DataFrame(data_t_h))

                with col_t2:
                    st.markdown(f"#### 🚀 {m_data['Away']} (Timing)")
                    data_t_a = {
                        "Métrica": ["Primeiro Gol (Individual)", "Primeiro Gol (Partida)", "Primeiro Gol 2T (se 0x0 HT)", "Primeiro Gol Partida 2T (se 0x0 HT)"],
                        "Média": [
                            format_minutes(t_stats_a["avg_first_team"]),
                            format_minutes(t_stats_a["avg_first_match"]),
                            format_minutes(t_stats_a["avg_team_2h_00ht"]),
                            format_minutes(t_stats_a["avg_match_2h_00ht"]),
                        ],
                        "Amostra": [f"{t_stats_a['sample_size']} jogos", f"{t_stats_a['sample_size']} jogos", f"{t_stats_a['sample_00ht']} jogos", f"{t_stats_a['sample_00ht']} jogos"],
                    }
                    st.table(pd.DataFrame(data_t_a))

                # Gráfico Comparativo de Distribuição de Tempo
                st.markdown("#### 📊 Distribuição do Minuto do Primeiro Gol")
                fig_time = go.Figure()
                fig_time.add_trace(go.Box(y=t_stats_h["raw_first_team"], name=m_data["Home"], marker_color="#00ff88", boxpoints="all"))
                fig_time.add_trace(go.Box(y=t_stats_a["raw_first_team"], name=m_data["Away"], marker_color="#ff4b4b", boxpoints="all"))
                fig_time.update_layout(title="Boxplot: Quando o 1º gol costuma sair?", yaxis_title="Minuto", template="plotly_dark", height=400, showlegend=False)
                # Adicionar linha de 45' para referência HT
                fig_time.add_hline(y=45, line_dash="dash", line_color="white", annotation_text="Fim 1T")
                st.plotly_chart(fig_time, use_container_width=True)

                # Visualização de Placares Frequentes
                st.markdown("---")
                st.subheader("🔢 Placares Mais Frequentes (%)")

                def display_scores(scores_dict, title):
                    st.write(f"**{title}**")
                    df_scores = pd.DataFrame(list(scores_dict.items()), columns=["Placar", "Freq %"])
                    df_scores = df_scores.sort_values("Freq %", ascending=False)
                    fig = px.bar(df_scores, x="Placar", y="Freq %", text_auto=".1f", color="Freq %", color_continuous_scale="Viridis", template="plotly_dark")
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key=f"scores_{title}")

                tab_scores1, tab_scores2, tab_scores3 = st.tabs([f"🏠 {m_data['Home']}", f"🚀 {m_data['Away']}", "🤝 Confronto (Ambos)"])

                with tab_scores1:
                    c1, c2 = st.columns(2)
                    with c1:
                        display_scores(t_stats_h["frequent_scores"]["HT"], "HT - Frequência")
                    with c2:
                        display_scores(t_stats_h["frequent_scores"]["FT"], "FT - Frequência")

                with tab_scores2:
                    c1, c2 = st.columns(2)
                    with c1:
                        display_scores(t_stats_a["frequent_scores"]["HT"], "HT - Frequência ")
                    with c2:
                        display_scores(t_stats_a["frequent_scores"]["FT"], "FT - Frequência ")

                with tab_scores3:
                    c1, c2 = st.columns(2)
                    with c1:
                        display_scores(t_combined_scores["HT"], "HT - Frequência Combinada")
                    with c2:
                        display_scores(t_combined_scores["FT"], "FT - Frequência Combinada")

                st.info("""
                💡 **Interpretação:**
                - **Primeiro Gol (Individual):** Média de quando o time faz seu primeiro gol.
                - **Primeiro Gol (Partida):** Média de quando sai o primeiro gol do jogo (qualquer time).
                - **Cenário 0x0 HT:** Foco total no comportamento das equipes no segundo tempo quando o placar está travado.
                """)
            else:
                st.warning("Dados de minutagem insuficientes para este confronto.")

            # GRÁFICOS DE VOLATILIDADE
            st.markdown("---")
            st.subheader("📊 Métricas de Eficiência e Variância")
            v1, v2 = st.columns(2)
            with v1:
                st.write(f"### 🏠 {m_data['Home']}")
                st.write(f"**Variância de Gols:** {results['home']['variance']:.3f}")
                st.write(
                    f"**Custo do Gol (Eficiência):** {results['home']['cost']:.3f}",
                )
                fig_h = go.Figure()
                fig_h.add_trace(
                    go.Scatter(
                        y=results["h_games"]["Goals_H_FT"],
                        mode="lines+markers",
                        name="Gols",
                        line=dict(color="#00ff88"),
                    ),
                )
                fig_h.update_layout(
                    title="Histórico de Gols (Casa)",
                    template="plotly_dark",
                    height=250,
                )
                st.plotly_chart(fig_h, use_container_width=True)
            with v2:
                st.write(f"### 🚀 {m_data['Away']}")
                st.write(f"**Variância de Gols:** {results['away']['variance']:.3f}")
                st.write(
                    f"**Custo do Gol (Eficiência):** {results['away']['cost']:.3f}",
                )
                fig_a = go.Figure()
                fig_a.add_trace(
                    go.Scatter(
                        y=results["a_games"]["Goals_A_FT"],
                        mode="lines+markers",
                        name="Gols",
                        line=dict(color="#ff4b4b"),
                    ),
                )
                fig_a.update_layout(
                    title="Histórico de Gols (Fora)",
                    template="plotly_dark",
                    height=250,
                )
                st.plotly_chart(fig_a, use_container_width=True)

            # MATRIZ POISSON
            st.markdown("---")
            st.subheader("🎲 Matriz de Probabilidades (Poisson)")
            max_goals = 5
            matrix = np.zeros((max_goals, max_goals))
            for i in range(max_goals):
                for j in range(max_goals):
                    matrix[i, j] = (poisson.pmf(i, results["home"]["mean"]) * poisson.pmf(j, results["away"]["mean"])) * 100
            fig_matrix = px.imshow(
                matrix,
                labels=dict(x="Gols Visitante", y="Gols Mandante", color="%"),
                x=[str(i) for i in range(max_goals)],
                y=[str(i) for i in range(max_goals)],
                color_continuous_scale="Viridis",
                text_auto=".1f",
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        else:
            st.warning("Dados históricos insuficientes para este confronto.")
    else:
        st.info(f"Nenhum jogo encontrado para {date_selected}.")
else:
    st.error("Arquivos de jogos do dia não encontrados.")

# Rodapé
st.markdown("---")
st.caption("FutStats PRO v4.0 | Estratégias Avançadas In-Play")
