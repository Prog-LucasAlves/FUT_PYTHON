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
                return df
    return pd.DataFrame()


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
    odd_lay_0x1 = current_match_data.get("Odd_CS_0x1_Lay", 0)
    odd_btts = current_match_data.get("Odd_BTTS_Yes_Back", 0)
    odd_over25 = current_match_data.get("Odd_Over25_FT_Back", 0)
    odd_h = current_match_data.get("Odd_H_Back", 0)

    if 0 < odd_lay_0x1 < 12:
        score += 2
        reasons.append("Odd Lay 0x1 atrativa (Responsabilidade controlada)")

    if 0 < odd_btts < 1.90:
        score += 2
        reasons.append(
            f"Odd BTTS baixa ({odd_btts:.2f}): Alta tendência de ambos marcarem",
        )

    if 0 < odd_over25 < 2.10:
        score += 1
        reasons.append(
            f"Odd Over 2.5 baixa ({odd_over25:.2f}): Expectativa de muitos gols",
        )

    if 0 < odd_h < 2.20:
        score += 1
        reasons.append(f"Mandante favorito ({odd_h:.2f}): Alta chance de gol do Home")

    # Critérios Estatísticos FootyStats
    if stats_h["avg_xg"] > 1.5:
        score += 1
        reasons.append(
            f"xG do Mandante alto ({stats_h['avg_xg']:.2f}): Forte produção ofensiva",
        )

    if stats_h["avg_ppg"] > 1.8:
        score += 1
        reasons.append(
            f"PPG do Mandante alto ({stats_h['avg_ppg']:.2f}): Time sólido em casa",
        )

    if stats_h["avg_da"] > 45:
        score += 1
        reasons.append(f"Ataques Perigosos do Mandante alto ({stats_h['avg_da']:.0f})")

    if poisson_0x1 < 7:
        score += 2
        reasons.append(f"Baixa probabilidade Poisson ({poisson_0x1:.1f}%)")

    combined_success = 100 - poisson_0x1
    if combined_success > 92:
        score += 2
        reasons.append(
            f"Taxa de sucesso histórica combinada excelente ({combined_success:.1f}%)",
        )

    recommendation = "NÃO INDICADO"
    if score >= 9:
        recommendation = "FORTE INDICAÇÃO"
    elif score >= 5:
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
                        <p style="margin:0; font-weight: bold;">Score de Confiança: {results["score"]}/11</p>
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
