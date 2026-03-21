import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Configuração da Página
st.set_page_config(page_title="Lay 0x1 - FutStats", page_icon="⚽", layout="wide")

# Estilo Customizado
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Funções de Carregamento de Dados
@st.cache_data(ttl=3600)
def load_historical_data():
    file_path = "data_total/dados_betfair.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=";")
        df["Date"] = pd.to_datetime(df["Date"])
        # Converter colunas de gols para int
        df["Goals_H_FT"] = pd.to_numeric(df["Goals_H_FT"], errors="coerce")
        df["Goals_A_FT"] = pd.to_numeric(df["Goals_A_FT"], errors="coerce")
        # Remover linhas sem placar
        df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])
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
                if "Odd_CS_0x1_Lay" not in df.columns:
                    df["Odd_CS_0x1_Lay"] = 0
                return df
    return pd.DataFrame()


# Lógica de Cálculo de Estatísticas
def get_team_stats(df, home_team, away_team, league=None):
    # Filtrar por liga se fornecido
    if league:
        df_league = df[df["League"] == league]
    else:
        df_league = df

    # Jogos em casa do time mandante
    home_games = df_league[df_league["Home"] == home_team].copy()
    # Jogos fora do time visitante
    away_games = df_league[df_league["Away"] == away_team].copy()
    # H2H
    h2h_games = df_league[((df_league["Home"] == home_team) & (df_league["Away"] == away_team)) | ((df_league["Home"] == away_team) & (df_league["Away"] == home_team))].copy()

    def calculate_metrics(games):
        total = len(games)
        if total == 0:
            return None

        count_0x1 = len(games[(games["Goals_H_FT"] == 0) & (games["Goals_A_FT"] == 1)])
        home_scored = len(games[games["Goals_H_FT"] > 0])
        away_over15 = len(games[games["Goals_A_FT"] > 1])
        away_zero = len(games[games["Goals_A_FT"] == 0])
        btts = len(games[(games["Goals_H_FT"] > 0) & (games["Goals_A_FT"] > 0)])
        over25 = len(games[(games["Goals_H_FT"] + games["Goals_A_FT"]) > 2])

        success_lay_0x1 = total - count_0x1

        return {
            "Total Jogos": total,
            "Frequência 0x1": count_0x1,
            "Freq 0x1 %": (count_0x1 / total) * 100,
            "Sucesso Lay 0x1 %": (success_lay_0x1 / total) * 100,
            "Home Scored %": (home_scored / total) * 100,
            "Away > 1.5 %": (away_over15 / total) * 100,
            "Away 0 Gols %": (away_zero / total) * 100,
            "BTTS %": (btts / total) * 100,
            "Over 2.5 %": (over25 / total) * 100,
            "Avg Goals H": games["Goals_H_FT"].mean(),
            "Avg Goals A": games["Goals_A_FT"].mean(),
            "Avg Total Goals": (games["Goals_H_FT"] + games["Goals_A_FT"]).mean(),
        }

    stats_home = calculate_metrics(home_games)
    stats_away = calculate_metrics(away_games)

    # League Average for context
    league_total = len(df_league)
    league_0x1 = len(df_league[(df_league["Goals_H_FT"] == 0) & (df_league["Goals_A_FT"] == 1)])
    league_avg_0x1 = (league_0x1 / league_total * 100) if league_total > 0 else 0

    return stats_home, stats_away, home_games, away_games, h2h_games, league_avg_0x1


# Interface Streamlit
st.title("⚽ Lay 0x1 Strategy Pro - FutStats")

df_hist = load_historical_data()
df_today = load_today_games()

if df_hist.empty:
    st.error("Erro ao carregar dados históricos.")
    st.stop()

# Sidebar
st.sidebar.image("https://www.betfair.com/ads-content/images/betfair-logo-new.svg", width=150)
st.sidebar.header("Configurações de Filtro")
leagues = sorted(df_hist["League"].unique().tolist())
selected_league = st.sidebar.selectbox("Selecione a Liga", ["Todas"] + leagues)

seasons = sorted(df_hist["Season"].unique().tolist(), reverse=True)
selected_season = st.sidebar.multiselect("Temporadas", seasons, default=seasons[:2])

if selected_league != "Todas":
    filtered_df = df_hist[(df_hist["League"] == selected_league) & (df_hist["Season"].isin(selected_season))]
    today_filtered = df_today[df_today["League"] == selected_league] if not df_today.empty else pd.DataFrame()
else:
    filtered_df = df_hist[df_hist["Season"].isin(selected_season)]
    today_filtered = df_today

# Jogos do Dia
st.subheader("🎯 Análise de Jogos do Dia")
if not today_filtered.empty:
    today_filtered["Match"] = today_filtered["Home"] + " vs " + today_filtered["Away"]
    selected_match = st.selectbox("Escolha o jogo para análise profunda", today_filtered["Match"].tolist())

    match_row = today_filtered[today_filtered["Match"] == selected_match].iloc[0]
    home_team = match_row["Home"]
    away_team = match_row["Away"]

    stats_h, stats_a, games_h, games_a, h2h, l_avg = get_team_stats(filtered_df, home_team, away_team, match_row["League"])

    # Layout de Topo
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Odd Lay 0x1", f"{match_row['Odd_CS_0x1_Lay']:.2f}" if match_row["Odd_CS_0x1_Lay"] > 0 else "N/A")
    with m2:
        prob_fair = (100 - l_avg) / 100
        st.metric("Sucesso Liga (Méd.)", f"{100 - l_avg:.1f}%")
    with m3:
        if stats_h:
            st.metric(f"Sucesso {home_team}", f"{stats_h['Sucesso Lay 0x1 %']:.1f}%")
    with m4:
        if stats_a:
            st.metric(f"Sucesso {away_team}", f"{stats_a['Sucesso Lay 0x1 %']:.1f}%")

    # Detalhes das Equipes
    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"### 🏠 {home_team} (Home Stats)")
        if stats_h:
            col_a, col_b = st.columns(2)
            col_a.metric("BTTS", f"{stats_h['BTTS %']:.1f}%")
            col_b.metric("Over 2.5", f"{stats_h['Over 2.5 %']:.1f}%")
            st.write(f"**Média Gols Total:** {stats_h['Avg Total Goals']:.2f}")
            st.write(f"**Fez gol em:** {stats_h['Home Scored %']:.1f}% dos jogos")

            # Gráfico de Gols
            fig_h = px.histogram(games_h, x="Goals_H_FT", title="Distribuição de Gols Mandante", labels={"Goals_H_FT": "Gols"}, color_discrete_sequence=["#2ecc71"])
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.warning("Sem dados históricos suficientes para o mandante.")

    with c2:
        st.markdown(f"### 🚀 {away_team} (Away Stats)")
        if stats_a:
            col_c, col_d = st.columns(2)
            col_c.metric("BTTS", f"{stats_a['BTTS %']:.1f}%")
            col_d.metric("Over 2.5", f"{stats_a['Over 2.5 %']:.1f}%")
            st.write(f"**Média Gols Total:** {stats_a['Avg Total Goals']:.2f}")
            st.write(f"**Sofreu gol em:** {100 - stats_a['Away 0 Gols %']:.1f}% dos jogos")

            # Gráfico de Gols
            fig_a = px.histogram(games_a, x="Goals_A_FT", title="Distribuição de Gols Visitante", labels={"Goals_A_FT": "Gols"}, color_discrete_sequence=["#e74c3c"])
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.warning("Sem dados históricos suficientes para o visitante.")

    # Análise de Valor (EV) e Recomendação
    st.markdown("---")
    st.subheader("⚖️ Análise de Valor e Recomendação")
    if stats_h and stats_a and match_row["Odd_CS_0x1_Lay"] > 0:
        est_prob = (stats_h["Sucesso Lay 0x1 %"] + stats_a["Sucesso Lay 0x1 %"]) / 2
        odd_justa = 100 / (100 - est_prob)
        current_lay = match_row["Odd_CS_0x1_Lay"]

        col_ev1, col_ev2, col_ev3 = st.columns(3)
        with col_ev1:
            st.write(f"**Probabilidade de Sucesso:** {est_prob:.1f}%")
            st.write(f"**Odd Justa (Back):** {odd_justa:.2f}")
        with col_ev2:
            st.write(f"**Odd Atual (Lay):** {current_lay:.2f}")
            # Simplified EV check for Lay: if current_lay < odd_justa (roughly)
            # Actually for Lay, we want current_lay to be as low as possible.
            if current_lay < 8:
                st.success("✅ Risco/Retorno Favorável")
            elif current_lay < 12:
                st.warning("⚠️ Risco Moderado")
            else:
                st.error("❌ Risco Elevado")

        with col_ev3:
            # Recomendação baseada em critérios
            score = 0
            if stats_h["Home Scored %"] > 70:
                score += 1
            if stats_a["Away 0 Gols %"] > 30:
                score += 1
            if stats_a["Away > 1.5 %"] > 30:
                score += 1
            if current_lay < 10:
                score += 1

            if score >= 3:
                st.info("💡 **RECOMENDAÇÃO:** Forte Candidato para Lay 0x1")
            elif score >= 2:
                st.info("💡 **RECOMENDAÇÃO:** Candidato Moderado")
            else:
                st.info("💡 **RECOMENDAÇÃO:** Evitar este jogo")

    # H2H e Últimos Jogos
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Confronto Direto (H2H)", "Últimos Mandante", "Últimos Visitante"])

    with tab1:
        if not h2h.empty:
            st.dataframe(h2h[["Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"]].sort_values("Date", ascending=False))
        else:
            st.write("Nenhum confronto direto encontrado.")

    with tab2:
        st.dataframe(games_h[["Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"]].tail(15))

    with tab3:
        st.dataframe(games_a[["Date", "Home", "Away", "Goals_H_FT", "Goals_A_FT"]].tail(15))

else:
    st.info("Selecione uma liga com jogos hoje para ver a análise detalhada.")

# Rodapé
st.markdown("---")
st.caption("FutStats v2.0 - Desenvolvido para Trading Esportivo | Lay 0x1 Method")
