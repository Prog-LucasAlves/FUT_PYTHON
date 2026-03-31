import ast
import os
from pathlib import Path

import lay0x1_core
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson

# Configuração da Página
st.set_page_config(page_title="Lay 0x1 PRO - FutStats", page_icon="📈", layout="wide")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CACHE_VERSION = "2026-03-29-team-aliases-v11"
BETS_TRACKER_FILE = CURRENT_DIR / "bets_lay_tracker.csv"
DEFAULT_DATE = pd.Timestamp.today().normalize().date()
BET_STATUS_OPTIONS = ["", "Green", "75min"]

# Mapeamento explícito FootyStats → Betfair (nomes canônicos)
FOOTYSTATS_TEAM_MAP = {
    "Roma": "AS Roma",
    "AFC Bournemouth": "Bournemouth",
    "Almería": "Almeria",
    "América Mineiro": "America Mineiro",
    "Angers SCO": "Angers",
    "Athletic Club Bilbao": "Ath Bilbao",
    "Atlético GO": "Atletico GO",
    "Atlético Madrid": "Atl. Madrid",
    "Atlético PR": "Athletico-PR",
    "Bayern München": "Bayern Munich",
    "Boavista FC": "Boavista",
    "Borussia Dortmund": "Dortmund",
    "Borussia M'gladbach": "B. Monchengladbach",
    "Botafogo": "Botafogo RJ",
    "Brighton & Hove Albion": "Brighton",
    "CA Osasuna": "Osasuna",
    "CD Nacional": "Nacional",
    "CD Tondela": "Tondela",
    "Ceará": "Ceara",
    "Celta de Vigo": "Celta Vigo",
    "Chapecoense": "Chapecoense-SC",
    "Criciúma": "Criciuma",
    "Cuiabá": "Cuiaba",
    "Cádiz": "Cadiz CF",
    "Darmstadt 98": "Darmstadt",
    "Deportivo Alavés": "Alaves",
    "Elche CF": "Elche",
    "Estrela Amadora": "Estrela",
    "FC Arouca": "Arouca",
    "FC Barcelona": "Barcelona",
    "FC Vizela": "Vizela",
    "Famalicão": "Famalicao",
    "Flamengo": "Flamengo RJ",
    "GD Chaves": "Chaves",
    "GD Estoril Praia": "Estoril",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Grêmio": "Gremio",
    "Hellas Verona": "Verona",
    "Inter Milan": "Inter",
    "Ipswich Town": "Ipswich",
    "Köln": "FC Koln",
    "Leeds United": "Leeds",
    "Leganés": "Leganes",
    "Leicester City": "Leicester",
    "Levante UD": "Levante",
    "Luton Town": "Luton",
    "Mainz 05": "Mainz",
    "Manchester United": "Manchester Utd",
    "Moreirense FC": "Moreirense",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nottingham",
    "Olympique Lyonnais": "Lyon",
    "Olympique Marseille": "Marseille",
    "Paris": "Paris FC",
    "Porto": "FC Porto",
    "RCD Espanyol": "Espanyol",
    "RCD Mallorca": "Mallorca",
    "Real Betis": "Betis",
    "Real Oviedo": "R. Oviedo",
    "Real Valladolid": "Valladolid",
    "Rio Ave FC": "Rio Ave",
    "Saint-Étienne": "St Etienne",
    "Sevilla FC": "Sevilla",
    "Sheffield United": "Sheffield Utd",
    "Sporting Braga": "Braga",
    "Sporting CP": "Sporting CP",
    "São Paulo": "Sao Paulo",
    "Tottenham Hotspur": "Tottenham",
    "UD Las Palmas": "Las Palmas",
    "Valencia CF": "Valencia",
    "Vasco da Gama": "Vasco",
    "Vitória": "Vitoria",
    "Vitória Guimarães": "Vitoria Guimaraes",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


normalize_team_name = lay0x1_core.normalize_team_name


def load_lay_bets():
    expected_cols = [
        "data",
        "mandante",
        "visitante",
        "hora",
        "mercado",
        "odd_entrada",
        "valor_aposta",
        "responsabilidade",
        "entrada",
        "saida",
        "odd_saida_75min",
        "resultado",
        "percentual_resultado",
    ]
    if os.path.exists(BETS_TRACKER_FILE):
        df = pd.read_csv(BETS_TRACKER_FILE)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[expected_cols].copy()
        df["saida"] = df["saida"].fillna("")
        df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date.astype("string")
        df["hora"] = df["hora"].fillna("").astype(str)
        df["mercado"] = df["mercado"].fillna("").astype(str)
        df["percentual_resultado"] = np.where(
            df["responsabilidade"].fillna(0) > 0,
            (df["resultado"] / df["responsabilidade"]) * 100,
            np.nan,
        )
        df = df.drop_duplicates(subset=["data", "mandante", "visitante", "hora", "mercado"], keep="last").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=expected_cols)


def save_lay_bets(df_bets):
    df_bets = df_bets.copy()
    if not df_bets.empty:
        df_bets = df_bets.drop_duplicates(subset=["data", "mandante", "visitante", "hora", "mercado"], keep="last")
        df_bets = df_bets.sort_values(by=["data", "hora", "mandante", "visitante"], ascending=[False, False, True, True])
    df_bets.to_csv(BETS_TRACKER_FILE, index=False)


def calculate_lay_liability(odd_entrada, valor_aposta):
    return valor_aposta * max(odd_entrada - 1, 0)


def calculate_lay_result(odd_entrada, valor_aposta, saida, odd_saida_75min):
    if saida == "Green":
        return valor_aposta
    if saida == "75min" and odd_saida_75min and odd_saida_75min > 0:
        # Hedge teórico para fechar a posição no mesmo resultado líquido em ambos os lados.
        hedge_back_stake = (odd_entrada * valor_aposta) / odd_saida_75min
        return valor_aposta - hedge_back_stake
    return np.nan


def build_bet_label(row_idx, row):
    return f"#{row_idx} | {row['data']} {row['hora']} | {row['mandante']} vs {row['visitante']} | {row['mercado']}"


def style_bets_dataframe(df_bets):
    display_columns = [
        "Data",
        "Mandante",
        "Visitante",
        "Hora",
        "Mercado",
        "Odd Entrada",
        "Stake (R$)",
        "Responsabilidade (R$)",
        "Tipo de Entrada",
        "Saida",
        "Odd Saida 75min",
        "Resultado (R$)",
        "Performance %",
    ]
    df_display = df_bets.copy().rename(
        columns={
            "data": "Data",
            "mandante": "Mandante",
            "visitante": "Visitante",
            "hora": "Hora",
            "mercado": "Mercado",
            "odd_entrada": "Odd Entrada",
            "valor_aposta": "Stake (R$)",
            "responsabilidade": "Responsabilidade (R$)",
            "entrada": "Tipo de Entrada",
            "saida": "Saida",
            "odd_saida_75min": "Odd Saida 75min",
            "resultado": "Resultado (R$)",
            "percentual_resultado": "Performance %",
        },
    )
    df_display = df_display[display_columns]

    def color_performance(value):
        if pd.isna(value):
            return ""
        if value > 0:
            return "color: #00ff88; font-weight: bold;"
        if value < 0:
            return "color: #ff4b4b; font-weight: bold;"
        return "color: #ffcc00; font-weight: bold;"

    formatters = {
        "Odd Entrada": "{:.2f}",
        "Stake (R$)": "R$ {:.2f}",
        "Responsabilidade (R$)": "R$ {:.2f}",
        "Odd Saida 75min": lambda x: "" if pd.isna(x) else f"{x:.2f}",
        "Resultado (R$)": lambda x: "" if pd.isna(x) else f"R$ {x:.2f}",
        "Performance %": lambda x: "" if pd.isna(x) else f"{x:+.1f}%",
    }
    return df_display.style.format(formatters).map(color_performance, subset=["Resultado (R$)", "Performance %"])


build_risk_plan = lay0x1_core.build_risk_plan


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
@st.cache_data(ttl=3599)
def load_data(_cache_version=CACHE_VERSION):
    merged_path = PROJECT_ROOT / "data_total" / "dados_historico.csv"
    hist_path = PROJECT_ROOT / "data_total" / "dados_betfair.csv"
    footy_path = PROJECT_ROOT / "data_total" / "dados_footystats.csv"

    # Preferir arquivo pré-processado (dados_historico.csv) se disponível
    source_path = merged_path if merged_path.exists() else hist_path

    if source_path.exists():
        df = pd.read_csv(source_path, sep=";")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Goals_H_FT", "Goals_A_FT"])

        # Criar nomes normalizados imediatamente para garantir o match em qualquer cenário
        df["Norm_Home"] = df["Home"].apply(normalize_team_name)
        df["Norm_Away"] = df["Away"].apply(normalize_team_name)

        # Merge com FootyStats quando o arquivo existir e os campos ainda não estiverem presentes.
        if footy_path.exists():
            try:
                df_footy = pd.read_csv(footy_path, sep=";")
                df_footy["Date"] = pd.to_datetime(df_footy["Date"])

                # Aplicar mapeamento explícito de nomes antes de normalizar
                df_footy["Home"] = df_footy["Home"].replace(FOOTYSTATS_TEAM_MAP)
                df_footy["Away"] = df_footy["Away"].replace(FOOTYSTATS_TEAM_MAP)

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

                # Normalizar nomes para o merge
                df_footy["Norm_Home"] = df_footy["Home"].apply(normalize_team_name)
                df_footy["Norm_Away"] = df_footy["Away"].apply(normalize_team_name)

                # Remover Home/Away de cols_to_merge para evitar colunas duplicadas no merge
                cols_to_merge_filtered = [c for c in cols_to_merge if c not in ["Home", "Away"]]

                footy_subset = df_footy[cols_to_merge_filtered + ["Norm_Home", "Norm_Away"]].copy()
                footy_subset = footy_subset.drop_duplicates(subset=["Date", "Norm_Home", "Norm_Away"], keep="last")

                missing_footy_cols = [c for c in cols_to_merge_filtered if c not in df.columns]
                if source_path == hist_path or missing_footy_cols:
                    df = pd.merge(df, footy_subset, on=["Date", "Norm_Home", "Norm_Away"], how="left")
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
def load_today_games(_cache_version=CACHE_VERSION):
    data_day_dir = PROJECT_ROOT / "data_day"
    if data_day_dir.exists():
        # Listar arquivos e ordenar por data de modificação (mais recentes primeiro)
        files = [f for f in os.listdir(data_day_dir) if f.endswith(".csv") and f.startswith("dados_day_betfair")]
        if files:
            # Ordenar arquivos para que os mais novos (geralmente mais completos) venham primeiro
            files.sort(
                key=lambda x: os.path.getmtime(os.path.join(data_day_dir, x)),
                reverse=True,
            )

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

                # Priorizar linhas com mais dados (menos zeros) antes de remover duplicatas
                # Contar quantos valores não são zero nas colunas de odds principais
                odds_cols = [c for c in df.columns if "Odd_" in c]
                df["non_zero_count"] = (df[odds_cols] > 0).sum(axis=1)
                df = df.sort_values("non_zero_count", ascending=False)

                df = df.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
                df = df.drop(columns=["non_zero_count"])

                # Adicionar nomes normalizados para busca robusta
                df["Norm_Home"] = df["Home"].apply(normalize_team_name)
                df["Norm_Away"] = df["Away"].apply(normalize_team_name)
                df["Match"] = df["Home"].astype(str) + " vs " + df["Away"].astype(str)

                return df
    return pd.DataFrame()


# Funções Utilitárias de Formatação
format_minutes = lay0x1_core.format_minutes


def get_last_10_team_summary(df_games, team_name, target_score):
    team_norm = normalize_team_name(team_name)
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
        # Filtrar jogos em que o placar HT, do ponto de vista do time analisado,
        # é ht_score[0] (gols pró) x ht_score[1] (gols contra)
        def match_ht_score(row):
            if row["Norm_Home"] == team_norm:
                return row["Goals_H_HT"] == ht_score[0] and row["Goals_A_HT"] == ht_score[1]
            else:
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
            # Retorna (gols_pro_time, gols_contra_time) até o minuto 75
            if row["Norm_Home"] == team_norm:
                return count_goals_until(row["Min_Goals_H"], 75), count_goals_until(row["Min_Goals_A"], 75)
            else:
                return count_goals_until(row["Min_Goals_A"], 75), count_goals_until(row["Min_Goals_H"], 75)

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


normalize_goal_minute = lay0x1_core.normalize_goal_minute
count_goals_until = lay0x1_core.count_goals_until
count_goals_after = lay0x1_core.count_goals_after
get_h2h_stats = lay0x1_core.get_h2h_stats
get_goal_interval_stats = lay0x1_core.get_goal_interval_stats
build_poisson_timing_scenario = lay0x1_core.build_poisson_timing_scenario
analyze_goal_timing = lay0x1_core.analyze_goal_timing


calculate_pro_metrics = lay0x1_core.calculate_pro_metrics


# Interface
st.title("🛡️ Lay 0x1 Ultimate - Professional Trading Tool")
main_tab, bets_tab = st.tabs(["🛡️ Lay 0x1 Ultimate", "🧾 Planilha"])

df_hist = load_data()
df_today = load_today_games()

if df_hist.empty:
    st.error("Não foi possível carregar os dados históricos.")
    st.stop()

# Sidebar
st.sidebar.header("🔍 Busca & Filtros")
date_selected = st.sidebar.date_input(
    "Filtrar Jogos do Dia",
    value=DEFAULT_DATE,
)

leagues = sorted(df_hist["League"].unique().tolist())
selected_leagues = st.sidebar.multiselect("Ligas", leagues, default=leagues[:5])

with main_tab:
    if not df_today.empty:
        df_day_filtered = df_today[(df_today["Date"].dt.date == date_selected)]
        if selected_leagues:
            df_day_filtered = df_day_filtered[df_day_filtered["League"].isin(selected_leagues)]

        if not df_day_filtered.empty:
            df_day_filtered = df_day_filtered.copy()
            st.subheader(f"📅 Jogos Encontrados em {date_selected}")

            selected_match = st.selectbox(
                "Selecione o Jogo para Análise Profunda",
                df_day_filtered["Match"].tolist(),
            )

            m_data = df_day_filtered[df_day_filtered["Match"] == selected_match].iloc[0]
            results = calculate_pro_metrics(df_hist, m_data["Home"], m_data["Away"], m_data)

            if results:
                st.markdown("---")
                st.subheader("📝 Registrar Aposta")
                with st.form(key=f"bet_form_lay01_{m_data['Home']}_{m_data['Away']}"):
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        odd_entrada = st.number_input(
                            "Odd Entrada",
                            min_value=1.01,
                            value=float(m_data.get("Odd_CS_0x1_Lay", 0) or 1.01),
                            step=0.01,
                        )
                    with f2:
                        valor_aposta = st.number_input(
                            "Valor da Aposta (R$)",
                            min_value=0.01,
                            value=10.0,
                            step=1.0,
                        )
                    with f3:
                        responsabilidade = calculate_lay_liability(odd_entrada, valor_aposta)
                        st.metric("Responsabilidade", f"R$ {responsabilidade:.2f}")

                    entrada_tipo = st.selectbox("Entrada", ["Pre-Live", "Ao Vivo"])

                    submitted = st.form_submit_button("Apostar", use_container_width=True)
                    if submitted:
                        df_bets = load_lay_bets()
                        new_bet = pd.DataFrame(
                            [
                                {
                                    "data": str(pd.to_datetime(m_data["Date"]).date()),
                                    "mandante": m_data["Home"],
                                    "visitante": m_data["Away"],
                                    "hora": m_data.get("Time", ""),
                                    "mercado": "Lay 0x1",
                                    "odd_entrada": round(odd_entrada, 2),
                                    "valor_aposta": round(valor_aposta, 2),
                                    "responsabilidade": round(responsabilidade, 2),
                                    "entrada": entrada_tipo,
                                    "saida": "",
                                    "odd_saida_75min": np.nan,
                                    "resultado": np.nan,
                                    "percentual_resultado": np.nan,
                                },
                            ],
                        )
                        df_bets = pd.concat([df_bets, new_bet], ignore_index=True)
                        save_lay_bets(df_bets)
                        st.success("Entrada registrada com sucesso na planilha.")

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
                    odd_btts_back = m_data.get("Odd_BTTS_Yes_Back", 0)
                    odd_btts_lay = m_data.get("Odd_BTTS_Yes_Lay", 0)
                    if odd_btts_back > 0:
                        st.write(f"Back: {odd_btts_back:.2f}")
                        st.write(f"Lay: {odd_btts_lay:.2f}")
                    else:
                        st.write("Back: <span style='color: #888;'>Indisponível</span>", unsafe_allow_html=True)
                        st.write("Lay: <span style='color: #888;'>Indisponível</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with o4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Correct Score 0x1**")
                    odd_cs_back = m_data.get("Odd_CS_0x1_Back", 0)
                    odd_cs_lay = m_data.get("Odd_CS_0x1_Lay", 0)
                    if odd_cs_back > 0:
                        st.write(f"Back: {odd_cs_back:.2f}")
                        st.write(
                            f"Lay: <span class='highlight-red'>{odd_cs_lay:.2f}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write("Back: <span style='color: #888;'>Indisponível</span>", unsafe_allow_html=True)
                        st.write("Lay: <span style='color: #888;'>Indisponível</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with o5:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Valor (EV Heurístico)**")
                    odd_lay = m_data.get("Odd_CS_0x1_Lay", 0)
                    if odd_lay > 0:
                        ev = (results["heuristic_success"] / 100) * 1 - (1 - results["heuristic_success"] / 100) * (odd_lay - 1)
                        st.write(
                            f"EV: <span class='{'highlight-green' if ev > 0 else 'highlight-red'}'>{ev:+.2f}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write("EV: <span style='color: #888;'>Indisponível</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ESTRATÉGIA VENCEDORA
                st.markdown("---")
                if m_data.get("Odd_CS_0x1_Lay", 0) == 0:
                    st.warning("⚠️ **Atenção:** Odds de Correct Score 0x1 não encontradas para este jogo no arquivo de hoje. A análise de Score e Valor (EV/CLV) está limitada.")
                if results["sample_warning"]:
                    st.warning(
                        f"⚠️ **Atenção:** Amostra histórica pequena para este confronto. Mandante: {results['sample_home']} jogos | Visitante: {results['sample_away']} jogos | Qualidade: {results['sample_quality']}.",
                    )

                st.subheader("🎯 Recomendação de Estratégia Lay 0x1")

                res_col, reasons_col = st.columns([1, 2])

                with res_col:
                    color = "#00ff88" if "FORTE" in results["recommendation"] else "#ffcc00" if "MODERADA" in results["recommendation"] else "#ff4b4b"
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; color: black; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="margin:0;">{results["recommendation"]}</h2>
                            <p style="margin:0; font-weight: bold;">Score do Sinal: {results["score"]}/{results["max_score"]}</p>
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

                t_stats_h, t_stats_a, t_combined_scores = analyze_goal_timing(df_hist, m_data["Home"], m_data["Away"])

                # H2H
                h2h_stats = get_h2h_stats(df_hist, m_data["Home"], m_data["Away"])

                # CRITÉRIOS DE SAÍDA E GESTÃO DE RISCO (MÉTODO GET UP / LUKE)
                st.markdown("---")
                st.subheader("🚪 Gestão de Risco e Critérios de Saída (In-Play)")

                # Análise de Timing para Saída Dinâmica
                avg_match_2h = t_stats_h["avg_match_2h_00ht"] if t_stats_h else 75
                risk_plan = build_risk_plan(avg_match_2h, 75, results["sample_quality"], results["score"], results["recommendation"])

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.success("✅ **REGRA DE SAÍDA PRIMÁRIA**")
                    st.write(f"**Hedge alvo:** {risk_plan['adjusted_exit']}'")
                    st.write(f"**Base histórica:** {risk_plan['base_exit']}'")
                    st.write(f"**Confiança da amostra:** {risk_plan['confidence']}")

                with r2:
                    st.warning("⚠️ **ZONA DE DECISÃO**")
                    st.write(f"**Faixa de risco:** {risk_plan['risk_band']}")
                    st.write("**0x0:** manter até o alvo, se houver pressão e o jogo estiver vivo.")
                    st.write("**0x1 no HT:** stop loss antecipado, não insistir no rolo do mercado.")

                with r3:
                    st.error("🛑 **STOP LOSS OPERACIONAL**")
                    st.write("**Se o jogo travar:** reduzir antes do limite.")
                    st.write("**Se a odd fugir:** não esperar melhora artificial.")
                    st.write("**Se o mandante não pressionar:** sair antes do alvo.")

                st.info(
                    f"""
                    💡 **Plano objetivo de saída**
                    - Recomendação atual: **{risk_plan["recommendation"]}**
                    - Hedge sugerido: **{risk_plan["adjusted_exit"]}'**
                    - Leitura da amostra: **{results["sample_quality"]}**
                    - Regra prática: {risk_plan["hedge_note"]}
                    - Stop: {risk_plan["stop_note"]}
                    """,
                )

                # CLEAN SHEET DO VISITANTE
                st.markdown("---")
                st.subheader("🧹 Clean Sheet — Risco do 0x1")
                cs1, cs2 = st.columns(2)
                with cs1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cs_home = results["home"]["clean_sheet_pct"]
                    st.metric(f"Clean Sheet — {m_data['Home']}", f"{cs_home:.1f}%")
                    st.caption("% de jogos em que o mandante não sofreu gol. Quanto maior, menor o risco de o visitante marcar.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with cs2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cs_away = results["away"]["clean_sheet_pct"]
                    st.metric(f"Clean Sheet — {m_data['Away']}", f"{cs_away:.1f}%")
                    st.caption("% de jogos em que o visitante não sofreu gol. Quanto maior, maior o risco de ele manter o 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # HEAD-TO-HEAD
                st.markdown("---")
                st.subheader("⚔️ Head-to-Head (H2H)")
                if h2h_stats:
                    h2h_col1, h2h_col2 = st.columns([1, 2])
                    with h2h_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.write(f"**Total de confrontos:** {h2h_stats['total']}")
                        st.write(f"**Placar 0x1:** {h2h_stats['score_0x1']} ({h2h_stats['score_0x1_pct']:.1f}%)")
                        st.write(f"**Placar 1x0:** {h2h_stats['score_1x0']} ({(h2h_stats['score_1x0'] / h2h_stats['total']) * 100:.1f}%)")
                        st.write(f"**Placar 0x0:** {h2h_stats['score_0x0']} ({(h2h_stats['score_0x0'] / h2h_stats['total']) * 100:.1f}%)")
                        cs_risk = "🔴 Alto" if h2h_stats["score_0x1_pct"] > 10 else "🟡 Moderado" if h2h_stats["score_0x1_pct"] > 5 else "🟢 Baixo"
                        st.write(f"**Risco H2H 0x1:** {cs_risk}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with h2h_col2:
                        h2h_scores_df = pd.DataFrame(
                            list(h2h_stats["top_scores"].items()),
                            columns=["Placar FT", "Freq %"],
                        ).sort_values("Freq %", ascending=False)
                        fig_h2h = px.bar(
                            h2h_scores_df,
                            x="Placar FT",
                            y="Freq %",
                            text_auto=".1f",
                            color="Freq %",
                            color_continuous_scale="Viridis",
                            template="plotly_dark",
                            title="Placares Mais Frequentes no H2H",
                        )
                        fig_h2h.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_h2h, use_container_width=True)
                else:
                    st.info("Nenhum confronto direto encontrado no histórico.")

                # PROBABILIDADE DE GOL POR INTERVALO DE 15 MINUTOS
                st.markdown("---")
                st.subheader("⏱️ Probabilidade de Gol por Intervalo de 15 Minutos")
                interval_stats = get_goal_interval_stats(df_hist, m_data["Home"], m_data["Away"])

                tab_int_h, tab_int_a = st.tabs([f"🏠 {m_data['Home']}", f"🚀 {m_data['Away']}"])

                def render_interval_chart(attack_data, combined_data, team_name, sample):
                    intervals_labels = list(attack_data.keys())
                    fig_int = go.Figure()
                    fig_int.add_trace(
                        go.Bar(
                            x=intervals_labels,
                            y=list(attack_data.values()),
                            name="Gols Marcados",
                            marker_color="#00ff88",
                        ),
                    )
                    fig_int.add_trace(
                        go.Bar(
                            x=intervals_labels,
                            y=list(combined_data.values()),
                            name="Qualquer Gol na Partida",
                            marker_color="#4a9eff",
                            opacity=0.7,
                        ),
                    )
                    fig_int.update_layout(
                        title=f"{team_name} — % de jogos com gol no intervalo (amostra: {sample} jogos)",
                        xaxis_title="Intervalo",
                        yaxis_title="% de Jogos",
                        template="plotly_dark",
                        barmode="group",
                        height=380,
                    )
                    st.plotly_chart(fig_int, use_container_width=True)

                with tab_int_h:
                    render_interval_chart(
                        interval_stats["home_attack"],
                        interval_stats["home_combined"],
                        m_data["Home"],
                        interval_stats["home_sample"],
                    )
                with tab_int_a:
                    render_interval_chart(
                        interval_stats["away_attack"],
                        interval_stats["away_combined"],
                        m_data["Away"],
                        interval_stats["away_sample"],
                    )

                st.caption("**Gols Marcados:** % de jogos em que o time analisado marcou naquele intervalo. **Qualquer Gol:** % em que houve pelo menos um gol na partida (qualquer time) naquele intervalo.")

                # DASHBOARD PRINCIPAL (POISSON E VOLATILIDADE)
                st.markdown("---")
                st.subheader("📈 Análise Quantitativa e In-Play")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(f"{results['poisson_0x1_label']}", f"{results['poisson_0x1']:.2f}%")
                    st.write(f"{results['heuristic_success_label']}: {results['heuristic_success']:.1f}%")
                    st.caption("A leitura acima é heurística, não probabilidade calibrada.")
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

                st.markdown("---")
                st.subheader("🧠 Índice de Força FootyStats - Lay 0x1")
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(f"Índice Anti-0x1 - {m_data['Home']}", f"{results['lay_strength_home']:.1f}")
                    st.write(f"Classificação: **{results['lay_strength_home_label']}**")
                    st.caption("Elite/Forte: perfil muito favorável ao Lay 0x1 | Moderado: exige confirmação | Baixo: risco maior de placar magro.")
                    st.write("Foco em pressão ofensiva, evitar zero gol e sustentar jogo aberto.")
                    st.markdown("---")
                    st.metric("Variância do Índice (Mandante)", f"{results['lay_var_home']:.1f}")
                    st.write(f"Consistência: **{results['lay_var_home_label']}**")
                    st.caption(results["lay_var_home_desc"])
                    st.markdown("</div>", unsafe_allow_html=True)
                with s2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(f"Índice Anti-0x1 - {m_data['Away']}", f"{results['lay_strength_away']:.1f}")
                    st.write(f"Classificação: **{results['lay_strength_away_label']}**")
                    st.caption("Elite/Forte: perfil muito favorável ao Lay 0x1 | Moderado: exige confirmação | Baixo: risco maior de placar magro.")
                    st.write("Leitura da capacidade do visitante participar de um jogo menos estático.")
                    st.markdown("---")
                    st.metric("Variância do Índice (Visitante)", f"{results['lay_var_away']:.1f}")
                    st.write(f"Consistência: **{results['lay_var_away_label']}**")
                    st.caption(results["lay_var_away_desc"])
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("🔥 Performance nos Últimos 10 Jogos")
                p1, p2 = st.columns(2)
                with p1:
                    last10 = results["last10_home"]
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"**{m_data['Home']}**")
                    st.write(f"Forma: {last10['form_sequence']}")
                    st.write(f"Campanha: {last10['record']}")
                    st.write(f"Pontos: {last10['points']}/{last10['max_points']} | Win Rate: {last10['win_rate']:.1f}%")
                    st.write(f"Placares 0x1: {last10['target_score_count']} em {last10['games_analyzed']} jogos")
                    st.write("---")
                    st.write(f"0x0 HT: {last10['ht_00']['total']} | Gol mandante até 75': {last10['ht_00']['home_goal_to_75']} | Gol visitante até 75': {last10['ht_00']['away_goal_to_75']}")
                    st.write(f"Permaneceu 0x0 até 75': {last10['ht_00']['stayed_score_to_75']} | Outro placar: {last10['ht_00']['changed_score_to_75']}")
                    st.write(f"0x1 HT: {last10['ht_01']['total']} | Gol mandante até 75': {last10['ht_01']['home_goal_to_75']} | Gol visitante até 75': {last10['ht_01']['away_goal_to_75']}")
                    st.write(f"Permaneceu 0x1 até 75': {last10['ht_01']['stayed_score_to_75']} | Outro placar: {last10['ht_01']['changed_score_to_75']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with p2:
                    last10 = results["last10_away"]
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"**{m_data['Away']}**")
                    st.write(f"Forma: {last10['form_sequence']}")
                    st.write(f"Campanha: {last10['record']}")
                    st.write(f"Pontos: {last10['points']}/{last10['max_points']} | Win Rate: {last10['win_rate']:.1f}%")
                    st.write(f"Placares 0x1: {last10['target_score_count']} em {last10['games_analyzed']} jogos")
                    st.write("---")
                    st.write(f"0x0 HT: {last10['ht_00']['total']} | Gol mandante até 75': {last10['ht_00']['home_goal_to_75']} | Gol visitante até 75': {last10['ht_00']['away_goal_to_75']}")
                    st.write(f"Permaneceu 0x0 até 75': {last10['ht_00']['stayed_score_to_75']} | Outro placar: {last10['ht_00']['changed_score_to_75']}")
                    st.write(f"0x1 HT: {last10['ht_01']['total']} | Gol mandante até 75': {last10['ht_01']['home_goal_to_75']} | Gol visitante até 75': {last10['ht_01']['away_goal_to_75']}")
                    st.write(f"Permaneceu 0x1 até 75': {last10['ht_01']['stayed_score_to_75']} | Outro placar: {last10['ht_01']['changed_score_to_75']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Métrica de Chutes por Gol
                st.markdown("---")
                st.subheader("🎯 Eficiência de Finalização (Chutes por Gol)")
                col_sh1, col_sh2 = st.columns(2)
                with col_sh1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    home_shots_value = results["home"]["shots_per_goal"]
                    st.metric(
                        f"{results['home']['shots_per_goal_label']} - {m_data['Home']}",
                        f"{home_shots_value:.1f}" if pd.notna(home_shots_value) else "N/D",
                    )
                    st.write(results["home"]["shots_per_goal_desc"])
                    st.markdown("</div>", unsafe_allow_html=True)
                with col_sh2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    away_shots_value = results["away"]["shots_per_goal"]
                    st.metric(
                        f"{results['away']['shots_per_goal_label']} - {m_data['Away']}",
                        f"{away_shots_value:.1f}" if pd.notna(away_shots_value) else "N/D",
                    )
                    st.write(results["away"]["shots_per_goal_desc"])
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("⏳ Poisson por Tempo Após 75'")

                scenario_00_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 0))
                scenario_00_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 0))
                scenario_01_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 1))
                scenario_01_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 1))

                def render_poisson_time_block(title, home_scenario, away_scenario):
                    st.markdown(f"#### {title}")
                    b1, b2 = st.columns(2)

                    with b1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.write(f"**{m_data['Home']}**")
                        if home_scenario:
                            st.write(f"Cenário: {home_scenario['scenario_label']}")
                            st.write(f"Amostra: {home_scenario['sample_size']} jogos")
                            st.write(f"P(gol do mandante até 90'): {home_scenario['prob_home_goal']:.1f}%")
                            st.write(f"P(gol na partida até 90'): {home_scenario['prob_match_goal']:.1f}%")
                        else:
                            st.write("Dados insuficientes para este cenário.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    with b2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.write(f"**{m_data['Away']}**")
                        if away_scenario:
                            st.write(f"Cenário: {away_scenario['scenario_label']}")
                            st.write(f"Amostra: {away_scenario['sample_size']} jogos")
                            st.write(f"P(gol do visitante até 90'): {away_scenario['prob_away_goal']:.1f}%")
                            st.write(f"P(gol na partida até 90'): {away_scenario['prob_match_goal']:.1f}%")
                        else:
                            st.write("Dados insuficientes para este cenário.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    fig = go.Figure()
                    if home_scenario:
                        fig.add_trace(
                            go.Scatter(
                                x=home_scenario["timeline"]["Minute"],
                                y=home_scenario["timeline"]["Match"],
                                mode="lines+markers",
                                name=f"{m_data['Home']} - Partida",
                                line=dict(color="#00ff88"),
                            ),
                        )
                    if away_scenario:
                        fig.add_trace(
                            go.Scatter(
                                x=away_scenario["timeline"]["Minute"],
                                y=away_scenario["timeline"]["Match"],
                                mode="lines+markers",
                                name=f"{m_data['Away']} - Partida",
                                line=dict(color="#ff4b4b"),
                            ),
                        )

                    if fig.data:
                        fig.update_layout(
                            title=f"Probabilidade acumulada de gol após 75' - {title}",
                            xaxis_title="Minuto",
                            yaxis_title="Probabilidade (%)",
                            template="plotly_dark",
                            height=380,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Sem amostra suficiente para calcular a distribuição deste cenário.")

                render_poisson_time_block("Cenário 0x0 aos 75'", scenario_00_home, scenario_00_away)
                render_poisson_time_block("Cenário 0x1 aos 75'", scenario_01_home, scenario_01_away)

                # --- NOVA SEÇÃO: ANÁLISE QUANTITATIVA DE TIMING ---
                st.markdown("---")
                st.subheader("⏱️ Análise de Timing e Explosão (First Goal)")

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
                    norm_h = normalize_team_name(m_data["Home"])
                    h_goals = results["h_games"].apply(
                        lambda r: r["Goals_H_FT"] if normalize_team_name(r["Home"]) == norm_h else r["Goals_A_FT"],
                        axis=1,
                    )
                    fig_h = go.Figure()
                    fig_h.add_trace(
                        go.Scatter(
                            y=h_goals,
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
                    norm_a = normalize_team_name(m_data["Away"])
                    a_goals = results["a_games"].apply(
                        lambda r: r["Goals_A_FT"] if normalize_team_name(r["Away"]) == norm_a else r["Goals_H_FT"],
                        axis=1,
                    )
                    fig_a = go.Figure()
                    fig_a.add_trace(
                        go.Scatter(
                            y=a_goals,
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

with bets_tab:
    st.subheader("🧾 Planilha de Apostas")
    df_bets_all = load_lay_bets()
    if df_bets_all.empty:
        st.info("Nenhuma entrada registrada ainda.")
    else:
        st.dataframe(style_bets_dataframe(df_bets_all.iloc[::-1]), use_container_width=True)

        options_map = {build_bet_label(idx, row): idx for idx, row in df_bets_all.iloc[::-1].iterrows()}
        selected_bet_label = st.selectbox(
            "Selecione uma entrada para atualizar saída ou excluir",
            list(options_map.keys()),
        )
        selected_bet_idx = options_map[selected_bet_label]
        selected_bet = df_bets_all.loc[selected_bet_idx]

        e1, e2, e3 = st.columns(3)
        with e1:
            saida_edit = st.selectbox(
                "Saída",
                BET_STATUS_OPTIONS,
                index=BET_STATUS_OPTIONS.index(selected_bet.get("saida", "") if selected_bet.get("saida", "") in BET_STATUS_OPTIONS else ""),
                key=f"saida_edit_lay01_{selected_bet_idx}",
            )
        with e2:
            odd_saida_default = 0.0 if pd.isna(selected_bet.get("odd_saida_75min")) else float(selected_bet.get("odd_saida_75min", 0.0))
            odd_saida_edit = st.number_input(
                "Odd Saída 75min",
                min_value=0.0,
                value=odd_saida_default,
                step=0.01,
                disabled=saida_edit != "75min",
                key=f"odd_saida_edit_lay01_{selected_bet_idx}",
            )
        with e3:
            resultado_preview = calculate_lay_result(
                float(selected_bet["odd_entrada"]),
                float(selected_bet["valor_aposta"]),
                saida_edit,
                odd_saida_edit,
            )
            st.metric(
                "Resultado",
                f"R$ {resultado_preview:.2f}" if pd.notna(resultado_preview) else "Pendente",
            )

        a1, a2 = st.columns(2)
        with a1:
            if st.button("Salvar Saída", use_container_width=True, key=f"save_exit_lay01_{selected_bet_idx}"):
                df_bets_all.loc[selected_bet_idx, "saida"] = saida_edit
                df_bets_all.loc[selected_bet_idx, "odd_saida_75min"] = round(odd_saida_edit, 2) if saida_edit == "75min" else np.nan
                df_bets_all.loc[selected_bet_idx, "resultado"] = round(resultado_preview, 2) if pd.notna(resultado_preview) else np.nan
                df_bets_all.loc[selected_bet_idx, "percentual_resultado"] = round((resultado_preview / float(selected_bet["responsabilidade"])) * 100, 2) if pd.notna(resultado_preview) and float(selected_bet["responsabilidade"]) > 0 else np.nan
                save_lay_bets(df_bets_all)
                st.success("Saída atualizada com sucesso.")
        with a2:
            if st.button("Excluir Entrada", use_container_width=True, key=f"delete_bet_lay01_{selected_bet_idx}"):
                df_bets_all = df_bets_all.drop(index=selected_bet_idx).reset_index(drop=True)
                save_lay_bets(df_bets_all)
                st.success("Entrada excluída com sucesso.")

        st.markdown("---")
        st.markdown("#### 📊 Estatísticas da Planilha")
        settled_bets = df_bets_all[df_bets_all["resultado"].notna()].copy()
        total_bets = len(df_bets_all)
        settled_count = len(settled_bets)
        greens_count = int((settled_bets["resultado"] > 0).sum())
        reds_count = int((settled_bets["resultado"] < 0).sum())
        total_stake = float(df_bets_all["valor_aposta"].fillna(0).sum())
        total_liability = float(df_bets_all["responsabilidade"].fillna(0).sum())
        total_result = float(settled_bets["resultado"].fillna(0).sum())
        avg_odd = float(df_bets_all["odd_entrada"].fillna(0).mean()) if total_bets else 0.0
        avg_result = float(settled_bets["resultado"].fillna(0).mean()) if settled_count else 0.0
        roi_stake = (total_result / total_stake) * 100 if total_stake > 0 else 0.0
        roi_liability = (total_result / total_liability) * 100 if total_liability > 0 else 0.0
        green_rate = (greens_count / settled_count) * 100 if settled_count > 0 else 0.0

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("ROI Stake", f"{roi_stake:+.1f}%")
        with m2:
            st.metric("ROI Respons.", f"{roi_liability:+.1f}%")
        with m3:
            st.metric("Acumulado", f"R$ {total_result:.2f}")
        with m4:
            st.metric("Greens", str(greens_count))
        with m5:
            st.metric("Reds", str(reds_count))
        with m6:
            st.metric("Odd Média", f"{avg_odd:.2f}")

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Volume**")
            st.write(f"Entradas totais: {total_bets}")
            st.write(f"Apostas liquidadas: {settled_count}")
            st.markdown("</div>", unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Capital Exposto**")
            st.write(f"Stake total: R$ {total_stake:.2f}")
            st.write(f"Responsabilidade total: R$ {total_liability:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with s3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Eficiência**")
            st.write(f"Taxa de green: {green_rate:.1f}%")
            st.write(f"Resultado médio: R$ {avg_result:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with s4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Leitura Rápida**")
            st.write(f"Melhor base %: {roi_liability:+.1f}% sobre responsabilidade")
            st.write("Mercado dominante: Lay 0x1")
            st.markdown("</div>", unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.caption("FutStats PRO v4.0 | Estratégias Avançadas In-Play")
