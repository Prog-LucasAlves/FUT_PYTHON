from pathlib import Path

import lay0x1_core
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from app_config import DEFAULT_DATE, UNKNOWN_TEAMS_LOG_FILE
from bet_tracker_utils import (
    build_bet_label,
    calculate_lay_liability,
    calculate_lay_result,
    load_lay_bets,
    save_lay_bets,
    style_bets_dataframe,
)
from data_utils import (
    load_historical_data,
    load_resolved_unknown_team_names,
    load_unknown_team_names,
    save_resolved_unknown_team_names,
)
from data_utils import (
    load_today_games as load_today_games_raw,
)
from notes_utils import load_notes, new_note_id, save_note_attachment, save_notes
from scipy.stats import poisson
from ui_constants import BET_STATUS_OPTIONS, NOTE_PRIORITY_OPTIONS, NOTE_STATUS_OPTIONS

# Configuração da Página
st.set_page_config(page_title="Lay 0x1 PRO - FutStats", page_icon="📈", layout="wide")

normalize_team_name = lay0x1_core.normalize_team_name


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
    .panel-title {
        font-size: 0.9rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa4b2;
        margin-bottom: 0.35rem;
    }
    .badge {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        margin: 0 0.35rem 0.35rem 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(255,255,255,0.08);
        color: #e8eef7;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .badge.ok { background: rgba(0,255,136,0.14); color: #7dffb7; border-color: rgba(0,255,136,0.22); }
    .badge.warn { background: rgba(255,193,7,0.14); color: #ffd56a; border-color: rgba(255,193,7,0.24); }
    .badge.info { background: rgba(74,158,255,0.14); color: #93c7ff; border-color: rgba(74,158,255,0.24); }
    .sticky-hero {
        position: sticky;
        top: 0;
        z-index: 999;
        backdrop-filter: blur(10px);
        background: rgba(11, 13, 17, 0.82);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .section-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: #7f8ca3;
        margin-bottom: 0.2rem;
    }
    .section-heading {
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        color: #f3f6fb;
    }
    .score-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        font-weight: 700;
    }
    .hero-score {
        font-size: 1.25rem;
        font-weight: 800;
        color: #f7fafc;
        margin: 0.15rem 0 0.35rem 0;
    }
    .compact-panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 12px 14px;
    }
    .highlight-green { color: #00ff88; font-weight: bold; }
    .highlight-red { color: #ff4b4b; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Funções de Carregamento de Dados
load_data = st.cache_data(ttl=3599)(load_historical_data)
load_today_games = st.cache_data(ttl=600)(load_today_games_raw)
build_risk_plan = lay0x1_core.build_risk_plan


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
main_tab, bets_tab, notes_tab = st.tabs(["🛡️ Lay 0x1 Ultimate", "🧾 Planilha", "🗒️ Notas"])

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

with st.sidebar.expander("Nomes desconhecidos", expanded=False):
    df_unknown_teams = load_unknown_team_names(limit=30)
    if df_unknown_teams.empty:
        st.caption("Nenhum nome novo foi registrado.")
    else:
        st.dataframe(df_unknown_teams, use_container_width=True, hide_index=True)
        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("Marcar resolvido", use_container_width=True):
                current = load_resolved_unknown_team_names()
                current.extend(df_unknown_teams["name"].astype(str).tolist())
                save_resolved_unknown_team_names(current)
                st.success("Nomes marcados como resolvidos.")
                st.rerun()
        with action_col2:
            csv_export = df_unknown_teams.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Exportar CSV",
                data=csv_export,
                file_name="nomes_desconhecidos.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_col3:
            if st.button("Limpar log", use_container_width=True):
                UNKNOWN_TEAMS_LOG_FILE.write_text("", encoding="utf-8")
                st.success("Log limpo com sucesso.")
                st.rerun()

with main_tab:
    if not df_today.empty:
        df_day_filtered = df_today[(df_today["Date"].dt.date == date_selected)]
        if selected_leagues:
            df_day_filtered = df_day_filtered[df_day_filtered["League"].isin(selected_leagues)]

        if not df_day_filtered.empty:
            df_day_filtered = df_day_filtered.copy()
            st.markdown(
                f"""
                <div class="sticky-hero">
                    <div class="panel-title">Jogos Encontrados</div>
                    <h3 style="margin:0;">{date_selected}</h3>
                    <div class="badge info">{len(df_day_filtered)} jogos disponíveis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            selected_match = st.selectbox(
                "Selecione o Jogo para Análise Profunda",
                df_day_filtered["Match"].tolist(),
            )

            m_data = df_day_filtered[df_day_filtered["Match"] == selected_match].iloc[0]
            results = calculate_pro_metrics(df_hist, m_data["Home"], m_data["Away"], m_data)

            if results:
                st.markdown("---")
                st.markdown('<div class="section-kicker">Ação principal</div><div class="section-heading">Registrar Aposta</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="panel-title">Aposta atual</div>
                    <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.35rem;">{m_data["Home"]} x {m_data["Away"]}</div>
                    <div class="badge info">Lay 0x1</div>
                    <div class="badge ok">{m_data.get("League", "")}</div>
                    <div class="badge warn">{str(m_data.get("Date", "")).split(" ")[0]}</div>
                    """,
                    unsafe_allow_html=True,
                )
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
                st.markdown("</div>", unsafe_allow_html=True)

                # DASHBOARD DE ODDS
                st.markdown("---")
                st.markdown('<div class="section-kicker">Análises de apoio</div><div class="section-heading">Monitoramento de Odds de Mercado</div>', unsafe_allow_html=True)
                o1, o2, o3, o4, o5 = st.columns(5)
                with o1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">Match Odds</div>', unsafe_allow_html=True)
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
                    st.markdown('<div class="panel-title">Over 2.5 Goals</div>', unsafe_allow_html=True)
                    st.write(f"Back: {m_data.get('Odd_Over25_FT_Back', 0):.2f}")
                    st.write(f"Lay: {m_data.get('Odd_Over25_FT_Lay', 0):.2f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with o3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">BTTS (Yes)</div>', unsafe_allow_html=True)
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
                    st.markdown('<div class="panel-title">Correct Score 0x1</div>', unsafe_allow_html=True)
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
                    st.markdown('<div class="panel-title">Valor (EV)</div>', unsafe_allow_html=True)
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
                    st.warning("Odds de Correct Score 0x1 não encontradas no arquivo de hoje. A leitura de Score e Valor fica limitada.")
                if results["sample_warning"]:
                    st.warning(
                        f"Amostra histórica pequena para este confronto. Mandante: {results['sample_home']} jogos | Visitante: {results['sample_away']} jogos | Qualidade: {results['sample_quality']}.",
                    )

                st.markdown('<div class="section-kicker">Bloco principal</div><div class="section-heading">Recomendação de Estratégia Lay 0x1</div>', unsafe_allow_html=True)

                res_col, reasons_col = st.columns([1, 2])

                with res_col:
                    color = "#00ff88" if "FORTE" in results["recommendation"] else "#ffcc00" if "MODERADA" in results["recommendation"] else "#ff4b4b"
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 5px solid {color}; color: black; text-align: center;">
                            <div class="section-kicker">Sinal final</div>
                            <h2 style="margin:0;">{results["recommendation"]}</h2>
                            <p style="margin:0; font-weight: bold;">Score do Sinal: {results["score"]}/{results["max_score"]}</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with reasons_col:
                    st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">Por que esta indicação?</div>', unsafe_allow_html=True)
                    for reason in results["reasons"]:
                        st.write(f"✅ {reason}")
                    st.markdown("</div>", unsafe_allow_html=True)

                t_stats_h, t_stats_a, t_combined_scores = analyze_goal_timing(df_hist, m_data["Home"], m_data["Away"])

                # H2H
                h2h_stats = get_h2h_stats(df_hist, m_data["Home"], m_data["Away"])

                # CRITÉRIOS DE SAÍDA E GESTÃO DE RISCO (MÉTODO GET UP / LUKE)
                st.markdown("---")
                st.markdown('<div class="section-kicker">Análises de apoio</div><div class="section-heading">Critérios de Saída (In-Play)</div>', unsafe_allow_html=True)

                # Análise de Timing para Saída Dinâmica
                avg_match_2h = t_stats_h["avg_match_2h_00ht"] if t_stats_h else 75
                risk_plan = build_risk_plan(avg_match_2h, 75, results["sample_quality"], results["score"], results["recommendation"])

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.success("REGRA DE SAÍDA PRIMÁRIA")
                    st.write(f"**Hedge alvo:** {risk_plan['adjusted_exit']}'")
                    st.write(f"**Base histórica:** {risk_plan['base_exit']}'")
                    st.write(f"**Confiança da amostra:** {risk_plan['confidence']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                with r2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.warning("ZONA DE DECISÃO")
                    st.write(f"**Faixa de risco:** {risk_plan['risk_band']}")
                    st.write("**0x0:** manter até o alvo, se houver pressão e o jogo estiver vivo.")
                    st.write("**0x1 no HT:** stop loss antecipado, não insistir no rolo do mercado.")
                    st.markdown("</div>", unsafe_allow_html=True)

                with r3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.error("STOP LOSS OPERACIONAL")
                    st.write("**Se o jogo travar:** reduzir antes do limite.")
                    st.write("**Se a odd fugir:** não esperar melhora artificial.")
                    st.write("**Se o mandante não pressionar:** sair antes do alvo.")
                    st.markdown("</div>", unsafe_allow_html=True)

                st.info(
                    f"""
                    Plano objetivo de saída
                    - Recomendação atual: **{risk_plan["recommendation"]}**
                    - Hedge sugerido: **{risk_plan["adjusted_exit"]}'**
                    - Leitura da amostra: **{results["sample_quality"]}**
                    - Regra prática: {risk_plan["hedge_note"]}
                    - Stop: {risk_plan["stop_note"]}
                    """,
                )

                # CLEAN SHEET DO VISITANTE
                st.markdown("---")
                st.markdown('<div class="section-kicker">Análises de apoio</div><div class="section-heading">Clean Sheet e risco do 0x1</div>', unsafe_allow_html=True)
                cs1, cs2 = st.columns(2)
                with cs1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cs_home = results["home"]["clean_sheet_pct"]
                    st.metric(f"Clean Sheet — {m_data['Home']}", f"{cs_home:.1f}%")
                    st.caption("Quanto maior o clean sheet, menor o risco do visitante marcar.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with cs2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    cs_away = results["away"]["clean_sheet_pct"]
                    st.metric(f"Clean Sheet — {m_data['Away']}", f"{cs_away:.1f}%")
                    st.caption("Quanto maior o clean sheet, maior a chance de sustentar o 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # HEAD-TO-HEAD
                st.markdown("---")
                st.markdown('<div class="section-kicker">Análises de apoio</div><div class="section-heading">Head-to-Head (H2H)</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="section-kicker">Análises de apoio</div><div class="section-heading">Probabilidade de gol por intervalo de 15 minutos</div>', unsafe_allow_html=True)
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
                        title=f"{team_name} — jogos com gol no intervalo (amostra: {sample})",
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

                st.caption("Gols Marcados = % de jogos em que o time marcou no intervalo. Qualquer Gol = % de jogos com ao menos um gol no intervalo.")

                # DASHBOARD PRINCIPAL (POISSON E VOLATILIDADE)
                st.markdown("---")
                st.markdown('<div class="section-kicker">Modelos e projeções</div><div class="section-heading">Análise Quantitativa e In-Play</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(f"{results['poisson_0x1_label']}", f"{results['poisson_0x1']:.2f}%")
                    st.write(f"{results['heuristic_success_label']}: {results['heuristic_success']:.1f}%")
                    st.caption("Leitura heurística, não probabilidade calibrada.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Cenário: 0x0 no HT**")
                    st.write(f"Risco de Red: {results['red_from_00']:.1f}%")
                    st.info("Chance de o 0-0 sobreviver ao 2º tempo.")
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
                st.markdown('<div class="section-kicker">Modelos e projeções</div><div class="section-heading">Índice de Força FootyStats - Lay 0x1</div>', unsafe_allow_html=True)
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(f"Índice Anti-0x1 - {m_data['Home']}", f"{results['lay_strength_home']:.1f}")
                    st.write(f"Classificação: **{results['lay_strength_home_label']}**")
                    st.caption("Elite/Forte: perfil muito favorável | Moderado: pede confirmação | Baixo: risco maior de placar magro.")
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
                    st.caption("Elite/Forte: perfil muito favorável | Moderado: pede confirmação | Baixo: risco maior de placar magro.")
                    st.write("Leitura da capacidade do visitante participar de um jogo menos estático.")
                    st.markdown("---")
                    st.metric("Variância do Índice (Visitante)", f"{results['lay_var_away']:.1f}")
                    st.write(f"Consistência: **{results['lay_var_away_label']}**")
                    st.caption(results["lay_var_away_desc"])
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<div class="section-kicker">Histórico aprofundado</div><div class="section-heading">Resumo Histórico</div>', unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="sticky-hero">
                        <div class="panel-title">Lay 0x1 Ultimate</div>
                        <h3 style="margin:0;">{m_data["Home"]} x {m_data["Away"]}</h3>
                        <div class="badge ok">Base: dados_historicos.csv</div>
                        <div class="badge info">{m_data["League"]}</div>
                        <div class="badge warn">{str(m_data.get("Date", "")).split(" ")[0]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                def render_exec_summary(home_profile, away_profile, home_last10, away_last10):
                    st.markdown('<div class="section-kicker">Resumo Executivo</div><div class="section-heading">Leitura rápida do confronto</div>', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    cards = [
                        (c1, "📅 Jogos", f"{home_profile['sample_size']} / {away_profile['sample_size']}", "ok"),
                        (c2, "⚡ PPG", f"{home_profile['ppg_season']:.2f} / {away_profile['ppg_season']:.2f}", "info"),
                        (c3, "🥅 Gols", f"{home_profile['goals_for']} / {away_profile['goals_for']}", "warn"),
                        (c4, "🎯 1º Gol", f"{home_profile['first_goal']['scored_first_pct']:.1f}% / {away_profile['first_goal']['scored_first_pct']:.1f}%", "ok"),
                    ]
                    for col, title, value, badge in cards:
                        with col:
                            st.metric(title, value)
                            st.markdown(f'<span class="badge {badge}">executivo</span>', unsafe_allow_html=True)
                    st.caption(f"Base histórica: `dados_historicos.csv` | Últimos 10: {home_last10['record']} vs {away_last10['record']}")
                    st.caption("Leituras de 1º gol dependem da precisão dos minutos registrados em `Min_Goals_H` e `Min_Goals_A`.")

                    home_score = round(
                        (home_profile["ppg_season"] * 25) + (home_profile["first_goal"]["scored_first_pct"] * 0.45) + (home_profile["goals_for"] / max(home_profile["sample_size"], 1)),
                        1,
                    )
                    away_score = round(
                        (away_profile["ppg_season"] * 25) + (away_profile["first_goal"]["scored_first_pct"] * 0.45) + (away_profile["goals_for"] / max(away_profile["sample_size"], 1)),
                        1,
                    )
                    score_gap = abs(home_score - away_score)
                    if score_gap >= 6:
                        score_label, score_color = "Confortável", "ok"
                    elif score_gap >= 3:
                        score_label, score_color = "Moderado", "warn"
                    else:
                        score_label, score_color = "Aperto", "info"
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 5px solid {"#00ff88" if score_color == "ok" else "#ffd56a" if score_color == "warn" else "#4a9eff"};">
                            <div class="section-kicker">Score Executivo</div>
                            <div class="hero-score">{home_score:.1f} x {away_score:.1f}</div>
                            <span class="badge {score_color}">{score_label}</span>
                            <span class="badge info">Semáforo automático</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                def render_last10_card(last10, team_name, accent_color):
                    spark_map = {"V": "█", "E": "▇", "D": "▁"}
                    sparkline = "".join(spark_map.get(x, "·") for x in last10["form_sequence"].replace(" | ", ""))
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 5px solid {accent_color};">
                            <div class="section-kicker" style="margin-bottom:0.25rem;">Últimos 10 jogos</div>
                            <h4 style="margin:0 0 8px 0;">{team_name}</h4>
                            <p style="margin:0;"><b>Forma:</b> {last10["form_sequence"]}</p>
                            <p style="margin:0;"><b>Campanha:</b> {last10["record"]}</p>
                            <p style="margin:0;"><b>Pontos:</b> {last10["points"]}/{last10["max_points"]} | <b>Win rate:</b> {last10["win_rate"]:.1f}%</p>
                            <p style="margin:0;"><b>0x1 FT:</b> {last10["target_score_count"]} em {last10["games_analyzed"]} jogos</p>
                            <p style="margin:0;"><b>Spark:</b> <span style="letter-spacing:0.14em;">{sparkline}</span></p>
                            <p style="margin:0;"><b>0x0 HT -> 75':</b> {last10["ht_00"]["stayed_score_to_75"]} estáveis | {last10["ht_00"]["changed_score_to_75"]} mudaram</p>
                            <p style="margin:0;"><b>0x1 HT -> 75':</b> {last10["ht_01"]["stayed_score_to_75"]} estáveis | {last10["ht_01"]["changed_score_to_75"]} mudaram</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                render_exec_summary(results["role_profile_home"], results["role_profile_away"], results["last10_home"], results["last10_away"])

                c_last_home, c_last_away = st.columns(2)
                with c_last_home:
                    render_last10_card(results["last10_home"], m_data["Home"], "#00ff88")
                with c_last_away:
                    render_last10_card(results["last10_away"], m_data["Away"], "#4a9eff")

                st.markdown("### 📋 Mandante x Visitante")

                def render_role_profile(profile, team_name, role_label, accent_color):
                    first_goal = profile["first_goal"]
                    role_kicker = "Mandante" if role_label.lower().startswith("mand") else "Visitante"
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left: 5px solid {accent_color};">
                            <div class="section-kicker">Perfil histórico</div>
                            <div class="section-heading" style="margin-bottom:0.5rem;">📌 {team_name} - {role_label}</div>
                            <span class="badge ok">Histórico</span>
                            <span class="badge info">Base: dados_historicos.csv</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    s1, s2 = st.columns(2)
                    stat_cards = [
                        (s1, "Jogos", profile["sample_size"], "ok"),
                        (s2, "Vitórias", profile["wins"], "ok"),
                    ]
                    for col, label, value, badge in stat_cards:
                        with col:
                            st.metric(label, f"{value:.2f}" if isinstance(value, (int, float)) else value)
                            st.markdown(f'<span class="badge {badge}">{role_kicker}</span>', unsafe_allow_html=True)
                    sg1, sg2, sg3 = st.columns(3)
                    quick_cards = [
                        (sg1, "PPG", profile["ppg_season"], "info"),
                        (sg2, "Gols pró", profile["goals_for"], "ok"),
                        (sg3, "Gols contra", profile["goals_against"], "warn"),
                    ]
                    for col, label, value, badge in quick_cards:
                        with col:
                            st.metric(label, f"{value:.2f}" if isinstance(value, (int, float)) else value)
                            st.markdown(f'<span class="badge {badge}">{role_kicker}</span>', unsafe_allow_html=True)
                    with st.expander("1º Gol", expanded=False):
                        fg_cols = st.columns(2)
                        fg_left = [
                            ("Marcou 1º gol", first_goal["scored_first_pct"]),
                            ("Marcou 1º gol e venceu", first_goal["scored_first_won_pct"]),
                            ("Marcou 1º gol e empatou", first_goal["scored_first_draw_pct"]),
                            ("Marcou 1º gol e perdeu", first_goal["scored_first_lost_pct"]),
                            ("Sofreu 1º gol", first_goal["suffered_first_pct"]),
                        ]
                        fg_right = [
                            ("Sofreu 1º gol e venceu", first_goal["suffered_first_won_pct"]),
                            ("Sofreu 1º gol e empatou", first_goal["suffered_first_draw_pct"]),
                            ("Sofreu 1º gol e perdeu", first_goal["suffered_first_lost_pct"]),
                            ("Marcou no 1º tempo", first_goal["first_goal_first_half_pct"]),
                            ("Marcou no 1º tempo e venceu", first_goal["first_goal_first_half_won_pct"]),
                        ]
                        with fg_cols[0]:
                            for label, value in fg_left:
                                st.write(f"{label}: {value:.1f}%")
                                st.progress(min(max(value / 100.0, 0.0), 1.0))
                        with fg_cols[1]:
                            for label, value in fg_right:
                                st.write(f"{label}: {value:.1f}%")
                                st.progress(min(max(value / 100.0, 0.0), 1.0))
                        if profile["sample_size"] > 0 and first_goal["scored_first_pct"] == 0.0:
                            st.info("Observação: 0.0% em 'marcou 1º gol' pode ocorrer quando o time não abriu o placar no recorte ou quando os minutos do gol estão incompletos.")

                def render_kpi_comparison(title, home_value, away_value, home_label, away_label, fmt="{:.2f}"):
                    st.markdown(f'<div class="section-kicker">{title}</div>', unsafe_allow_html=True)
                    left, right = st.columns(2)
                    max_value = max(float(home_value) if isinstance(home_value, (int, float)) else 0, float(away_value) if isinstance(away_value, (int, float)) else 0, 1.0)
                    with left:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"**{home_label}**")
                        st.write(fmt.format(home_value) if isinstance(home_value, (int, float)) else home_value)
                        st.progress(min(max((float(home_value) if isinstance(home_value, (int, float)) else 0) / max_value, 0.0), 1.0))
                        st.markdown("</div>", unsafe_allow_html=True)
                    with right:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"**{away_label}**")
                        st.write(fmt.format(away_value) if isinstance(away_value, (int, float)) else away_value)
                        st.progress(min(max((float(away_value) if isinstance(away_value, (int, float)) else 0) / max_value, 0.0), 1.0))
                        st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("")

                st.markdown('<div class="section-kicker">Perfil Mandante x Visitante</div><div class="section-heading">Comparativo estrutural</div>', unsafe_allow_html=True)
                rp_home, rp_away = st.columns(2)
                with rp_home:
                    render_role_profile(results["role_profile_home"], m_data["Home"], "Mandante", "#00ff88")
                with rp_away:
                    render_role_profile(results["role_profile_away"], m_data["Away"], "Visitante", "#4a9eff")

                st.markdown('<div class="section-kicker">Comparação Visual</div><div class="section-heading">KPIs lado a lado</div>', unsafe_allow_html=True)
                kpi_cols = st.columns(3)
                with kpi_cols[0]:
                    render_kpi_comparison("PPG da temporada", results["role_profile_home"]["ppg_season"], results["role_profile_away"]["ppg_season"], m_data["Home"], m_data["Away"])
                with kpi_cols[1]:
                    render_kpi_comparison("Gols marcados", results["role_profile_home"]["goals_for"], results["role_profile_away"]["goals_for"], m_data["Home"], m_data["Away"], fmt="{:.0f}")
                with kpi_cols[2]:
                    render_kpi_comparison("Marcou 1º gol", results["role_profile_home"]["first_goal"]["scored_first_pct"], results["role_profile_away"]["first_goal"]["scored_first_pct"], m_data["Home"], m_data["Away"])

                st.markdown('<div class="section-kicker">Conferência de dados</div><div class="section-heading">Auditoria dos Dados</div>', unsafe_allow_html=True)
                with st.expander("Ver auditoria detalhada", expanded=False):
                    audit_rows = []
                    audit_specs = [
                        ("Mandante", results["role_profile_home"], m_data["Home"], "home"),
                        ("Visitante", results["role_profile_away"], m_data["Away"], "away"),
                    ]
                    for label, profile, team_name, role in audit_specs:
                        expected = lay0x1_core.build_team_role_profile(df_hist, team_name, role, normalize_team_name)
                        audit_rows.extend(
                            [
                                {
                                    "Time": team_name,
                                    "Contexto": label,
                                    "Métrica": "Total de jogos",
                                    "Exibido": profile["sample_size"],
                                    "Base": expected["sample_size"],
                                    "Status": "OK" if profile["sample_size"] == expected["sample_size"] else "Divergente",
                                },
                                {
                                    "Time": team_name,
                                    "Contexto": label,
                                    "Métrica": "PPG temporada",
                                    "Exibido": round(profile["ppg_season"], 4),
                                    "Base": round(expected["ppg_season"], 4),
                                    "Status": "OK" if round(profile["ppg_season"], 4) == round(expected["ppg_season"], 4) else "Divergente",
                                },
                                {
                                    "Time": team_name,
                                    "Contexto": label,
                                    "Métrica": "Gols marcados",
                                    "Exibido": profile["goals_for"],
                                    "Base": expected["goals_for"],
                                    "Status": "OK" if profile["goals_for"] == expected["goals_for"] else "Divergente",
                                },
                                {
                                    "Time": team_name,
                                    "Contexto": label,
                                    "Métrica": "Gols sofridos",
                                    "Exibido": profile["goals_against"],
                                    "Base": expected["goals_against"],
                                    "Status": "OK" if profile["goals_against"] == expected["goals_against"] else "Divergente",
                                },
                                {
                                    "Time": team_name,
                                    "Contexto": label,
                                    "Métrica": "Marcou 1º gol %",
                                    "Exibido": round(profile["first_goal"]["scored_first_pct"], 4),
                                    "Base": round(expected["first_goal"]["scored_first_pct"], 4),
                                    "Status": "OK" if round(profile["first_goal"]["scored_first_pct"], 4) == round(expected["first_goal"]["scored_first_pct"], 4) else "Divergente",
                                },
                            ],
                        )
                    audit_df = pd.DataFrame(audit_rows)
                    audit_df["Semáforo"] = audit_df["Status"].map({"OK": "🟢", "Divergente": "🔴"}).fillna("🟡")
                    ok_count = int((audit_df["Status"] == "OK").sum()) if not audit_df.empty else 0
                    total_count = len(audit_df)
                    st.markdown('<div class="section-kicker">Resumo da auditoria</div>', unsafe_allow_html=True)
                    divergent_count = total_count - ok_count
                    coverage = (ok_count / total_count * 100) if total_count else 0.0
                    st.markdown(
                        f"""
                        <div class="compact-panel">
                            <span class="badge ok">OK: {ok_count}</span>
                            <span class="badge warn">Divergente: {divergent_count}</span>
                            <span class="badge info">Total: {total_count}</span>
                            <span class="badge {"ok" if coverage == 100 else "warn"}">Cobertura: {coverage:.0f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if total_count > 0:
                        st.progress(ok_count / total_count)
                    st.dataframe(audit_df, use_container_width=True, hide_index=True)
                    if not audit_df.empty and (audit_df["Status"] == "OK").all():
                        st.success("Auditoria concluída: os indicadores exibidos batem com o recálculo direto da base histórica.")
                    else:
                        st.warning("Auditoria encontrou divergências em pelo menos um indicador. Vale revisar a base ou o filtro aplicado.")

                with st.expander("Eficiência de Finalização", expanded=False):
                    st.markdown('<div class="section-kicker">Eficiência operacional</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-heading">Conversão por finalização</div>', unsafe_allow_html=True)
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

                with st.expander("Poisson por Tempo Após 75'", expanded=False):
                    st.markdown('<div class="section-kicker">Distribuição temporal e cenários</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-heading">Probabilidade acumulada pós-75"</div>', unsafe_allow_html=True)
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

                with st.expander("Timing e Placares", expanded=False):
                    st.markdown('<div class="section-kicker">Leitura de minutagem e placares</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-heading">Minuto do 1º gol e placares recorrentes</div>', unsafe_allow_html=True)
                    if t_stats_h and t_stats_a:
                        tsum1, tsum2, tsum3 = st.columns(3)
                        with tsum1:
                            st.metric("Mandante", format_minutes(t_stats_h["avg_first_team"]) if t_stats_h else "N/D", help="Média do 1º gol do time")
                        with tsum2:
                            st.metric("Visitante", format_minutes(t_stats_a["avg_first_team"]) if t_stats_a else "N/D", help="Média do 1º gol do time")
                        with tsum3:
                            st.metric("Amostra", f"{t_stats_h['sample_size']} jogos", help="Base usada no cálculo")

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

                        st.markdown("#### 📊 Distribuição do Minuto do Primeiro Gol")
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Box(y=t_stats_h["raw_first_team"], name=m_data["Home"], marker_color="#00ff88", boxpoints="all"))
                        fig_time.add_trace(go.Box(y=t_stats_a["raw_first_team"], name=m_data["Away"], marker_color="#ff4b4b", boxpoints="all"))
                        fig_time.update_layout(title="Boxplot: Quando o 1º gol costuma sair?", yaxis_title="Minuto", template="plotly_dark", height=400, showlegend=False)
                        fig_time.add_hline(y=45, line_dash="dash", line_color="white", annotation_text="Fim 1T")
                        st.plotly_chart(fig_time, use_container_width=True)

                        st.markdown("#### 🔢 Placares Mais Frequentes (%)")

                        def display_scores(scores_dict):
                            df_scores = pd.DataFrame(list(scores_dict.items()), columns=["Placar", "Freq %"]).sort_values("Freq %", ascending=False)
                            fig = px.bar(df_scores, x="Placar", y="Freq %", text_auto=".1f", color="Freq %", color_continuous_scale="Viridis", template="plotly_dark")
                            fig.update_layout(height=300, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                        tab_scores1, tab_scores2, tab_scores3 = st.tabs([f"🏠 {m_data['Home']}", f"🚀 {m_data['Away']}", "🤝 Confronto (Ambos)"])

                        with tab_scores1:
                            c1, c2 = st.columns(2)
                            with c1:
                                display_scores(t_stats_h["frequent_scores"]["HT"])
                            with c2:
                                display_scores(t_stats_h["frequent_scores"]["FT"])

                        with tab_scores2:
                            c1, c2 = st.columns(2)
                            with c1:
                                display_scores(t_stats_a["frequent_scores"]["HT"])
                            with c2:
                                display_scores(t_stats_a["frequent_scores"]["FT"])

                        with tab_scores3:
                            c1, c2 = st.columns(2)
                            with c1:
                                display_scores(t_combined_scores["HT"])
                            with c2:
                                display_scores(t_combined_scores["FT"])
                    else:
                        st.warning("Dados de minutagem insuficientes para este confronto.")

                with st.expander("Variância e Matriz Poisson", expanded=False):
                    st.markdown('<div class="section-kicker">Variância e probabilidade</div>', unsafe_allow_html=True)
                    st.markdown('<div class="section-heading">Eficiência, variância e matriz</div>', unsafe_allow_html=True)
                    vsum1, vsum2, vsum3 = st.columns(3)
                    with vsum1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(f"{m_data['Home']}", f"{results['home']['mean']:.2f}", help="Média esperada de gols")
                        st.caption(f"Variância: {results['home']['variance']:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with vsum2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(f"{m_data['Away']}", f"{results['away']['mean']:.2f}", help="Média esperada de gols")
                        st.caption(f"Variância: {results['away']['variance']:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with vsum3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Matriz", "Poisson", help="Probabilidades combinadas de placar")
                        st.caption("Grade de 0x0 a 4x4")
                        st.markdown("</div>", unsafe_allow_html=True)

                    v1, v2 = st.columns(2)
                    with v1:
                        st.write(f"### 🏠 {m_data['Home']}")
                        st.write(f"**Variância de Gols:** {results['home']['variance']:.3f}")
                        st.write(f"**Custo do Gol (Eficiência):** {results['home']['cost']:.3f}")
                        norm_h = normalize_team_name(m_data["Home"])
                        h_goals = results["h_games"].apply(
                            lambda r: r["Goals_H_FT"] if normalize_team_name(r["Home"]) == norm_h else r["Goals_A_FT"],
                            axis=1,
                        )
                        fig_h = go.Figure()
                        fig_h.add_trace(go.Scatter(y=h_goals, mode="lines+markers", name="Gols", line=dict(color="#00ff88")))
                        fig_h.update_layout(title="Histórico de Gols (Casa)", template="plotly_dark", height=250)
                        st.plotly_chart(fig_h, use_container_width=True)
                    with v2:
                        st.write(f"### 🚀 {m_data['Away']}")
                        st.write(f"**Variância de Gols:** {results['away']['variance']:.3f}")
                        st.write(f"**Custo do Gol (Eficiência):** {results['away']['cost']:.3f}")
                        norm_a = normalize_team_name(m_data["Away"])
                        a_goals = results["a_games"].apply(
                            lambda r: r["Goals_A_FT"] if normalize_team_name(r["Away"]) == norm_a else r["Goals_H_FT"],
                            axis=1,
                        )
                        fig_a = go.Figure()
                        fig_a.add_trace(go.Scatter(y=a_goals, mode="lines+markers", name="Gols", line=dict(color="#ff4b4b")))
                        fig_a.update_layout(title="Histórico de Gols (Fora)", template="plotly_dark", height=250)
                        st.plotly_chart(fig_a, use_container_width=True)

                    st.markdown("#### 🎲 Matriz de Probabilidades (Poisson)")
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

with notes_tab:
    st.subheader("🗒️ Bloco de Notas")
    st.caption("Tudo aqui é salvo automaticamente em arquivo local dentro da pasta `lay_0x1`.")
    df_notes = load_notes()

    top_left, top_right = st.columns([2, 1])
    with top_left:
        quick_note = st.text_input("Nota rápida", placeholder="Ex.: 0x0 forte no jogo de hoje")
    with top_right:
        quick_tag = st.text_input("Tag rápida", placeholder="Ex.: live, estudo")

    quick_a, quick_b, quick_c = st.columns(3)
    with quick_a:
        quick_priority = st.selectbox("Prioridade rápida", NOTE_PRIORITY_OPTIONS, index=1, key="quick_priority")
    with quick_b:
        quick_status = st.selectbox("Status rápido", NOTE_STATUS_OPTIONS, index=0, key="quick_status")
    with quick_c:
        quick_pin = st.checkbox("Fixar rápido", value=True, key="quick_pin")
    quick_image = st.file_uploader(
        "Printscreen rápido",
        type=["png", "jpg", "jpeg", "webp"],
        key="quick_image",
    )

    if st.button("Registrar nota rápida", use_container_width=True):
        if quick_note.strip():
            now = pd.Timestamp.now().isoformat(timespec="seconds")
            note_id = new_note_id()
            quick_row = pd.DataFrame(
                [
                    {
                        "id": note_id,
                        "created_at": now,
                        "updated_at": now,
                        "title": quick_note.strip()[:60],
                        "note": quick_note.strip(),
                        "tag": quick_tag.strip(),
                        "priority": quick_priority,
                        "status": quick_status,
                        "pinned": bool(quick_pin),
                        "image_path": save_note_attachment(quick_image, note_id) if quick_image is not None else "",
                    },
                ],
            )
            df_notes = pd.concat([df_notes, quick_row], ignore_index=True)
            save_notes(df_notes)
            st.success("Nota rápida registrada.")
            st.rerun()
        else:
            st.warning("Escreva ao menos o texto da nota rápida.")

    st.markdown("---")
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])
    with f1:
        search_text = st.text_input("Buscar", placeholder="título ou conteúdo")
    with f2:
        filter_tag = st.text_input("Filtrar tag", placeholder="ex.: live")
    with f3:
        filter_priority = st.selectbox("Filtrar prioridade", ["Todas"] + NOTE_PRIORITY_OPTIONS, index=0)
    with f4:
        filter_status = st.selectbox("Filtrar status", ["Todos"] + NOTE_STATUS_OPTIONS, index=0)

    if not df_notes.empty:
        filtered_notes = df_notes.copy()
        if search_text.strip():
            text_mask = filtered_notes["title"].astype(str).str.contains(search_text, case=False, na=False) | filtered_notes["note"].astype(str).str.contains(search_text, case=False, na=False)
            filtered_notes = filtered_notes[text_mask]
        if filter_tag.strip():
            filtered_notes = filtered_notes[filtered_notes["tag"].astype(str).str.contains(filter_tag, case=False, na=False)]
        if filter_priority != "Todas":
            filtered_notes = filtered_notes[filtered_notes["priority"] == filter_priority]
        if filter_status != "Todos":
            filtered_notes = filtered_notes[filtered_notes["status"] == filter_status]

        filtered_notes = filtered_notes.sort_values(
            by=["pinned", "updated_at", "priority"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

        cstats1, cstats2, cstats3, cstats4 = st.columns(4)
        with cstats1:
            st.metric("Total", len(df_notes))
        with cstats2:
            st.metric("Filtradas", len(filtered_notes))
        with cstats3:
            st.metric("Fixadas", int(df_notes["pinned"].sum()))
        with cstats4:
            st.metric("Urgentes", int((df_notes["priority"] == "Urgente").sum()))

        if filtered_notes.empty:
            st.info("Nenhuma nota encontrada com os filtros atuais.")
        else:
            st.markdown("### 📌 Cartões")
            for _, row in filtered_notes.iterrows():
                priority_colors = {
                    "Baixa": "#6c757d",
                    "Média": "#4a9eff",
                    "Alta": "#ff9f1c",
                    "Urgente": "#ff4b4b",
                }
                card_border = priority_colors.get(str(row["priority"]), "#4a9eff")
                pinned_label = "Fixada" if bool(row["pinned"]) else "Normal"
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(145deg, #1e2130, #161924);
                        border: 1px solid {card_border};
                        border-left: 8px solid {card_border};
                        border-radius: 14px;
                        padding: 16px 18px;
                        margin-bottom: 12px;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
                    ">
                        <div style="display:flex; justify-content:space-between; gap:12px; align-items:center;">
                            <div>
                                <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">{row["title"] or "Sem título"}</div>
                                <div style="color:#a9b4c3; margin-top:4px;">{row["note"] or "Sem conteúdo"} </div>
                            </div>
                            <div style="text-align:right; color:#d8deea; min-width:150px;">
                                <div><strong>Prioridade:</strong> {row["priority"]}</div>
                                <div><strong>Status:</strong> {row["status"]}</div>
                                <div><strong>Tag:</strong> {row["tag"] or "-"}</div>
                                <div><strong>Topo:</strong> {pinned_label}</div>
                            </div>
                        </div>
                        <div style="margin-top:10px; color:#8c97a8; font-size:0.82rem;">
                            Criada em {row["created_at"]} | Atualizada em {row["updated_at"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                image_path = str(row.get("image_path", "")).strip()
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="Printscreen anexado", use_container_width=True)
    else:
        st.info("Nenhuma nota criada ainda.")

    st.markdown("---")
    st.subheader("✍️ Nova Nota")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_title = st.text_input("Título", placeholder="Ex.: Jogo com risco alto no 1T")
    with c2:
        new_tag = st.text_input("Tag", placeholder="Ex.: risco, live, estudo")
    with c3:
        new_priority = st.selectbox("Prioridade", NOTE_PRIORITY_OPTIONS, index=1)

    n1, n2 = st.columns([2, 1])
    with n1:
        new_note = st.text_area("Conteúdo", placeholder="Escreva sua leitura, gatilhos, dúvidas e planos...", height=160)
    with n2:
        new_status = st.selectbox("Status", NOTE_STATUS_OPTIONS, index=0)
        new_pinned = st.checkbox("Fixar no topo", value=False)
        new_image = st.file_uploader(
            "Anexar printscreen",
            type=["png", "jpg", "jpeg", "webp"],
            key="new_note_image",
        )

    add_col, clear_col = st.columns(2)
    with add_col:
        if st.button("Salvar Nota", use_container_width=True):
            if new_title.strip() or new_note.strip():
                now = pd.Timestamp.now().isoformat(timespec="seconds")
                note_id = new_note_id()
                new_row = pd.DataFrame(
                    [
                        {
                            "id": note_id,
                            "created_at": now,
                            "updated_at": now,
                            "title": new_title.strip(),
                            "note": new_note.strip(),
                            "tag": new_tag.strip(),
                            "priority": new_priority,
                            "status": new_status,
                            "pinned": bool(new_pinned),
                            "image_path": save_note_attachment(new_image, note_id) if new_image is not None else "",
                        },
                    ],
                )
                df_notes = pd.concat([df_notes, new_row], ignore_index=True)
                save_notes(df_notes)
                st.success("Nota salva com sucesso.")
                st.rerun()
            else:
                st.warning("Preencha pelo menos o título ou o conteúdo da nota.")
    with clear_col:
        if st.button("Limpar Campos", use_container_width=True):
            st.rerun()

    if not df_notes.empty:
        st.markdown("---")
        st.subheader("🛠️ Editar Notas")
        editable_order = df_notes.sort_values(by=["pinned", "updated_at"], ascending=[False, False]).reset_index(drop=True)
        note_options = {f"{idx + 1}. {row['title'] or 'Sem título'} | {row['priority']} | {row['status']}": row["id"] for idx, row in editable_order.iterrows()}
        selected_note_label = st.selectbox("Selecionar nota", list(note_options.keys()))
        selected_note_id = note_options[selected_note_label]
        selected_idx = df_notes.index[df_notes["id"] == selected_note_id][0]
        selected_note = df_notes.loc[selected_idx]

        e1, e2, e3 = st.columns(3)
        with e1:
            edit_title = st.text_input("Título da nota", value=str(selected_note["title"]))
        with e2:
            edit_tag = st.text_input("Tag da nota", value=str(selected_note["tag"]))
        with e3:
            edit_priority = st.selectbox(
                "Prioridade da nota",
                NOTE_PRIORITY_OPTIONS,
                index=NOTE_PRIORITY_OPTIONS.index(str(selected_note["priority"])) if str(selected_note["priority"]) in NOTE_PRIORITY_OPTIONS else 1,
            )

        s1, s2 = st.columns(2)
        with s1:
            edit_status = st.selectbox(
                "Status da nota",
                NOTE_STATUS_OPTIONS,
                index=NOTE_STATUS_OPTIONS.index(str(selected_note["status"])) if str(selected_note["status"]) in NOTE_STATUS_OPTIONS else 0,
            )
        with s2:
            edit_pinned = st.checkbox("Fixada no topo", value=bool(selected_note["pinned"]))

        edit_note = st.text_area("Conteúdo da nota", value=str(selected_note["note"]), height=180)

        current_image_path = str(selected_note.get("image_path", "")).strip()
        if current_image_path and Path(current_image_path).exists():
            st.image(current_image_path, caption="Printscreen atual", use_container_width=True)

        edit_image = st.file_uploader(
            "Trocar printscreen",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"edit_note_image_{selected_note_id}",
        )

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Salvar Alterações", use_container_width=True):
                if edit_image is not None:
                    saved_image_path = save_note_attachment(edit_image, selected_note_id)
                    df_notes.loc[selected_idx, "image_path"] = saved_image_path
                df_notes.loc[selected_idx, "title"] = edit_title.strip()
                df_notes.loc[selected_idx, "tag"] = edit_tag.strip()
                df_notes.loc[selected_idx, "priority"] = edit_priority
                df_notes.loc[selected_idx, "status"] = edit_status
                df_notes.loc[selected_idx, "pinned"] = bool(edit_pinned)
                df_notes.loc[selected_idx, "note"] = edit_note.strip()
                df_notes.loc[selected_idx, "updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
                save_notes(df_notes)
                st.success("Nota atualizada com sucesso.")
                st.rerun()
        with a2:
            if st.button("Remover Printscreen", use_container_width=True):
                old_image_path = str(df_notes.loc[selected_idx, "image_path"]).strip()
                if old_image_path:
                    try:
                        old_path = Path(old_image_path)
                        if old_path.exists():
                            old_path.unlink()
                    except OSError:
                        pass
                df_notes.loc[selected_idx, "image_path"] = ""
                df_notes.loc[selected_idx, "updated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
                save_notes(df_notes)
                st.success("Printscreen removido com sucesso.")
                st.rerun()
        with a3:
            if st.button("Duplicar Nota", use_container_width=True):
                now = pd.Timestamp.now().isoformat(timespec="seconds")
                dup_row = selected_note.copy()
                dup_row["id"] = new_note_id()
                dup_row["created_at"] = now
                dup_row["updated_at"] = now
                dup_row["title"] = f"Cópia de {dup_row['title']}" if str(dup_row["title"]).strip() else "Cópia da nota"
                df_notes = pd.concat([df_notes, pd.DataFrame([dup_row])], ignore_index=True)
                save_notes(df_notes)
                st.success("Nota duplicada com sucesso.")
                st.rerun()
        delete_col, _ = st.columns([1, 2])
        with delete_col:
            if st.button("Excluir Nota", use_container_width=True):
                df_notes = df_notes.drop(index=selected_idx).reset_index(drop=True)
                save_notes(df_notes)
                st.success("Nota excluída com sucesso.")
                st.rerun()

# Rodapé
st.markdown("---")
st.caption("FutStats PRO v4.0 | Estratégias Avançadas In-Play")
