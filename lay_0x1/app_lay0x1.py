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
from ui_constants import BET_STATUS_OPTIONS, NOTE_PRIORITY_OPTIONS, NOTE_STATUS_OPTIONS
from ui_helpers import (
    display_scores,
    render_audit_section,
    render_badge,
    render_callout,
    render_edit_note_form,
    render_exec_summary,
    render_interval_chart,
    render_kpi_comparison,
    render_last10_card,
    render_metric_card,
    render_new_note_form,
    render_note_card,
    render_poisson_time_block,
    render_role_profile,
    render_section_header,
    render_stat_grid,
    render_variance_section,
)

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
load_today_games = load_today_games_raw
build_risk_plan = lay0x1_core.build_risk_plan


# Funções Utilitárias de Formatação
format_minutes = lay0x1_core.format_minutes
get_last_10_team_summary = lay0x1_core.get_last_10_team_summary
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
selected_leagues = st.sidebar.multiselect("Ligas", leagues, default=[])

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

with st.sidebar.expander("Diagnóstico jogos do dia", expanded=False):
    if df_today.empty:
        st.caption("df_today está vazio.")
    else:
        st.dataframe(
            df_today[["Date", "Home", "Away", "Match"]].head(10),
            use_container_width=True,
            hide_index=True,
        )

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
                        lay_success_prob = max(0.0, 1 - (results["heuristic_success"] / 100))
                        ev = lay_success_prob * 1 - (1 - lay_success_prob) * (odd_lay - 1)
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
                    recommendation_badge = "ok" if color == "#00ff88" else "warn" if color == "#ffcc00" else "bad"
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
                    render_badge(results["recommendation"], recommendation_badge)

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
                    render_callout(
                        "success",
                        "REGRA DE SAÍDA PRIMÁRIA",
                        [
                            f"Hedge alvo: {risk_plan['adjusted_exit']}'",
                            f"Base histórica: {risk_plan['base_exit']}'",
                            f"Confiança da amostra: {risk_plan['confidence']}",
                        ],
                    )

                with r2:
                    render_callout(
                        "warning",
                        "ZONA DE DECISÃO",
                        [
                            f"Faixa de risco: {risk_plan['risk_band']}",
                            "0x0: manter até o alvo, se houver pressão e o jogo estiver vivo.",
                            "0x1 no HT: stop loss antecipado, não insistir no rolo do mercado.",
                        ],
                    )

                with r3:
                    render_callout(
                        "error",
                        "STOP LOSS OPERACIONAL",
                        [
                            "Se o jogo travar: reduzir antes do limite.",
                            "Se a odd fugir: não esperar melhora artificial.",
                            "Se o mandante não pressionar: sair antes do alvo.",
                        ],
                    )

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
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("0x0 HT", f"{results['red_from_00']:.2f}%")
                    st.write("Heurística Poisson HT")
                    st.caption("Leitura heurística de terminar o 1º tempo sem gols.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("0x0 FT", f"{results['pct_00_ft']:.2f}%")
                    st.write("Heurística Poisson FT")
                    st.caption("Leitura histórica de terminar sem gols no jogo.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("0x1 HT", f"{results['pct_01_ht']:.2f}%")
                    st.write("Heurística Poisson HT")
                    st.caption("Leitura histórica de terminar o 1º tempo em 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)

                c4, c5, c6 = st.columns(3)
                with c4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("0x1 FT", f"{results['pct_01_ft']:.2f}%")
                    st.write("Heurística Poisson FT")
                    st.caption("Leitura histórica de terminar o jogo em 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c5:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Qualquer outro HT", f"{results['pct_other_ht']:.2f}%")
                    st.write("Heurística Poisson HT")
                    st.caption("Qualquer placar no intervalo diferente de 0x0 e 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c6:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Qualquer outro FT", f"{results['pct_other_ft']:.2f}%")
                    st.write("Heurística Poisson FT")
                    st.caption("Qualquer placar final diferente de 0x0 e 0x1.")
                    st.markdown("</div>", unsafe_allow_html=True)

                signal_color = "#00ff88" if results["poisson_0x1"] <= 7 else "#ffd56a" if results["poisson_0x1"] <= 12 else "#ff4b4b"
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 5px solid {signal_color};">
                        <div class="section-kicker">Leitura prática</div>
                        <div class="panel-title">Como interpretar o cenário</div>
                        <div style="margin:0.15rem 0 0.35rem 0;">0x0 HT: {results["pct_00_ht"]:.2f}% | 0x1 HT: {results["pct_01_ht"]:.2f}% | outros HT: {results["pct_other_ht"]:.2f}%</div>
                        <div style="margin:0.15rem 0 0.35rem 0;">0x0 FT: {results["pct_00_ft"]:.2f}% | 0x1 FT: {results["pct_01_ft"]:.2f}% | outros FT: {results["pct_other_ft"]:.2f}%</div>
                        <span class="badge {"ok" if results["poisson_0x1"] <= 7 else "warn" if results["poisson_0x1"] <= 12 else "bad"}">Sinal 0x1 HT: {results["poisson_0x1"]:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown('<div class="section-kicker">Modelos e projeções</div><div class="section-heading">Índice de Força FootyStats - Lay 0x1</div>', unsafe_allow_html=True)
                s1, s2 = st.columns(2)
                with s1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="panel-title">{m_data["Home"]}</div>', unsafe_allow_html=True)
                    st.metric("Índice Anti-0x1", f"{results['lay_strength_home']:.1f}")
                    home_strength_badge = "ok" if "Forte" in results["lay_strength_home_label"] or "Elite" in results["lay_strength_home_label"] else "warn" if "Moderado" in results["lay_strength_home_label"] else "bad"
                    render_badge(results["lay_strength_home_label"], home_strength_badge)
                    st.caption("Força ofensiva e chance de sustentar um jogo aberto.")
                    st.markdown("---")
                    st.metric("Variância", f"{results['lay_var_home']:.1f}")
                    var_home_badge = "ok" if "Baixa" in results["lay_var_home_label"] else "warn" if "Média" in results["lay_var_home_label"] else "bad"
                    render_badge(results["lay_var_home_label"], var_home_badge)
                    st.caption("Consistência do perfil.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with s2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="panel-title">{m_data["Away"]}</div>', unsafe_allow_html=True)
                    st.metric("Índice Anti-0x1", f"{results['lay_strength_away']:.1f}")
                    away_strength_badge = "ok" if "Forte" in results["lay_strength_away_label"] or "Elite" in results["lay_strength_away_label"] else "warn" if "Moderado" in results["lay_strength_away_label"] else "bad"
                    render_badge(results["lay_strength_away_label"], away_strength_badge)
                    st.caption("Capacidade de manter o jogo menos travado.")
                    st.markdown("---")
                    st.metric("Variância", f"{results['lay_var_away']:.1f}")
                    var_away_badge = "ok" if "Baixa" in results["lay_var_away_label"] else "warn" if "Média" in results["lay_var_away_label"] else "bad"
                    render_badge(results["lay_var_away_label"], var_away_badge)
                    st.caption("Consistência do perfil.")
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

                render_exec_summary(results["role_profile_home"], results["role_profile_away"], results["last10_home"], results["last10_away"])

                c_last_home, c_last_away = st.columns(2)
                with c_last_home:
                    render_last10_card(results["last10_home"], m_data["Home"], "#00ff88")
                with c_last_away:
                    render_last10_card(results["last10_away"], m_data["Away"], "#4a9eff")

                st.markdown("### 📋 Mandante x Visitante")

                st.markdown('<div class="section-kicker">Perfil Mandante x Visitante</div><div class="section-heading">Comparativo estrutural</div>', unsafe_allow_html=True)
                rp_home, rp_away = st.columns(2)
                with rp_home:
                    render_role_profile(results["role_profile_home"], m_data["Home"], "Mandante", "#00ff88", results["home"]["avg_xg"])
                with rp_away:
                    render_role_profile(results["role_profile_away"], m_data["Away"], "Visitante", "#4a9eff", results["away"]["avg_xg"])

                render_section_header("Comparação Visual", "KPIs lado a lado")
                kpi_cols = st.columns(3)
                with kpi_cols[0]:
                    render_kpi_comparison("Gols marcados", results["role_profile_home"]["goals_for"], results["role_profile_away"]["goals_for"], m_data["Home"], m_data["Away"], fmt="{:.0f}")
                with kpi_cols[1]:
                    render_kpi_comparison("Marcou 1º gol", results["role_profile_home"]["first_goal"]["scored_first_pct"], results["role_profile_away"]["first_goal"]["scored_first_pct"], m_data["Home"], m_data["Away"], fmt="{:.1f}%", is_percent=True)
                with kpi_cols[2]:
                    render_kpi_comparison("xG médio", results["home"]["avg_xg"], results["away"]["avg_xg"], m_data["Home"], m_data["Away"], fmt="{:.2f}")

                with st.expander("Ver auditoria detalhada", expanded=False):
                    render_audit_section(df_hist, results, m_data["Home"], m_data["Away"], normalize_team_name)

                with st.expander("Eficiência de Finalização", expanded=False):
                    render_section_header("Eficiência operacional", "Conversão por finalização")
                    col_sh1, col_sh2 = st.columns(2)
                    with col_sh1:
                        home_shots_value = results["home"]["shots_per_goal"]
                        render_metric_card(
                            f"{results['home']['shots_per_goal_label']} - {m_data['Home']}",
                            [f"Valor: {home_shots_value:.1f}" if pd.notna(home_shots_value) else "Valor: N/D", results["home"]["shots_per_goal_desc"]],
                        )
                    with col_sh2:
                        away_shots_value = results["away"]["shots_per_goal"]
                        render_metric_card(
                            f"{results['away']['shots_per_goal_label']} - {m_data['Away']}",
                            [f"Valor: {away_shots_value:.1f}" if pd.notna(away_shots_value) else "Valor: N/D", results["away"]["shots_per_goal_desc"]],
                        )

                with st.expander("Poisson por Tempo Após 75'", expanded=False):
                    render_section_header("Distribuição temporal e cenários", 'Probabilidade acumulada pós-75"')
                    scenario_00_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 0))
                    scenario_00_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 0))
                    scenario_01_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 1))
                    scenario_01_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 1))

                    render_poisson_time_block("Cenário 0x0 aos 75'", scenario_00_home, scenario_00_away, m_data["Home"], m_data["Away"])
                    render_poisson_time_block("Cenário 0x1 aos 75'", scenario_01_home, scenario_01_away, m_data["Home"], m_data["Away"])

                with st.expander("Timing e Placares", expanded=False):
                    render_section_header("Leitura de minutagem e placares", "Minuto do 1º gol e placares recorrentes")
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
                    render_section_header("Variância e probabilidade", "Eficiência, variância e matriz")
                    render_variance_section(results, m_data["Home"], m_data["Away"], normalize_team_name)

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

        render_stat_grid(
            [
                ("ROI Stake", f"{roi_stake:+.1f}%", None),
                ("ROI Respons.", f"{roi_liability:+.1f}%", None),
                ("Acumulado", f"R$ {total_result:.2f}", None),
                ("Greens", str(greens_count), None),
                ("Reds", str(reds_count), None),
                ("Odd Média", f"{avg_odd:.2f}", None),
            ],
            columns=6,
        )

        render_stat_grid(
            [
                ("Volume", f"{total_bets}", f"Apostas liquidadas: {settled_count}"),
                ("Capital Exposto", f"R$ {total_stake:.2f}", f"Responsabilidade total: R$ {total_liability:.2f}"),
                ("Eficiência", f"{green_rate:.1f}%", f"Resultado médio: R$ {avg_result:.2f}"),
                ("Leitura Rápida", f"{roi_liability:+.1f}%", "Mercado dominante: Lay 0x1"),
            ],
            columns=4,
        )

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
                render_note_card(row)
                image_path = str(row.get("image_path", "")).strip()
                if image_path and Path(image_path).exists():
                    st.image(image_path, caption="Printscreen anexado", use_container_width=True)
    else:
        st.info("Nenhuma nota criada ainda.")

    st.markdown("---")
    new_note_form = render_new_note_form(NOTE_PRIORITY_OPTIONS, NOTE_STATUS_OPTIONS)

    add_col, clear_col = st.columns(2)
    with add_col:
        if st.button("Salvar Nota", use_container_width=True):
            if new_note_form["title"].strip() or new_note_form["note"].strip():
                now = pd.Timestamp.now().isoformat(timespec="seconds")
                note_id = new_note_id()
                new_row = pd.DataFrame(
                    [
                        {
                            "id": note_id,
                            "created_at": now,
                            "updated_at": now,
                            "title": new_note_form["title"].strip(),
                            "note": new_note_form["note"].strip(),
                            "tag": new_note_form["tag"].strip(),
                            "priority": new_note_form["priority"],
                            "status": new_note_form["status"],
                            "pinned": bool(new_note_form["pinned"]),
                            "image_path": save_note_attachment(new_note_form["image"], note_id) if new_note_form["image"] is not None else "",
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
        editable_order = df_notes.sort_values(by=["pinned", "updated_at"], ascending=[False, False]).reset_index(drop=True)
        note_options = {f"{idx + 1}. {row['title'] or 'Sem título'} | {row['priority']} | {row['status']}": row["id"] for idx, row in editable_order.iterrows()}
        selected_note_label = st.selectbox("Selecionar nota", list(note_options.keys()))
        selected_note_id = note_options[selected_note_label]
        selected_idx = df_notes.index[df_notes["id"] == selected_note_id][0]
        selected_note = df_notes.loc[selected_idx]
        edit_form = render_edit_note_form(selected_note, NOTE_PRIORITY_OPTIONS, NOTE_STATUS_OPTIONS, selected_note_id)

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("Salvar Alterações", use_container_width=True):
                if edit_form["image"] is not None:
                    saved_image_path = save_note_attachment(edit_form["image"], selected_note_id)
                    df_notes.loc[selected_idx, "image_path"] = saved_image_path
                df_notes.loc[selected_idx, "title"] = edit_form["title"].strip()
                df_notes.loc[selected_idx, "tag"] = edit_form["tag"].strip()
                df_notes.loc[selected_idx, "priority"] = edit_form["priority"]
                df_notes.loc[selected_idx, "status"] = edit_form["status"]
                df_notes.loc[selected_idx, "pinned"] = bool(edit_form["pinned"])
                df_notes.loc[selected_idx, "note"] = edit_form["note"].strip()
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
