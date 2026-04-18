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


# ── DESIGN SYSTEM ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { background-color: #0a0c10; color: #c8d0dc; }
    [data-testid="stMetric"] { background: #12151c; border: 1px solid #1e2433; border-radius: 6px; padding: 8px 10px; }
    [data-testid="stMetricLabel"] { font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.06em; color: #4f5b6e !important; }
    [data-testid="stMetricValue"] { font-size: 1rem !important; font-weight: 700 !important; font-family: 'JetBrains Mono', monospace !important; }
    .metric-card { background: #12151c; border: 1px solid #1e2433; border-radius: 6px; padding: 10px 12px; }
    .strategy-card { background: #12151c; border-left: 4px solid #00ff88; padding: 10px 12px; margin: 4px 0; }
    .panel-title { font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: #4f5b6e; margin-bottom: 3px; }
    .badge { display: inline-block; padding: 2px 7px; margin: 0 3px 3px 0; border-radius: 4px; font-size: 0.68rem; font-weight: 600; background: rgba(255,255,255,0.04); color: #8e99a9; }
    .badge.ok { background: rgba(0,255,136,0.08); color: #3ddb8a; }
    .badge.warn { background: rgba(255,193,7,0.08); color: #e0b842; }
    .badge.info { background: rgba(74,158,255,0.08); color: #6aa8e0; }
    .section-kicker { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; color: #3d4a5c; margin-bottom: 2px; font-weight: 600; }
    .section-heading { font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; color: #d4dbe8; }
    .sticky-hero { position: sticky; top: 0; z-index: 999; backdrop-filter: blur(8px); background: rgba(10,12,16,0.92); border: 1px solid #161b26; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
    .hero-score { font-size: 1.05rem; font-weight: 800; color: #d4dbe8; }
    .score-chip { display: inline-flex; align-items: center; gap: 0.3rem; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.04); font-weight: 700; }
    .compact-panel { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 6px; padding: 10px 12px; }
    .highlight-green { color: #00ff88; font-weight: bold; }
    .highlight-red { color: #ff4b4b; font-weight: bold; }
    .sec-hdr { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.16em; color: #343e4f; margin: 14px 0 6px 0; border-bottom: 1px solid #151922; padding-bottom: 4px; font-weight: 700; }
    .tag { display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 0.68rem; font-weight: 600; margin-right: 3px; letter-spacing: 0.02em; }
    .tag-g { background: rgba(0,255,136,0.08); color: #3ddb8a; }
    .tag-y { background: rgba(255,193,7,0.08); color: #e0b842; }
    .tag-r { background: rgba(255,75,75,0.08); color: #e05a5a; }
    .tag-b { background: rgba(74,158,255,0.08); color: #6aa8e0; }
    .cpanel { background: #0e1118; border: 1px solid #171c27; border-radius: 6px; padding: 10px 12px; margin-bottom: 6px; }
    .drow { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #12151e; font-size: 0.8rem; }
    .drow:last-child { border-bottom: none; }
    .drow-k { color: #4f5b6e; }
    .drow-v { color: #c8d0dc; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }
    .mc-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: #4f5b6e; margin-bottom: 2px; font-weight: 600; }
    .verdict-score { font-family: 'JetBrains Mono', monospace; font-size: 2.4rem; font-weight: 800; line-height: 1; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Funções de Carregamento de Dados
load_data = st.cache_data(ttl=3599)(load_historical_data)
load_today_games = load_today_games_raw
build_risk_plan = lay0x1_core.build_risk_plan


@st.cache_data
def get_team_averages(df):
    return df.groupby("Home")["Goals_H_FT"].mean().to_dict()


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
st.title("🛡️ Lay 0x1 Ultimate")
main_tab, bets_tab, notes_tab = st.tabs(["🛡️ Análise", "🧾 Planilha", "🗒️ Notas"])

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

# ═══════════════════════════════════════════════
# MAIN TAB — ANÁLISE
# ═══════════════════════════════════════════════
with main_tab:
    if not df_today.empty:
        df_day_filtered = df_today[(df_today["Date"].dt.date == date_selected)]
        if selected_leagues:
            df_day_filtered = df_day_filtered[df_day_filtered["League"].isin(selected_leagues)]

        if not df_day_filtered.empty:
            df_day_filtered = df_day_filtered.copy()

            sel1, sel2 = st.columns([4, 1])
            with sel1:
                selected_match = st.selectbox(
                    "Jogo para Análise",
                    df_day_filtered["Match"].tolist(),
                )
            with sel2:
                st.markdown(
                    f'<div style="padding-top:28px;"><span class="tag tag-b">{len(df_day_filtered)} jogos</span> <span class="tag tag-b">{date_selected}</span></div>',
                    unsafe_allow_html=True,
                )

            m_data = df_day_filtered[df_day_filtered["Match"] == selected_match].iloc[0]
            results = calculate_pro_metrics(df_hist, m_data["Home"], m_data["Away"], m_data)

            if results:
                # ── COMPUTE ALL DATA UPFRONT ──
                t_stats_h, t_stats_a, t_combined_scores = analyze_goal_timing(df_hist, m_data["Home"], m_data["Away"])
                h2h_stats = get_h2h_stats(df_hist, m_data["Home"], m_data["Away"])
                avg_match_2h = t_stats_h["avg_match_2h_00ht"] if t_stats_h else 75
                risk_plan = build_risk_plan(avg_match_2h, 75, results["sample_quality"], results["score"], results["recommendation"])
                interval_stats = get_goal_interval_stats(df_hist, m_data["Home"], m_data["Away"])

                v_color = "#00ff88" if "FORTE" in results["recommendation"] else "#ffd54f" if "MODERADA" in results["recommendation"] else "#ff4b4b"
                v_tag = "tag-g" if "FORTE" in results["recommendation"] else "tag-y" if "MODERADA" in results["recommendation"] else "tag-r"

                # Warnings
                if results["sample_warning"]:
                    st.warning(f"Amostra pequena — H: {results['sample_home']} | A: {results['sample_away']} | {results['sample_quality']}")
                if m_data.get("Odd_CS_0x1_Lay", 0) == 0:
                    st.warning("Odds de Correct Score 0x1 indisponíveis.")

                # ════════════════════════════════════
                # 1. HERO VERDICT — FOCAL POINT
                # ════════════════════════════════════

                # --- PORTFOLIO LOGIC ---
                team_avg_goals = get_team_averages(df_hist)
                home_avg = team_avg_goals.get(m_data["Home"], 0)

                def get_bin(val, bins, labels):
                    for i in range(len(bins) - 1):
                        if bins[i] < val <= bins[i + 1]:
                            return labels[i]
                    return labels[-1]

                bin_h = get_bin(m_data["Odd_H_Back"], [1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 100], ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.1-2.5", "2.6-3.0", "3.0+"])
                bin_over = get_bin(m_data["Odd_Over25_FT_Back"], [0, 1.6, 1.8, 2.0, 100], ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
                bin_btts = get_bin(m_data["Odd_BTTS_Yes_Back"], [0, 1.6, 1.8, 2.0, 100], ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
                bin_lay = get_bin(m_data["Odd_CS_0x1_Lay"], [0, 10, 15, 20, 30, 100], ["<10", "10-15", "15-20", "20-30", "30+"])
                bin_avg_h = get_bin(home_avg, [0, 1.2, 1.5, 1.8, 5.0], ["<1.2", "1.2-1.5", "1.5-1.8", "1.8+"])

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
                portfolio_tag = ""
                if is_portfolio_match:
                    portfolio_tag = '<span class="tag tag-g" style="background:#00ff88; color:#000; font-weight:800;">🔥 GOLDEN PORTFOLIO BRANCH</span>'

                st.markdown(
                    f"""
<div style="background:#12151c; border:1px solid #1e2433; border-left:4px solid {v_color}; border-radius:6px; padding:14px 18px; margin-bottom:8px;">
<div style="display:flex; justify-content:space-between; align-items:center; gap:16px;">
<div style="flex:1;">
<div class="mc-label">{m_data.get("League", "")} · Lay 0×1</div>
<div style="font-size:1.2rem; font-weight:700; color:#e8edf4; margin:4px 0;">{m_data["Home"]} × {m_data["Away"]}</div>
<span class="tag {v_tag}">{results["recommendation"]}</span>{portfolio_tag}
<span class="tag tag-b">{results["score"]}/{results["max_score"]} pts</span>
<span class="tag tag-b">{results["sample_quality"]}</span>
<span class="tag tag-b">{str(m_data.get("Date", "")).split(" ")[0]}</span>
</div>
<div style="text-align:right; min-width:70px;">
<div class="verdict-score" style="color:{v_color};">{results["score"]}</div>
<div class="mc-label">de {results["max_score"]}</div>
</div>
</div>
<div style="margin-top:8px; padding-top:8px; border-top:1px solid #1e2433; font-size:0.75rem; color:#5a6577; line-height:1.5;">
{"  ·  ".join(results["reasons"])}
</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                # ════════════════════════════════════
                # 2. ODDS STRIP
                # ════════════════════════════════════
                st.markdown('<div class="sec-hdr">ODDS DE MERCADO</div>', unsafe_allow_html=True)
                odd_cs_lay = m_data.get("Odd_CS_0x1_Lay", 0)
                if odd_cs_lay > 0:
                    ev_prob = max(0.0, 1 - (results["pct_01_ft"] / 100))
                    ev_val = ev_prob - (1 - ev_prob) * (odd_cs_lay - 1)
                else:
                    ev_val = None

                o1, o2, o3, o4, o5, o6 = st.columns(6)
                with o1:
                    st.metric("Home", f"{m_data.get('Odd_H_Back', 0):.2f}")
                with o2:
                    st.metric("Draw", f"{m_data.get('Odd_D_Back', 0):.2f}")
                with o3:
                    st.metric("Away", f"{m_data.get('Odd_A_Back', 0):.2f}")
                with o4:
                    st.metric("Over 2.5", f"{m_data.get('Odd_Over25_FT_Back', 0):.2f}")
                with o5:
                    st.metric("CS 0×1 Lay", f"{odd_cs_lay:.2f}" if odd_cs_lay > 0 else "—")
                with o6:
                    st.metric("EV", f"{ev_val:+.3f}" if ev_val is not None else "—")

                # ════════════════════════════════════
                # 3. TACTICAL: CENÁRIOS + SAÍDA
                # ════════════════════════════════════
                st.markdown('<div class="sec-hdr">CENÁRIOS & PLANO DE SAÍDA</div>', unsafe_allow_html=True)
                tac_l, tac_r = st.columns(2)

                with tac_l:
                    signal_c = "#00ff88" if results["poisson_0x1"] <= 7 else "#ffd54f" if results["poisson_0x1"] <= 12 else "#ff4b4b"
                    st.markdown(
                        f"""
<div class="cpanel">
<div class="mc-label">Distribuição de Placares</div>
<div class="drow"><span class="drow-k">0×0 HT</span><span class="drow-v">{results["pct_00_ht"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">0×1 HT</span><span class="drow-v" style="color:#ff4b4b">{results["pct_01_ht"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">Outros HT</span><span class="drow-v">{results["pct_other_ht"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">0×0 FT</span><span class="drow-v">{results["pct_00_ft"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">0×1 FT</span><span class="drow-v" style="color:#ff4b4b">{results["pct_01_ft"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">Outros FT</span><span class="drow-v">{results["pct_other_ft"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">Poisson 0×1 HT</span><span class="drow-v" style="color:{signal_c}">{results["poisson_0x1"]:.2f}%</span></div>
<div class="drow"><span class="drow-k">Red de 0×0 HT</span><span class="drow-v">{results["red_from_00"]:.1f}%</span></div>
<div class="drow"><span class="drow-k">Red de 0×1 HT</span><span class="drow-v">{results["red_from_01"]:.1f}%</span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    cs_home = results["home"]["clean_sheet_pct"]
                    cs_away = results["away"]["clean_sheet_pct"]
                    st.markdown(
                        f"""
<div class="cpanel">
<div class="mc-label">Clean Sheet</div>
<div class="drow"><span class="drow-k">{m_data["Home"]}</span><span class="drow-v">{cs_home:.1f}%</span></div>
<div class="drow"><span class="drow-k">{m_data["Away"]}</span><span class="drow-v">{cs_away:.1f}%</span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                with tac_r:
                    st.markdown(
                        f"""
<div class="cpanel" style="border-left:3px solid #00ff88;">
<div class="mc-label">Saída Primária</div>
<div class="drow"><span class="drow-k">Hedge alvo</span><span class="drow-v" style="color:#00ff88">{risk_plan["adjusted_exit"]}'</span></div>
<div class="drow"><span class="drow-k">Base histórica</span><span class="drow-v">{risk_plan["base_exit"]}'</span></div>
<div class="drow"><span class="drow-k">Confiança</span><span class="drow-v">{risk_plan["confidence"]}</span></div>
</div>
<div class="cpanel" style="border-left:3px solid #ffd54f;">
<div class="mc-label">Zona de Decisão</div>
<div class="drow"><span class="drow-k">Faixa</span><span class="drow-v">{risk_plan["risk_band"]}</span></div>
<div style="font-size:0.72rem; color:#4f5b6e; margin-top:3px;">0×0: manter até alvo · 0×1 no HT: stop antecipado</div>
</div>
<div class="cpanel" style="border-left:3px solid #ff4b4b;">
<div class="mc-label">Stop Loss</div>
<div style="font-size:0.72rem; color:#4f5b6e;">{risk_plan["hedge_note"]}</div>
<div style="font-size:0.72rem; color:#4f5b6e; margin-top:2px;">{risk_plan["stop_note"]}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                # ════════════════════════════════════
                # 4. H2H + STRENGTH INDEX
                # ════════════════════════════════════
                st.markdown('<div class="sec-hdr">CONFRONTO & FORÇA</div>', unsafe_allow_html=True)
                h2h_c, str_c = st.columns(2)

                with h2h_c:
                    if h2h_stats:
                        cs_risk_icon = "🔴" if h2h_stats["score_0x1_pct"] > 10 else "🟡" if h2h_stats["score_0x1_pct"] > 5 else "🟢"
                        st.markdown(
                            f"""
<div class="cpanel">
    <div class="mc-label">H2H — {h2h_stats["total"]} jogos</div>
    <div class="drow"><span class="drow-k">0×1 FT</span><span class="drow-v">{h2h_stats["score_0x1"]} ({h2h_stats["score_0x1_pct"]:.1f}%) {cs_risk_icon}</span></div>
    <div class="drow"><span class="drow-k">1×0 FT</span><span class="drow-v">{h2h_stats["score_1x0"]} ({(h2h_stats["score_1x0"] / h2h_stats["total"]) * 100:.1f}%)</span></div>
    <div class="drow"><span class="drow-k">0×0 FT</span><span class="drow-v">{h2h_stats["score_0x0"]} ({(h2h_stats["score_0x0"] / h2h_stats["total"]) * 100:.1f}%)</span></div>
</div>
""",
                            unsafe_allow_html=True,
                        )
                        h2h_scores_df = pd.DataFrame(list(h2h_stats["top_scores"].items()), columns=["Placar", "Freq %"]).sort_values("Freq %", ascending=False)
                        fig_h2h = px.bar(h2h_scores_df, x="Placar", y="Freq %", text_auto=".1f", color="Freq %", color_continuous_scale="Viridis", template="plotly_dark")
                        fig_h2h.update_layout(height=200, showlegend=False, margin=dict(t=10, b=20, l=30, r=10))
                        st.plotly_chart(fig_h2h, use_container_width=True)
                    else:
                        st.info("Nenhum confronto direto encontrado.")

                with str_c:
                    h_sc = "#00ff88" if results["lay_strength_home"] >= 60 else "#ffd54f" if results["lay_strength_home"] >= 45 else "#ff4b4b"
                    a_sc = "#00ff88" if results["lay_strength_away"] >= 60 else "#ffd54f" if results["lay_strength_away"] >= 45 else "#ff4b4b"
                    st.markdown(
                        f"""
<div class="cpanel">
    <div class="mc-label">Índice Anti-0×1</div>
    <div class="drow"><span class="drow-k">{m_data["Home"]}</span><span class="drow-v" style="color:{h_sc}">{results["lay_strength_home"]:.1f} · {results["lay_strength_home_label"]}</span></div>
    <div class="drow"><span class="drow-k">Variância H</span><span class="drow-v">{results["lay_var_home"]:.1f} · {results["lay_var_home_label"]}</span></div>
    <div class="drow"><span class="drow-k">{m_data["Away"]}</span><span class="drow-v" style="color:{a_sc}">{results["lay_strength_away"]:.1f} · {results["lay_strength_away_label"]}</span></div>
    <div class="drow"><span class="drow-k">Variância A</span><span class="drow-v">{results["lay_var_away"]:.1f} · {results["lay_var_away_label"]}</span></div>
</div>
""",
                        unsafe_allow_html=True,
                    )
                    x1, x2 = st.columns(2)
                    with x1:
                        st.metric("xG Home", f"{results['home']['avg_xg']:.2f}")
                        st.metric("Home Avg Scored", f"{home_avg:.2f}")
                    with x2:
                        st.metric("xG Away", f"{results['away']['avg_xg']:.2f}")

                # ════════════════════════════════════
                # 5. GOAL INTERVALS
                # ════════════════════════════════════
                st.markdown('<div class="sec-hdr">INTERVALOS DE GOL — 15 min</div>', unsafe_allow_html=True)
                tab_int_h, tab_int_a = st.tabs([f"🏠 {m_data['Home']}", f"🚀 {m_data['Away']}"])
                with tab_int_h:
                    render_interval_chart(interval_stats["home_attack"], interval_stats["home_combined"], m_data["Home"], interval_stats["home_sample"])
                with tab_int_a:
                    render_interval_chart(interval_stats["away_attack"], interval_stats["away_combined"], m_data["Away"], interval_stats["away_sample"])

                # ═══════════════════════════════════════
                # DEEP DIVE — EXPANDERS
                # ═══════════════════════════════════════
                st.markdown('<div class="sec-hdr">ANÁLISE DETALHADA</div>', unsafe_allow_html=True)

                with st.expander("💰 Registrar Aposta", expanded=False):
                    st.markdown(
                        f"""
<div class="mc-label">Aposta</div>
<div style="font-size:1rem; font-weight:600; color:#d4dbe8;">{m_data["Home"]} × {m_data["Away"]}</div>
<span class="tag tag-b">Lay 0×1</span>
<span class="tag tag-g">{m_data.get("League", "")}</span>
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
                                "Valor (R$)",
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
                            st.success("Entrada registrada.")

                with st.expander("📋 Histórico & Perfis", expanded=False):
                    render_exec_summary(results["role_profile_home"], results["role_profile_away"], results["last10_home"], results["last10_away"])
                    c_l10h, c_l10a = st.columns(2)
                    with c_l10h:
                        render_last10_card(results["last10_home"], m_data["Home"], "#00ff88")
                    with c_l10a:
                        render_last10_card(results["last10_away"], m_data["Away"], "#4a9eff")

                    render_section_header("Perfil Mandante × Visitante", "Comparativo estrutural")
                    rp_h, rp_a = st.columns(2)
                    with rp_h:
                        render_role_profile(results["role_profile_home"], m_data["Home"], "Mandante", "#00ff88", results["home"]["avg_xg"])
                    with rp_a:
                        render_role_profile(results["role_profile_away"], m_data["Away"], "Visitante", "#4a9eff", results["away"]["avg_xg"])

                    render_section_header("Comparação Visual", "KPIs lado a lado")
                    kpi_cols = st.columns(3)
                    with kpi_cols[0]:
                        render_kpi_comparison("Gols marcados", results["role_profile_home"]["goals_for"], results["role_profile_away"]["goals_for"], m_data["Home"], m_data["Away"], fmt="{:.0f}")
                    with kpi_cols[1]:
                        render_kpi_comparison("Marcou 1º gol", results["role_profile_home"]["first_goal"]["scored_first_pct"], results["role_profile_away"]["first_goal"]["scored_first_pct"], m_data["Home"], m_data["Away"], fmt="{:.1f}%", is_percent=True)
                    with kpi_cols[2]:
                        render_kpi_comparison("xG médio", results["home"]["avg_xg"], results["away"]["avg_xg"], m_data["Home"], m_data["Away"], fmt="{:.2f}")

                with st.expander("🎯 Eficiência de Finalização", expanded=False):
                    col_sh1, col_sh2 = st.columns(2)
                    with col_sh1:
                        home_shots_value = results["home"]["shots_per_goal"]
                        render_metric_card(
                            f"{results['home']['shots_per_goal_label']} — {m_data['Home']}",
                            [f"Valor: {home_shots_value:.1f}" if pd.notna(home_shots_value) else "Valor: N/D", results["home"]["shots_per_goal_desc"]],
                        )
                    with col_sh2:
                        away_shots_value = results["away"]["shots_per_goal"]
                        render_metric_card(
                            f"{results['away']['shots_per_goal_label']} — {m_data['Away']}",
                            [f"Valor: {away_shots_value:.1f}" if pd.notna(away_shots_value) else "Valor: N/D", results["away"]["shots_per_goal_desc"]],
                        )

                with st.expander("⏱ Poisson 75'+", expanded=False):
                    scenario_00_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 0))
                    scenario_00_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 0))
                    scenario_01_home = build_poisson_timing_scenario(df_hist, m_data["Home"], "home", (0, 1))
                    scenario_01_away = build_poisson_timing_scenario(df_hist, m_data["Away"], "away", (0, 1))
                    render_poisson_time_block("Cenário 0×0 aos 75'", scenario_00_home, scenario_00_away, m_data["Home"], m_data["Away"])
                    render_poisson_time_block("Cenário 0×1 aos 75'", scenario_01_home, scenario_01_away, m_data["Home"], m_data["Away"])

                with st.expander("⏱ Timing & Placares", expanded=False):
                    if t_stats_h and t_stats_a:
                        tsum1, tsum2, tsum3 = st.columns(3)
                        with tsum1:
                            st.metric("Mandante", format_minutes(t_stats_h["avg_first_team"]) if t_stats_h else "N/D", help="Média do 1º gol do time")
                        with tsum2:
                            st.metric("Visitante", format_minutes(t_stats_a["avg_first_team"]) if t_stats_a else "N/D", help="Média do 1º gol do time")
                        with tsum3:
                            st.metric("Amostra", f"{t_stats_h['sample_size']} jogos")

                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.markdown(f"#### 🏠 {m_data['Home']}")
                            data_t_h = {
                                "Métrica": ["1º Gol (Ind.)", "1º Gol (Partida)", "1º Gol 2T (0×0 HT)", "1º Gol Part. 2T (0×0 HT)"],
                                "Média": [
                                    format_minutes(t_stats_h["avg_first_team"]),
                                    format_minutes(t_stats_h["avg_first_match"]),
                                    format_minutes(t_stats_h["avg_team_2h_00ht"]),
                                    format_minutes(t_stats_h["avg_match_2h_00ht"]),
                                ],
                                "N": [f"{t_stats_h['sample_size']}", f"{t_stats_h['sample_size']}", f"{t_stats_h['sample_00ht']}", f"{t_stats_h['sample_00ht']}"],
                            }
                            st.table(pd.DataFrame(data_t_h))
                        with col_t2:
                            st.markdown(f"#### 🚀 {m_data['Away']}")
                            data_t_a = {
                                "Métrica": ["1º Gol (Ind.)", "1º Gol (Partida)", "1º Gol 2T (0×0 HT)", "1º Gol Part. 2T (0×0 HT)"],
                                "Média": [
                                    format_minutes(t_stats_a["avg_first_team"]),
                                    format_minutes(t_stats_a["avg_first_match"]),
                                    format_minutes(t_stats_a["avg_team_2h_00ht"]),
                                    format_minutes(t_stats_a["avg_match_2h_00ht"]),
                                ],
                                "N": [f"{t_stats_a['sample_size']}", f"{t_stats_a['sample_size']}", f"{t_stats_a['sample_00ht']}", f"{t_stats_a['sample_00ht']}"],
                            }
                            st.table(pd.DataFrame(data_t_a))

                        fig_time = go.Figure()
                        fig_time.add_trace(go.Box(y=t_stats_h["raw_first_team"], name=m_data["Home"], marker_color="#00ff88", boxpoints="all"))
                        fig_time.add_trace(go.Box(y=t_stats_a["raw_first_team"], name=m_data["Away"], marker_color="#ff4b4b", boxpoints="all"))
                        fig_time.update_layout(title="Boxplot: Quando o 1º gol costuma sair?", yaxis_title="Minuto", template="plotly_dark", height=320, showlegend=False)
                        fig_time.add_hline(y=45, line_dash="dash", line_color="white", annotation_text="Fim 1T")
                        st.plotly_chart(fig_time, use_container_width=True)

                        st.markdown("#### 🔢 Placares Mais Frequentes")
                        tab_s1, tab_s2, tab_s3 = st.tabs([f"🏠 {m_data['Home']}", f"🚀 {m_data['Away']}", "🤝 Confronto"])
                        with tab_s1:
                            sc1, sc2 = st.columns(2)
                            with sc1:
                                display_scores(t_stats_h["frequent_scores"]["HT"])
                            with sc2:
                                display_scores(t_stats_h["frequent_scores"]["FT"])
                        with tab_s2:
                            sc1, sc2 = st.columns(2)
                            with sc1:
                                display_scores(t_stats_a["frequent_scores"]["HT"])
                            with sc2:
                                display_scores(t_stats_a["frequent_scores"]["FT"])
                        with tab_s3:
                            sc1, sc2 = st.columns(2)
                            with sc1:
                                display_scores(t_combined_scores["HT"])
                            with sc2:
                                display_scores(t_combined_scores["FT"])
                    else:
                        st.warning("Dados de minutagem insuficientes.")

                with st.expander("🎲 Variância & Matriz Poisson", expanded=False):
                    render_variance_section(results, m_data["Home"], m_data["Away"], normalize_team_name)

                with st.expander("🔍 Auditoria", expanded=False):
                    render_audit_section(df_hist, results, m_data["Home"], m_data["Away"], normalize_team_name)

# ═══════════════════════════════════════════════
# BETS TAB — PLANILHA
# ═══════════════════════════════════════════════
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

# ═══════════════════════════════════════════════
# NOTES TAB — NOTAS
# ═══════════════════════════════════════════════
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
