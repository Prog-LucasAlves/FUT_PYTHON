from pathlib import Path

import lay0x1_core
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import poisson


def render_section_header(kicker, heading):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-heading">{heading}</div>', unsafe_allow_html=True)


def render_badge(text, variant="info"):
    st.markdown(f'<span class="badge {variant}">{text}</span>', unsafe_allow_html=True)


def render_metric_card(title, value_lines, accent_color=None):
    style = f' style="border-left: 5px solid {accent_color};"' if accent_color else ""
    st.markdown(f'<div class="metric-card"{style}>', unsafe_allow_html=True)
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    for line in value_lines:
        st.write(line)
    st.markdown("</div>", unsafe_allow_html=True)


def render_stat_grid(items, columns=4):
    cols = st.columns(columns)
    for idx, item in enumerate(items):
        col = cols[idx % columns]
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if len(item) == 3:
                title, value, caption = item
                accent = None
            else:
                title, value, caption, accent = item
            if accent:
                st.markdown(f'<div class="panel-title" style="color:{accent};">{title}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
            st.metric(title, value)
            if caption:
                st.caption(caption)
            st.markdown("</div>", unsafe_allow_html=True)


def render_note_card(row):
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
<div style="background: linear-gradient(145deg, #1e2130, #161924); border: 1px solid {card_border}; border-left: 8px solid {card_border}; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 8px 25px rgba(0,0,0,0.25);">
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


def render_new_note_form(note_priority_options, note_status_options):
    st.subheader("✍️ Nova Nota")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_title = st.text_input("Título", placeholder="Ex.: Jogo com risco alto no 1T")
    with c2:
        new_tag = st.text_input("Tag", placeholder="Ex.: risco, live, estudo")
    with c3:
        new_priority = st.selectbox("Prioridade", note_priority_options, index=1)

    n1, n2 = st.columns([2, 1])
    with n1:
        new_note = st.text_area("Conteúdo", placeholder="Escreva sua leitura, gatilhos, dúvidas e planos...", height=160)
    with n2:
        new_status = st.selectbox("Status", note_status_options, index=0)
        new_pinned = st.checkbox("Fixar no topo", value=False)
        new_image = st.file_uploader(
            "Anexar printscreen",
            type=["png", "jpg", "jpeg", "webp"],
            key="new_note_image",
        )

    return {
        "title": new_title,
        "tag": new_tag,
        "priority": new_priority,
        "note": new_note,
        "status": new_status,
        "pinned": new_pinned,
        "image": new_image,
    }


def render_edit_note_form(selected_note, note_priority_options, note_status_options, selected_note_id):
    st.subheader("🛠️ Editar Notas")
    e1, e2, e3 = st.columns(3)
    with e1:
        edit_title = st.text_input("Título da nota", value=str(selected_note["title"]))
    with e2:
        edit_tag = st.text_input("Tag da nota", value=str(selected_note["tag"]))
    with e3:
        edit_priority = st.selectbox(
            "Prioridade da nota",
            note_priority_options,
            index=note_priority_options.index(str(selected_note["priority"])) if str(selected_note["priority"]) in note_priority_options else 1,
        )

    s1, s2 = st.columns(2)
    with s1:
        edit_status = st.selectbox(
            "Status da nota",
            note_status_options,
            index=note_status_options.index(str(selected_note["status"])) if str(selected_note["status"]) in note_status_options else 0,
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

    return {
        "title": edit_title,
        "tag": edit_tag,
        "priority": edit_priority,
        "status": edit_status,
        "pinned": edit_pinned,
        "note": edit_note,
        "image": edit_image,
    }


def render_variance_section(results, home_name, away_name, normalize_team_name_fn):
    st.markdown("#### 🎲 Matriz de Probabilidades (Poisson)")
    max_goals = 5
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = (poisson.pmf(i, results["home"]["mean"]) * poisson.pmf(j, results["away"]["mean"])) * 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(home_name, f"{results['home']['mean']:.2f}", help="Média esperada de gols")
        st.caption(f"Variância: {results['home']['variance']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(away_name, f"{results['away']['mean']:.2f}", help="Média esperada de gols")
        st.caption(f"Variância: {results['away']['variance']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Matriz", "Poisson", help="Probabilidades combinadas de placar")
        st.caption("Grade de 0x0 a 4x4")
        st.markdown("</div>", unsafe_allow_html=True)

    v1, v2 = st.columns(2)
    with v1:
        st.write(f"### 🏠 {home_name}")
        st.write(f"**Variância de Gols:** {results['home']['variance']:.3f}")
        st.write(f"**Custo do Gol (Eficiência):** {results['home']['cost']:.3f}")
        norm_h = normalize_team_name_fn(home_name)
        h_goals = results["h_games"].apply(
            lambda r: r["Goals_H_FT"] if normalize_team_name_fn(r["Home"]) == norm_h else r["Goals_A_FT"],
            axis=1,
        )
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(y=h_goals, mode="lines+markers", name="Gols", line=dict(color="#00ff88")))
        fig_h.update_layout(title="Histórico de Gols (Casa)", template="plotly_dark", height=250)
        st.plotly_chart(fig_h, use_container_width=True)
    with v2:
        st.write(f"### 🚀 {away_name}")
        st.write(f"**Variância de Gols:** {results['away']['variance']:.3f}")
        st.write(f"**Custo do Gol (Eficiência):** {results['away']['cost']:.3f}")
        norm_a = normalize_team_name_fn(away_name)
        a_goals = results["a_games"].apply(
            lambda r: r["Goals_A_FT"] if normalize_team_name_fn(r["Away"]) == norm_a else r["Goals_H_FT"],
            axis=1,
        )
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(y=a_goals, mode="lines+markers", name="Gols", line=dict(color="#ff4b4b")))
        fig_a.update_layout(title="Histórico de Gols (Fora)", template="plotly_dark", height=250)
        st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("#### 🎲 Matriz de Probabilidades (Poisson)")
    fig_matrix = px.imshow(
        matrix,
        labels=dict(x="Gols Visitante", y="Gols Mandante", color="%"),
        x=[str(i) for i in range(max_goals)],
        y=[str(i) for i in range(max_goals)],
        color_continuous_scale="Viridis",
        text_auto=".1f",
    )
    st.plotly_chart(fig_matrix, use_container_width=True)


def render_audit_section(df_hist, results, home_name, away_name, normalize_team_name_fn):
    audit_rows = []
    audit_specs = [
        ("Mandante", results["role_profile_home"], home_name, "home"),
        ("Visitante", results["role_profile_away"], away_name, "away"),
    ]
    for label, profile, team_name, role in audit_specs:
        expected = lay0x1_core.build_team_role_profile(df_hist, team_name, role, normalize_team_name_fn)
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
    divergent_count = total_count - ok_count
    coverage = (ok_count / total_count * 100) if total_count else 0.0
    st.markdown('<div class="section-kicker">Resumo da auditoria</div>', unsafe_allow_html=True)
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


def render_exec_summary(home_profile, away_profile, home_last10, away_last10):
    st.markdown('<div class="section-kicker">Resumo executivo</div><div class="section-heading">Leitura rápida do confronto</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "⚡ PPG", f"{home_profile['ppg_season']:.2f} / {away_profile['ppg_season']:.2f}", "info"),
        (c2, "📅 Jogos", f"{int(home_profile['sample_size'])} / {int(away_profile['sample_size'])}", "ok"),
        (c3, "🥅 Gols", f"{int(home_profile['goals_for'])} / {int(away_profile['goals_for'])}", "warn"),
        (c4, "🎯 1º Gol", f"{home_profile['first_goal']['scored_first_pct']:.1f}% / {away_profile['first_goal']['scored_first_pct']:.1f}%", "ok"),
    ]
    for col, title, value, badge in cards:
        with col:
            st.metric(title, value)
            st.markdown(f'<span class="badge {badge}">executivo</span>', unsafe_allow_html=True)
    st.caption(f"Base histórica: `dados_historicos.csv` | Últimos 10: {home_last10['record']} vs {away_last10['record']}")
    st.caption("Leitura de 1º gol depende dos minutos registrados em `Min_Goals_H` e `Min_Goals_A`.")

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
<div style="margin-top:0.35rem;">Leitura consolidada do confronto.</div>
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


def render_role_profile(profile, team_name, role_label, accent_color, xg_value):
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
    top_row = st.columns(2)
    with top_row[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Jogos", f"{int(profile['sample_size'])}")
        st.caption(f"{role_kicker} | Jogos analisados na base histórica")
        st.markdown("</div>", unsafe_allow_html=True)
    with top_row[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Vitórias", f"{int(profile['wins'])}")
        st.caption(f"{role_kicker} | Vitórias no recorte analisado")
        st.markdown("</div>", unsafe_allow_html=True)

    mid_row = st.columns(3)
    quick_cards = [
        (mid_row[0], "Gols pró", profile["goals_for"], "ok"),
        (mid_row[1], "Gols contra", profile["goals_against"], "warn"),
        (mid_row[2], "Jogos", profile["sample_size"], "info"),
    ]
    for col, label, value, badge in quick_cards:
        with col:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            display_value = f"{int(round(value))}" if label.startswith("Gols") and isinstance(value, (int, float)) else f"{int(value)}" if isinstance(value, (int, float)) else value
            st.metric(label, display_value)
            st.caption(role_kicker)
            st.markdown("</div>", unsafe_allow_html=True)

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


def render_kpi_comparison(title, home_value, away_value, home_label, away_label, fmt="{:.2f}", is_percent=False):
    st.markdown(f'<div class="section-kicker">{title}</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    max_value = 100.0 if is_percent else max(float(home_value) if isinstance(home_value, (int, float)) else 0, float(away_value) if isinstance(away_value, (int, float)) else 0, 1.0)
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


def render_poisson_time_block(title, home_scenario, away_scenario, home_name, away_name):
    st.markdown(f"#### {title}")
    b1, b2 = st.columns(2)

    with b1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.write(f"**{home_name}**")
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
        st.write(f"**{away_name}**")
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
        fig.add_trace(go.Scatter(x=home_scenario["timeline"]["Minute"], y=home_scenario["timeline"]["Match"], mode="lines+markers", name=f"{home_name} - Partida", line=dict(color="#00ff88")))
    if away_scenario:
        fig.add_trace(go.Scatter(x=away_scenario["timeline"]["Minute"], y=away_scenario["timeline"]["Match"], mode="lines+markers", name=f"{away_name} - Partida", line=dict(color="#ff4b4b")))

    if fig.data:
        fig.update_layout(title=f"Probabilidade acumulada de gol após 75' - {title}", xaxis_title="Minuto", yaxis_title="Probabilidade (%)", template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem amostra suficiente para calcular a distribuição deste cenário.")


def display_scores(scores_dict):
    df_scores = pd.DataFrame(list(scores_dict.items()), columns=["Placar", "Freq %"]).sort_values("Freq %", ascending=False)
    fig = px.bar(df_scores, x="Placar", y="Freq %", text_auto=".1f", color="Freq %", color_continuous_scale="Viridis", template="plotly_dark")
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


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
