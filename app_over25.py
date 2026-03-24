import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Back Over 2.5 — Multi-Model", layout="wide")

DATA_ROOT = Path("data_total")
DATA_DAY = Path("data_day")

# ======================== FUNÇÕES BASE ========================


def poisson_pmf(k, lam):
    if lam < 0.001:
        lam = 0.001
    try:
        return math.exp(-lam) * (lam**k) / math.factorial(k)
    except:
        return 0


def norm_name(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()


# ======================== MODELOS ========================


def model_poisson(xg_h, xg_a, odd=None):
    """1. Poisson Puro: baseado em xG"""
    xg_total = max(0.001, float(xg_h) + float(xg_a))
    lam = xg_total
    return 1.0 - (poisson_pmf(0, lam) + poisson_pmf(1, lam) + poisson_pmf(2, lam))


def model_negative_binomial(xg_h, xg_a, odd=None, r=1.8):
    """2. Binomial Negativa: captura overdispersion melhor"""
    mean = max(0.001, float(xg_h) + float(xg_a))
    p = r / (r + mean)
    q = 1 - p
    prob_0 = p**r
    prob_1 = r * (q * (p**r))
    prob_2 = (r * (r + 1) / 2) * (q**2) * (p**r)
    return 1.0 - (prob_0 + prob_1 + prob_2)


def model_odds_implied(xg_h, xg_a, odd, margin=0.96):
    """3. Odds Implícitas: apenas probabilidade do mercado"""
    try:
        return (1.0 / float(odd)) * margin
    except:
        return 0.5


def model_hybrid(xg_h, xg_a, odd, weight_xg=0.60):
    """4. Híbrido: 60% xG + 40% Odds"""
    prob_xg = model_poisson(xg_h, xg_a)
    prob_odds = model_odds_implied(xg_h, xg_a, odd)
    return weight_xg * prob_xg + (1 - weight_xg) * prob_odds


def model_dixon_coles(xg_h, xg_a, odd=None, rho=0.05):
    """5. Dixon-Coles: modelo clássico para gols futebol com autocorrelação"""
    # Versão simplificada do modelo DC que ajusta correlação entre gols baixos
    mean_h = max(0.001, float(xg_h))
    mean_a = max(0.001, float(xg_a))

    # Poisson base
    prob_base = 1.0 - (poisson_pmf(0, mean_h + mean_a) + poisson_pmf(1, mean_h + mean_a) + poisson_pmf(2, mean_h + mean_a))

    # Ajuste DC para resultados baixos (reduz pequenos escores)
    adjustment = 1.0 - (rho * 0.15)  # pequeno fator de ajuste

    return prob_base * adjustment


def model_logistic_shots(xg_h, xg_a, odd, shots_h=None, shots_a=None):
    """6. Logístico com Shots: usa tentativas além xG"""
    # Se shots disponível, ajusta xG
    if shots_h and shots_a and shots_h > 0 and shots_a > 0:
        shots_total = float(shots_h) + float(shots_a)
        xg_adjusted = (float(xg_h) + float(xg_a)) * (1.0 + 0.1 * np.log1p(shots_total / 15))
    else:
        xg_adjusted = float(xg_h) + float(xg_a)

    # Aplica regressão logística simples
    z = -3.5 + (2.8 * xg_adjusted) + (0.5 * (1.0 / float(odd) if odd > 0 else 0))
    return 1.0 / (1.0 + np.exp(-z))


def model_zero_inflated_poisson(xg_h, xg_a, odd=None, inflation=0.08):
    """7. Zero-Inflated Poisson: para defesas fortes (muitos 0-0, 0-1)"""
    xg_total = max(0.001, float(xg_h) + float(xg_a))

    # ZIP: P(X=k) = inflation + (1-inflation)*Poisson(k)
    poisson_prob = 1.0 - (poisson_pmf(0, xg_total) + poisson_pmf(1, xg_total) + poisson_pmf(2, xg_total))

    # Reduz baseado em inflação de zeros
    return max(0.1, poisson_prob * (1.0 - inflation))


def model_possession_weighted(xg_h, xg_a, odd, poss_h=None, poss_a=None):
    """8. Ponderado por Possessão: times agressivos tendem marcar mais"""
    base_prob = model_poisson(xg_h, xg_a)

    if poss_h and poss_a and poss_h > 0 and poss_a > 0:
        poss_h = float(poss_h)
        poss_a = float(poss_a)
        # Possessão desequilibrada tende a mais gols
        poss_diff = abs(poss_h - poss_a)
        poss_adj = 1.0 + (poss_diff / 100.0) * 0.15  # até 15% boost
        return min(0.95, base_prob * poss_adj)

    return base_prob


def model_bayesian_prior(xg_h, xg_a, odd, prior_over25=0.37):
    """9. Bayesiano: Prior + Likelihood + Odds"""
    prior = prior_over25
    likelihood = model_poisson(xg_h, xg_a)
    try:
        prob_implied = (1.0 / float(odd)) * 0.96
    except:
        prob_implied = 0.5
    return 0.35 * prior + 0.45 * likelihood + 0.20 * prob_implied


def model_ensemble_average(xg_h, xg_a, odd, shots_h=None, shots_a=None, poss_h=None, poss_a=None, prior=0.37):
    """10. Ensemble: média de todos os modelos"""
    probs = [
        model_poisson(xg_h, xg_a),
        model_negative_binomial(xg_h, xg_a),
        model_odds_implied(xg_h, xg_a, odd),
        model_hybrid(xg_h, xg_a, odd),
        model_dixon_coles(xg_h, xg_a),
        model_logistic_shots(xg_h, xg_a, odd, shots_h, shots_a),
        model_zero_inflated_poisson(xg_h, xg_a),
        model_possession_weighted(xg_h, xg_a, odd, poss_h, poss_a),
        model_bayesian_prior(xg_h, xg_a, odd, prior),
    ]
    return np.mean(probs)


# ======================== LOAD DATA ========================


@st.cache_data
def load_data():
    try:
        betfair_df = pd.read_csv(DATA_ROOT / "dados_betfair.csv", low_memory=False, sep=";")
        footy_df = pd.read_csv(DATA_ROOT / "dados_footystats.csv", low_memory=False, sep=";")
        return betfair_df, footy_df
    except:
        return None, None


@st.cache_data
def get_available_days():
    return sorted([f.stem for f in DATA_DAY.glob("*.csv")])


# ======================== BACKTESTING ========================


@st.cache_data
def calculate_roi_all_models(betfair_json, footy_json):
    """Testa 10 modelos diferentes"""
    betfair_df = pd.read_json(betfair_json, orient="split")
    footy_df = pd.read_json(footy_json, orient="split")

    # Calcula prior histórico
    prior = (footy_df["TotalGoals_FT"] >= 3).sum() / len(footy_df) if len(footy_df) > 0 else 0.37

    models = {"Poisson": [], "Binomial Neg.": [], "Odds Impl.": [], "Híbrido": [], "Dixon-Coles": [], "Logístico": [], "ZIP": [], "Possessão": [], "Bayesiano": [], "Ensemble": []}

    for idx, row in betfair_df.iterrows():
        try:
            home_norm = norm_name(row["Home"])
            away_norm = norm_name(row["Away"])

            try:
                odd = float(row.get("Odd_Over25_FT_Back", 0))
                if odd <= 0 or odd > 10:
                    continue
            except:
                continue

            footy_match = footy_df[(footy_df["Home"].apply(norm_name) == home_norm) & (footy_df["Away"].apply(norm_name) == away_norm)]

            if footy_match.empty:
                continue

            try:
                xg_h = float(footy_match.iloc[0]["xG_H"])
                xg_a = float(footy_match.iloc[0]["xG_A"])
                total_goals = int(footy_match.iloc[0]["TotalGoals_FT"])
                shots_h = float(footy_match.iloc[0].get("Shots_H", 0))
                shots_a = float(footy_match.iloc[0].get("Shots_A", 0))
                poss_h = float(footy_match.iloc[0].get("Possession_H", 50))
                poss_a = float(footy_match.iloc[0].get("Possession_A", 50))
            except:
                continue

            if xg_h < 0 or xg_a < 0:
                continue

            hit = 1 if total_goals >= 3 else 0
            ret = (odd - 1) if hit == 1 else -1.0

            # Testa cada modelo
            probs = {
                "Poisson": model_poisson(xg_h, xg_a),
                "Binomial Neg.": model_negative_binomial(xg_h, xg_a),
                "Odds Impl.": model_odds_implied(xg_h, xg_a, odd),
                "Híbrido": model_hybrid(xg_h, xg_a, odd),
                "Dixon-Coles": model_dixon_coles(xg_h, xg_a),
                "Logístico": model_logistic_shots(xg_h, xg_a, odd, shots_h, shots_a),
                "ZIP": model_zero_inflated_poisson(xg_h, xg_a),
                "Possessão": model_possession_weighted(xg_h, xg_a, odd, poss_h, poss_a),
                "Bayesiano": model_bayesian_prior(xg_h, xg_a, odd, prior),
                "Ensemble": model_ensemble_average(xg_h, xg_a, odd, shots_h, shots_a, poss_h, poss_a, prior),
            }

            for model_name, prob in probs.items():
                ev = prob * (odd - 1.0) - (1.0 - prob)
                if ev >= 0.0:
                    models[model_name].append({"hit": hit, "return": ret})
        except:
            continue

    # Calcula ROI
    results = {}
    for name, trades in models.items():
        if trades:
            df_trades = pd.DataFrame(trades)
            roi = df_trades["return"].sum() / len(df_trades)
            win_rate = (df_trades["hit"].sum() / len(df_trades)) * 100
            results[name] = {"roi": roi, "win_rate": win_rate, "trades": len(df_trades)}
        else:
            results[name] = {"roi": -999, "win_rate": 0, "trades": 0}

    return results, prior


# ======================== UI ========================

st.title("⚡ Back Over 2.5 — 10 Modelos Competindo")
st.markdown("Poisson | Binomial Neg. | Odds | Híbrido | Dixon-Coles | Logístico | ZIP | Possessão | Bayesiano | Ensemble")

betfair_df, footy_df = load_data()

if betfair_df is None or footy_df is None:
    st.error("Erro ao carregar dados")
    st.stop()

st.write(f"✅ {len(betfair_df)} partidas Betfair + {len(footy_df)} com estatísticas")

# ======================== BACKTESTING ========================

st.subheader("📊 ROI dos 10 Modelos (Backtesting)")

with st.spinner("Testando 10 modelos..."):
    models_roi, prior = calculate_roi_all_models(betfair_df.to_json(orient="split"), footy_df.to_json(orient="split"))

# Ordena por ROI
sorted_models = sorted(models_roi.items(), key=lambda x: x[1]["roi"], reverse=True)

# Top 5
st.write("**Top 5 Modelos:**")
cols = st.columns(5)
for i, (name, stats) in enumerate(sorted_models[:5]):
    with cols[i]:
        st.metric(f"{i + 1}. {name}", f"{stats['roi'] * 100:+.2f}%", f"WR: {stats['win_rate']:.0f}% | {stats['trades']} trades")

best_model_name = sorted_models[0][0]
best_model_roi = sorted_models[0][1]["roi"]

st.markdown("---")

if best_model_roi > 0:
    st.success(f"🏆 **MELHOR: {best_model_name}** com ROI de {best_model_roi * 100:+.2f}%")
else:
    st.warning(f"⚠️ Melhor ROI: {best_model_roi * 100:+.2f}% ({best_model_name})")

st.info(f"📊 Prior Bayesiano: {prior * 100:.1f}%")

# Todos os modelos
with st.expander("📈 Ver todos os 10 modelos"):
    for name, stats in sorted_models:
        st.write(f"**{name}**: {stats['roi'] * 100:+.2f}% ROI | WR: {stats['win_rate']:.0f}% | {stats['trades']} trades")

# ======================== APLICAR AO DIA ========================

st.markdown("---")
st.subheader("💰 Apostas EV+ — Usando Melhor Modelo")

available_days = get_available_days()

if not available_days:
    st.error("Nenhum dia em data_day")
    st.stop()

selected_day = st.selectbox("📅 Escolha um dia:", available_days)

try:
    day_df = pd.read_csv(DATA_DAY / f"{selected_day}.csv", low_memory=False, sep=";")
except:
    day_df = None

if day_df is None or day_df.empty:
    st.warning(f"Nenhuma partida em {selected_day}")
    st.stop()

st.write(f"📋 {len(day_df)} partidas para {selected_day}")

# ======================== PROCESSA ========================

day_results = []

for idx, match in day_df.iterrows():
    try:
        home_norm = norm_name(match["Home"])
        away_norm = norm_name(match["Away"])

        try:
            odd = float(match.get("Odd_Over25_FT_Back", 0))
            if odd <= 0 or odd > 10:
                continue
        except:
            continue

        footy_match = footy_df[(footy_df["Home"].apply(norm_name) == home_norm) & (footy_df["Away"].apply(norm_name) == away_norm)]

        if footy_match.empty:
            continue

        try:
            xg_h = float(footy_match.iloc[0]["xG_H"])
            xg_a = float(footy_match.iloc[0]["xG_A"])
            shots_h = float(footy_match.iloc[0].get("Shots_H", 0))
            shots_a = float(footy_match.iloc[0].get("Shots_A", 0))
            poss_h = float(footy_match.iloc[0].get("Possession_H", 50))
            poss_a = float(footy_match.iloc[0].get("Possession_A", 50))
        except:
            continue

        if xg_h < 0 or xg_a < 0:
            continue

        # Seleciona melhor modelo
        if best_model_name == "Poisson":
            prob = model_poisson(xg_h, xg_a)
        elif best_model_name == "Binomial Neg.":
            prob = model_negative_binomial(xg_h, xg_a)
        elif best_model_name == "Odds Impl.":
            prob = model_odds_implied(xg_h, xg_a, odd)
        elif best_model_name == "Híbrido":
            prob = model_hybrid(xg_h, xg_a, odd)
        elif best_model_name == "Dixon-Coles":
            prob = model_dixon_coles(xg_h, xg_a)
        elif best_model_name == "Logístico":
            prob = model_logistic_shots(xg_h, xg_a, odd, shots_h, shots_a)
        elif best_model_name == "ZIP":
            prob = model_zero_inflated_poisson(xg_h, xg_a)
        elif best_model_name == "Possessão":
            prob = model_possession_weighted(xg_h, xg_a, odd, poss_h, poss_a)
        elif best_model_name == "Bayesiano":
            prob = model_bayesian_prior(xg_h, xg_a, odd, prior)
        else:  # Ensemble
            prob = model_ensemble_average(xg_h, xg_a, odd, shots_h, shots_a, poss_h, poss_a, prior)

        ev = prob * (odd - 1.0) - (1.0 - prob)

        if ev >= 0.0:
            day_results.append(
                {
                    "Casa": match["Home"],
                    "Visitante": match["Away"],
                    "xG": f"{xg_h:.2f}+{xg_a:.2f}",
                    "Shots": f"{int(shots_h)}+{int(shots_a)}",
                    "Poss%": f"{poss_h:.0f}% vs {poss_a:.0f}%",
                    "Odd": f"{odd:.2f}",
                    "P(O2.5)": f"{prob * 100:.1f}%",
                    "EV": f"{ev:.4f}",
                    "EV%": f"{ev * 100:.2f}%",
                },
            )
    except:
        continue

if day_results:
    result_df = pd.DataFrame(day_results)
    st.dataframe(result_df, use_container_width=True)
    st.success(f"✅ **{len(result_df)}** apostas EV+ com {best_model_name}!")
else:
    st.warning(f"❌ Nenhuma aposta EV+ para {selected_day}")

st.markdown("---")
st.caption(f"🏆 Melhor: **{best_model_name}** | ROI: {best_model_roi * 100:+.2f}%")
