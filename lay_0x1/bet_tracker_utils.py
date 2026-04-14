import os

import numpy as np
import pandas as pd
from app_config import BETS_TRACKER_FILE


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
    if saida == "Red":
        return -(odd_entrada - 1) * valor_aposta
    if saida == "75min" and odd_saida_75min and odd_saida_75min > 0:
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
