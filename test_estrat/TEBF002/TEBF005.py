import pandas as pd

# Estrategia Lay 0x1 - Portfolio de 10 Faixas Lucrativas

pd.set_option("display.max_columns", None)

data = pd.read_csv("data_total/dados_betfair.csv", sep=";")

datatest = data[["Date", "League", "Home", "Away", "Goals_H_FT", "Goals_A_FT", "Goals_Min_H", "Goals_Min_A", "Odd_H_Back", "Odd_A_Back", "Odd_Over25_FT_Back", "Odd_BTTS_Yes_Back", "Odd_CS_0x1_Lay"]].copy()

# Trava de Segurança
datatest = datatest[datatest["Odd_CS_0x1_Lay"] <= 60]


def get_score_at_minute(mins_str, limit=75):
    if pd.isna(mins_str) or str(mins_str).strip() == "[]" or str(mins_str).strip() == "":
        return 0
    try:
        content = str(mins_str).replace(" ", "").replace("'", "").replace('"', "")
        if content == "[]":
            return 0
        mins_list = content[1:-1].split(",") if (content.startswith("[") and content.endswith("]")) else [content]
        goals = 0
        for m in mins_list:
            if m.strip() == "":
                continue
            base_min = int(m.split("+")[0])
            if base_min <= limit:
                goals += 1
        return goals
    except:
        return 0


datatest["H_75"] = datatest["Goals_Min_H"].apply(get_score_at_minute)
datatest["A_75"] = datatest["Goals_Min_A"].apply(get_score_at_minute)

STAKE = 1
COMISSAO = 0.065


def calculate_profit_75(row):
    liability = STAKE * (row["Odd_CS_0x1_Lay"] - 1)
    if row["H_75"] == 0 and row["A_75"] == 0:
        return round(-liability * 0.10, 2)
    elif row["H_75"] == 0 and row["A_75"] == 1:
        return round(-liability * 0.50, 2)
    else:
        return round(STAKE * (1 - COMISSAO), 2)


datatest["Profit"] = datatest.apply(calculate_profit_75, axis=1)

# Calcular média de gols marcados em casa por cada time (usando apenas dados_betfair.csv)
avg_goals_home = datatest.groupby("Home")["Goals_H_FT"].mean().reset_index()
avg_goals_home.columns = ["Home", "Avg_Goals_H"]
datatest = datatest.merge(avg_goals_home, on="Home", how="left")

# --- Classificação de Faixas ---
datatest["Bin_H"] = pd.cut(datatest["Odd_H_Back"], bins=[1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 100], labels=["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.1-2.5", "2.6-3.0", "3.0+"])
datatest["Bin_Over"] = pd.cut(datatest["Odd_Over25_FT_Back"], bins=[0, 1.6, 1.8, 2.0, 100], labels=["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
datatest["Bin_BTTS"] = pd.cut(datatest["Odd_BTTS_Yes_Back"], bins=[0, 1.6, 1.8, 2.0, 100], labels=["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"])
datatest["Bin_Lay"] = pd.cut(datatest["Odd_CS_0x1_Lay"], bins=[0, 10, 15, 20, 30, 100], labels=["<10", "10-15", "15-20", "20-30", "30+"])
datatest["Bin_Avg_H"] = pd.cut(datatest["Avg_Goals_H"], bins=[0, 1.2, 1.5, 1.8, 5.0], labels=["<1.2", "1.2-1.5", "1.5-1.8", "1.8+"])

# --- Busca das Melhores Faixas (Portfolio Optimizer) ---
# Agrupar por todas as faixas e calcular lucro e volume
brackets = datatest.groupby(["Bin_H", "Bin_Over", "Bin_BTTS", "Bin_Lay", "Bin_Avg_H"], observed=True).agg(Lucro=("Profit", "sum"), Qtd=("Profit", "count"), WinRate=("Profit", lambda x: (x > 0).mean())).reset_index()

# Filtrar brackets com volume mínimo e lucro positivo
min_volume = 15
top_brackets = brackets[(brackets["Qtd"] >= min_volume) & (brackets["Lucro"] > 0)].sort_values(by="Lucro", ascending=False).head(10)

# Lista de chaves do Portfolio
winning_brackets = set(zip(top_brackets["Bin_H"].astype(str), top_brackets["Bin_Over"].astype(str), top_brackets["Bin_BTTS"].astype(str), top_brackets["Bin_Lay"].astype(str), top_brackets["Bin_Avg_H"].astype(str)))


# Marcar jogos que pertencem ao Portfolio Vencedor
def is_winner(row):
    key = (str(row["Bin_H"]), str(row["Bin_Over"]), str(row["Bin_BTTS"]), str(row["Bin_Lay"]), str(row["Bin_Avg_H"]))
    return 1 if key in winning_brackets else 0


datatest["Bet"] = datatest.apply(is_winner, axis=1)

# Resultados Finais
print("OTIMIZACAO DE PORTFOLIO COM MEDIA DE GOLS")
print("-" * 50)

datatest_bets = datatest[datatest["Bet"] == 1]
if not datatest_bets.empty:
    print(f"Total de Apostas: {len(datatest_bets)}")
    print(f"Lucro Total (PL): {datatest_bets['Profit'].sum():.2f}")
    print(f"Taxa de Acerto:   {(datatest_bets['Profit'] > 0).mean() * 100:.2f}%")
    print(f"Lucro Medio/Aposta: {datatest_bets['Profit'].mean():.4f}")
    print("-" * 50)

    print("Top 10 Brackets Encontrados:")
    print(top_brackets.to_string(index=False))
else:
    print("Nenhuma aposta encontrada com os criterios.")
