import pandas as pd

# Estrategia Lay 0x1 - Get Up Trading

# Mostrar todas as colunas
pd.set_option("display.max_columns", None)

data = pd.read_csv("../../data_total/dados_betfair.csv", sep=";")

# Filtrar colunas para análise (incluindo Goals_Min_A para calcular o placar aos 75')
datatest = data[["Date", "League", "Home", "Away", "Goals_H_HT", "Goals_A_HT", "Goals_H_FT", "Goals_A_FT", "Goals_Min_H", "Goals_Min_A", "Odd_H_Back", "Odd_A_Back", "Odd_Over25_FT_Back", "Odd_BTTS_Yes_Back", "Odd_CS_0x1_Lay"]].copy()

# Apagar linhas da coluna Odd_CS_0x1_Lay > 50
datatest = datatest[datatest["Odd_CS_0x1_Lay"] <= 50]

# --- Lógica de Saída aos 75 Minutos ---


def get_score_at_minute(mins_str, limit=75):
    if pd.isna(mins_str) or str(mins_str).strip() == "[]" or str(mins_str).strip() == "":
        return 0
    try:
        # Limpeza para garantir que a string seja interpretada corretamente
        # Remove espaços e aspas extras que podem vir no CSV
        content = str(mins_str).replace(" ", "").replace("'", "").replace('"', "")

        # Se for apenas colchetes vazios após a limpeza
        if content == "[]":
            return 0

        # Converter para lista
        if content.startswith("[") and content.endswith("]"):
            # Transforma "[10,20]" em ["10", "20"]
            mins_list = content[1:-1].split(",")
        else:
            mins_list = [content]

        goals = 0
        for m in mins_list:
            if m.strip() == "":
                continue
            # Trata o formato "45+2" pegando apenas o tempo regulamentar
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
    # Responsabilidade (Liability) = STAKE * (Odd_Lay - 1)
    responsabilidade = STAKE * (row["Odd_CS_0x1_Lay"] - 1)

    # Se jogo 0 x 0 aos 75': 10% de red da Responsabilidade
    if row["H_75"] == 0 and row["A_75"] == 0:
        return round(-responsabilidade * 0.10, 2)
    # Se jogo 0 x 1 aos 75': 50% de red da Responsabilidade
    elif row["H_75"] == 0 and row["A_75"] == 1:
        return round(-responsabilidade * 0.50, 2)
    # Outro placar: Green
    else:
        return round(STAKE * (1 - COMISSAO), 2)


datatest["Profit"] = datatest.apply(calculate_profit_75, axis=1)


datatest.to_csv("dados_betfair_estrategia.csv", index=False, sep=";")
