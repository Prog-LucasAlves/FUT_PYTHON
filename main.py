import datetime
import io
import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuração da API
TOKEN = os.getenv("API_KEY")
HEADERS = {"Authorization": f"Token {TOKEN}"}
FONTE1 = "betfair"
FONTE2 = "footystats"

# Cria a pasta 'data_total' se não existir
if not os.path.exists("data_total"):
    os.makedirs("data_total")
    print("Pasta 'data_total' criada com sucesso!")

# Cria a pasta 'data_day' se não existir
if not os.path.exists("data_day"):
    os.makedirs("data_day")
    print("Pasta 'data_day' criada com sucesso!")


def getDataTotalfootystats():
    """Baixa os dados da API e retorna um DataFrame."""
    print(f"Baixando dados da fonte '{FONTE2}' ...")
    URL = f"https://api.futpythontrader.com/api/dados/{FONTE2}/download/"
    response = requests.get(
        URL,
        headers=HEADERS,
    )

    if response.status_code == 200:
        # Lê os bytes do CSV diretamente para um DataFrame
        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Sucesso! DataFrame criado com {len(df)} linhas.")
        df.to_csv(f"data_total/dados_{FONTE2}.csv", index=False, sep=";")

        # Criar um csv com os nome das colunas
        df.columns.to_series().index.to_series().to_csv(
            f"data_total/columns_{FONTE2}.csv",
            header=False,
            index=False,
        )

        return df
    else:
        print(f"Erro na requisição: {response.status_code}")
        print(response.text)
        return pd.DataFrame()


def getDataTotalBetfair():
    """Baixa os dados da API e retorna um DataFrame."""
    print(f"Baixando dados da fonte '{FONTE1}' ...")
    URL = f"https://api.futpythontrader.com/api/dados/{FONTE1}/download/"
    response = requests.get(
        URL,
        headers=HEADERS,
    )

    if response.status_code == 200:
        # Lê os bytes do CSV diretamente para um DataFrame
        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Sucesso! DataFrame criado com {len(df)} linhas.")
        df.to_csv(f"data_total/dados_{FONTE1}.csv", index=False, sep=";")

        # Criar um csv com os nome das colunas
        df.columns.to_series().index.to_series().to_csv(
            f"data_total/columns_{FONTE1}.csv",
            header=False,
            index=False,
        )

        return df
    else:
        print(f"Erro na requisição: {response.status_code}")
        print(response.text)
        return pd.DataFrame()  # Retorna DF vazio em caso de erro


def getDataDay():
    """Baixa os dados diários da API e retorna um DataFrame."""
    print(f"Baixando dados diários da fonte '{FONTE1}' ...")

    # Pegar coma data de hoje, amanha e dpois
    date_hoje = datetime.date.today()
    date_amanha = date_hoje + datetime.timedelta(days=1)

    date_list = [date_hoje, date_amanha]

    for i in date_list:
        URL = f"https://api.futpythontrader.com/api/dados/jogos-do-dia/betfair/{i}/download/"
        response = requests.get(
            URL,
            headers=HEADERS,
        )

        if response.status_code == 200:
            df = pd.read_csv(io.BytesIO(response.content))
            df.to_csv(f"data_day/dados_day_{FONTE1}_{i}.csv", index=False, sep=";")
        else:
            print(f"Erro na requisição: {response.status_code}")
            print(response.text)
            return pd.DataFrame()  # Retorna DF vazio em caso de erro


if __name__ == "__main__":
    getDataTotalfootystats()
    getDataTotalBetfair()
    getDataDay()
