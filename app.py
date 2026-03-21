import os

import pandas as pd
import streamlit as st

from main import (
    createAwayHTFT,
    createAwayHTGoals,
    createAwayHTGoals00,
    createAwayHTGoals00Result,
    createHomeHTFT,
    createHomeHTGoals,
    createHomeHTGoals00,
    createHomeHTGoals00Result,
)

########################
# Configuração da Página
########################
st.set_page_config(page_title="FUTSTATS", page_icon="⚽", layout="wide")


########################
# Carregar os dados
########################
@st.cache_data(ttl=300)
def load_data():
    data = pd.read_csv("data_total/dados_betfair.csv", sep=";")
    return data


@st.cache_data(ttl=300)
def load_data_total():
    data = pd.read_csv("data_total/dados_betfair.csv", sep=";")
    return data


@st.cache_data(ttl=300)
def load_data_today():
    # Ler e unir varios arquivos csv da pasta gameday
    data = pd.concat(
        [pd.read_csv(f"data_day/{file}", sep=";") for file in os.listdir("data_day")],
    )
    return data


@st.cache_data(ttl=300)
def load_createHomeHTFT():
    data = createHomeHTFT()
    return data


@st.cache_data(ttl=300)
def load_createHomeHTGoals():
    data = createHomeHTGoals()
    return data


@st.cache_data(ttl=300)
def load_createHomeHTGoals00():
    data = createHomeHTGoals00()
    return data


@st.cache_data(ttl=300)
def load_createHomeHTGoals00Result():
    data = createHomeHTGoals00Result()
    return data


@st.cache_data(ttl=300)
def load_createAwayHTFT():
    data = createAwayHTFT()
    return data


@st.cache_data(ttl=300)
def load_createAwayHTGoals():
    data = createAwayHTGoals()
    return data


@st.cache_data(ttl=300)
def load_createAwayHTGoals00():
    data = createAwayHTGoals00()
    return data


@st.cache_data(ttl=300)
def load_createAwayHTGoals00Result():
    data = createAwayHTGoals00Result()
    return data


########################
# Título da página
########################
st.title("⚽ FutStats HT - FT")

ultima_data = load_data_total().iloc[-1]["Date"]
st.markdown(f"**📅 Última atualização: {ultima_data}**")

########################
# CSS
########################
st.markdown(
    """
<style>
    div[data-baseweb="select"] > div {
        width: 200px !important;  /* Ajuste a largura conforme necessário */
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    div[data-testid="stDateInput"] > div {
        width: 200px !important;  /* Ajuste a largura conforme necessário */
    }
</style>
""",
    unsafe_allow_html=True,
)

########################
# Abas da página
########################
tab1, tab2, tab3, tab4 = st.tabs(
    ["Dados Gerais", "Mandante | HT - FT", "Visitante | HT - FT", "Jogos do Dia"],
)

with tab1:
    minDateDG = load_data_total()["Date"].min()
    maxDateDG = load_data_total()["Date"].max()

    selectDataDG = st.date_input(
        "Selecione a data:",
        value=minDateDG,
        min_value=minDateDG,
        max_value=maxDateDG,
        format="DD/MM/YYYY",
    )

    dateDG = selectDataDG.strftime("%Y-%m-%d")

    st.dataframe(
        load_data_total()[load_data_total()["Date"] == dateDG],
        hide_index=True,
        width="stretch",
    )

with tab2:
    st.subheader("⚽ Informações do HT")

    home = st.selectbox(
        "Selecione o time mandante:",
        load_data_total()["Home"].sort_values().unique(),
    )
    # Exibir os dados baseado no home - time selecionado
    datamandantehtft = load_createHomeHTFT()[load_createHomeHTFT()["Home"] == home]

    # Exibr os dados baseado no home - - time selecionado
    datamandantegoalsht = load_createHomeHTGoals()[
        load_createHomeHTGoals()["Home"] == home
    ]

    datamanadanteht00 = load_createHomeHTGoals00()[
        load_createHomeHTGoals00()["Home"] == home
    ]

    datamandantehtftresult = load_createHomeHTGoals00Result()[
        load_createHomeHTGoals00Result()["Home"] == home
    ]

    # Organizar as colunas
    datamandantehtft = datamandantehtft[
        [
            "Home",
            "Games",
            "HT_Losse_Games",
            "D/D",
            "P_D_D",
            "Odd_back_D/D",
            "Odd_lay_D/D",
            "D/E",
            "P_D_E",
            "Odd_back_D/E",
            "Odd_lay_D/E",
            "D/V",
            "P_D_V",
            "Odd_back_D/V",
            "Odd_lay_D/V",
            "HT_Draw_Games",
            "E/D",
            "P_E_D",
            "Odd_back_E/D",
            "Odd_lay_E/D",
            "E/E",
            "P_E_E",
            "Odd_back_E/E",
            "Odd_lay_E/E",
            "E/V",
            "P_E_V",
            "Odd_back_E/V",
            "Odd_lay_E/V",
            "Odd_Lay_Draw",
            "Prob_V/D",
            "HT_Win_Games",
            "V/D",
            "P_V_D",
            "Odd_back_V/D",
            "Odd_lay_V/D",
            "V/E",
            "P_V_E",
            "Odd_back_V/E",
            "Odd_lay_V/E",
            "V/V",
            "P_V_V",
            "Odd_back_V/V",
            "Odd_lay_V/V",
        ]
    ]

    # Dataframe Derrota
    dataderrotahtmandante = datamandantehtft[
        [
            "Home",
            "HT_Losse_Games",
            "D/D",
            "P_D_D",
            "Odd_back_D/D",
            "Odd_lay_D/D",
            "D/E",
            "P_D_E",
            "Odd_back_D/E",
            "Odd_lay_D/E",
            "D/V",
            "P_D_V",
            "Odd_back_D/V",
            "Odd_lay_D/V",
        ]
    ]

    # Dataframe Empate
    dataempatehtmandante = datamandantehtft[
        [
            "Home",
            "HT_Draw_Games",
            "E/D",
            "P_E_D",
            "Odd_back_E/D",
            "Odd_lay_E/D",
            "E/E",
            "P_E_E",
            "Odd_back_E/E",
            "Odd_lay_E/E",
            "E/V",
            "P_E_V",
            "Odd_back_E/V",
            "Odd_lay_E/V",
            "Odd_Lay_Draw",
            "Prob_V/D",
        ]
    ]

    # Dataframe Vitória
    datavitoriahtmandante = datamandantehtft[
        [
            "Home",
            "HT_Win_Games",
            "V/D",
            "P_V_D",
            "Odd_back_V/D",
            "Odd_lay_V/D",
            "V/E",
            "P_V_E",
            "Odd_back_V/E",
            "Odd_lay_V/E",
            "V/V",
            "P_V_V",
            "Odd_back_V/V",
            "Odd_lay_V/V",
        ]
    ]

    # Dataframe Informações Gerais
    datainfohome = datamandantehtft[
        ["Home", "Games", "HT_Losse_Games", "HT_Draw_Games", "HT_Win_Games"]
    ]

    st.dataframe(
        datainfohome,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "Games": st.column_config.NumberColumn("Jogos"),
            "HT_Losse_Games": st.column_config.NumberColumn("Qtd. Derrotas HT"),
            "HT_Draw_Games": st.column_config.NumberColumn("Qtd. Empates HT"),
            "HT_Win_Games": st.column_config.NumberColumn("Qtd. Vitórias HT"),
        },
    )

    st.dataframe(
        datamandantegoalsht,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Jogos"),
            "total_gols_home": st.column_config.NumberColumn("Gols Marcados HT"),
            "total_gols_away": st.column_config.NumberColumn("Gols Sofridos HT"),
            "acima_05": st.column_config.NumberColumn("Jogos Acima 0.5 Gols HT"),
            "perc_acima_05": st.column_config.NumberColumn(
                "Prob. Acima 0.5 Gols HT",
                format="%.2f%%",
            ),
            "Odd_perc_acima_05": st.column_config.NumberColumn(
                "Odd Prob. Acima 0.5 Gols HT",
                format="%.2f",
            ),
            "acima_15": st.column_config.NumberColumn("Jogos Acima 1.5 Gols HT"),
            "perc_acima_15": st.column_config.NumberColumn(
                "Prob. Acima 1.5 Gols HT",
                format="%.2f%%",
            ),
            "Odd_perc_acima_15": st.column_config.NumberColumn(
                "Odd Prob. Acima 1.5 Gols HT",
                format="%.2f",
            ),
        },
    )

    st.dataframe(
        dataderrotahtmandante,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "HT_Losse_Games": st.column_config.NumberColumn("Qtd. Derrotas HT"),
            # Derrota -> Derrota
            "D/D": st.column_config.NumberColumn("D/D"),
            "P_D_D": st.column_config.NumberColumn("P(D/D)", format="%.2f%%"),
            "Odd_back_D/D": st.column_config.NumberColumn(
                "Odd Back D/D",
                format="%.2f",
            ),
            "Odd_lay_D/D": st.column_config.NumberColumn("Odd Lay D/D", format="%.2f"),
            # Derrota -> Empate
            "D/E": st.column_config.NumberColumn("D/E"),
            "P_D_E": st.column_config.NumberColumn("P(D/E)", format="%.2f%%"),
            "Odd_back_D/E": st.column_config.NumberColumn(
                "Odd Back D/E",
                format="%.2f",
            ),
            "Odd_lay_D/E": st.column_config.NumberColumn("Odd Lay D/E", format="%.2f"),
            # Derrota -> Vitória
            "D/V": st.column_config.NumberColumn("D/V"),
            "P_D_V": st.column_config.NumberColumn("P(D/V)", format="%.2f%%"),
            "Odd_back_D/V": st.column_config.NumberColumn(
                "Odd Back D/V",
                format="%.2f",
            ),
            "Odd_lay_D/V": st.column_config.NumberColumn("Odd Lay D/V", format="%.2f"),
        },
    )

    dataempatehtmandante_styled = dataempatehtmandante.style.apply(
        lambda x: [
            "background-color: #8d97a9; color: black" if (x["Prob_V/D"] > 76) else "",
        ],
        axis=1,
        subset=["Prob_V/D"],
    )

    st.dataframe(
        dataempatehtmandante_styled,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "HT_Draw_Games": st.column_config.NumberColumn("Qtd. Empates HT"),
            # Empate -> Derrota
            "E/D": st.column_config.NumberColumn("E/D"),
            "P_E_D": st.column_config.NumberColumn("P(E/D)", format="%.2f%%"),
            "Odd_back_E/D": st.column_config.NumberColumn(
                "Odd Back E/D",
                format="%.2f",
            ),
            "Odd_lay_E/D": st.column_config.NumberColumn("Odd Lay E/D", format="%.2f"),
            # Empate -> Empate
            "E/E": st.column_config.NumberColumn("E/E"),
            "P_E_E": st.column_config.NumberColumn("P(E/E)", format="%.2f%%"),
            "Odd_back_E/E": st.column_config.NumberColumn(
                "Odd Back E/E",
                format="%.2f",
            ),
            "Odd_lay_E/E": st.column_config.NumberColumn("Odd Lay E/E", format="%.2f"),
            # Empate -> Vitória
            "E/V": st.column_config.NumberColumn("E/V"),
            "P_E_V": st.column_config.NumberColumn("P(E/V)", format="%.2f%%"),
            "Odd_back_E/V": st.column_config.NumberColumn(
                "Odd Back E/V",
                format="%.2f",
            ),
            "Odd_lay_E/V": st.column_config.NumberColumn("Odd Lay E/V", format="%.2f"),
            # Lay Draw
            "Odd_Lay_Draw": st.column_config.NumberColumn(
                "Odd Lay Draw",
                format="%.2f",
            ),
            "Prob_V/D": st.column_config.NumberColumn("Prob. V/D", format="%.2f%%"),
        },
    )

    st.dataframe(
        datavitoriahtmandante,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "HT_Win_Games": st.column_config.NumberColumn("Qtd. Vitórias HT"),
            # Vitória -> Derrota
            "V/D": st.column_config.NumberColumn("V/D"),
            "P_V_D": st.column_config.NumberColumn("P(V/D)", format="%.2f%%"),
            "Odd_back_V/D": st.column_config.NumberColumn(
                "Odd Back V/D",
                format="%.2f",
            ),
            "Odd_lay_V/D": st.column_config.NumberColumn("Odd Lay V/D", format="%.2f"),
            # Vitória -> Empate
            "V/E": st.column_config.NumberColumn("V/E"),
            "P_V_E": st.column_config.NumberColumn("P(V/E)", format="%.2f%%"),
            "Odd_back_V/E": st.column_config.NumberColumn(
                "Odd Back V/E",
                format="%.2f",
            ),
            "Odd_lay_V/E": st.column_config.NumberColumn("Odd Lay V/E", format="%.2f"),
            # Vitória -> Vitória
            "V/V": st.column_config.NumberColumn("V/V"),
            "P_V_V": st.column_config.NumberColumn("P(V/V)", format="%.2f%%"),
            "Odd_back_V/V": st.column_config.NumberColumn(
                "Odd Back V/V",
                format="%.2f",
            ),
            "Odd_lay_V/V": st.column_config.NumberColumn("Odd Lay V/V", format="%.2f"),
        },
    )

    st.write("---")
    st.subheader("⚽ Informações do HT (0 x 0)")

    datamanadanteht00_styled = datamanadanteht00.style.apply(
        lambda x: (
            ["background-color: #8d97a9; color: black"] * len(x)
            if (x["onGoal_percent"] >= 80) and (x["Percentmmean"] > 0.80)
            else [""] * len(x)
        ),
        axis=1,
        subset=datamanadanteht00.columns,  # Aplica a todas as colunas
    )

    st.dataframe(
        datamanadanteht00_styled,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Jogos (0 x 0) HT"),
            "onGoal_count": st.column_config.NumberColumn("Jogos Com Gol FT"),
            "total_gols_home": st.column_config.NumberColumn("Gols M. no 2ºT FT"),
            "media_gols_home_ft": st.column_config.NumberColumn(
                "Média de Gols M. no 2ºT FT",
                format="%.2f",
            ),
            "total_gols_away": st.column_config.NumberColumn("Gols S. no 2ºT FT"),
            "media_gols_away_ft": st.column_config.NumberColumn(
                "Média de Gols S. no 2ºT FT",
                format="%.2f",
            ),
            "Percentmmean": st.column_config.NumberColumn(
                "Média Home - Away",
                format="%.2f",
            ),
            "onGoal_percent": st.column_config.NumberColumn(
                "Prob. de Gol no 2ºT FT",
                format="%.2f%%",
            ),
            "Odd_onGoal": st.column_config.NumberColumn(
                "Odd Prob. de Gol no 2ºT FT",
                format="%.2f",
            ),
        },
    )

    st.dataframe(
        datamandantehtftresult,
        hide_index=True,
        width="content",
        column_config={
            "Home": st.column_config.TextColumn("Time Mandante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Total Jogos"),
            "0x0": st.column_config.NumberColumn("J. FT 0 x 0"),
            "0x0_pct": st.column_config.NumberColumn("% J. FT 0 x 0", format="%.2f%%"),
            "1x0": st.column_config.NumberColumn("J. FT 1 x 0"),
            "1x0_pct": st.column_config.NumberColumn("% J. FT 1 x 0", format="%.2f%%"),
            "0x1": st.column_config.NumberColumn("J. FT 0 x 1"),
            "0x1_pct": st.column_config.NumberColumn("% J. FT 0 x 1", format="%.2f%%"),
            "1x1": st.column_config.NumberColumn("J. FT 1 x 1"),
            "1x1_pct": st.column_config.NumberColumn("% J. FT 1 x 1", format="%.2f%%"),
            "2x0": st.column_config.NumberColumn("J. FT 2 x 0"),
            "2x0_pct": st.column_config.NumberColumn("% J. FT 2 x 0", format="%.2f%%"),
            "0x2": st.column_config.NumberColumn("J. FT 0 x 2"),
            "0x2_pct": st.column_config.NumberColumn("% J. FT 0 x 2", format="%.2f%%"),
            "other_results": st.column_config.NumberColumn("Outros Resultados"),
            "other_results_pct": st.column_config.NumberColumn(
                "Outros Resultados %",
                format="%.2f%%",
            ),
        },
    )

with tab3:
    st.subheader("⚽ Informações do HT")

    away = st.selectbox(
        "Selecione o time visitante:",
        load_data_total()["Away"].sort_values().unique(),
    )

    # Exibir os dados baseado no home - time selecionado
    datavisitantehtft = load_createAwayHTFT()[load_createAwayHTFT()["Away"] == away]

    # Exibr os dados baseado no home - - time selecionado
    datavisitantegoalsht = load_createAwayHTGoals()[
        load_createAwayHTGoals()["Away"] == away
    ]

    datavisitanteht00 = load_createAwayHTGoals00()[
        load_createAwayHTGoals00()["Away"] == away
    ]

    datavisitantehtftresult = load_createAwayHTGoals00Result()[
        load_createAwayHTGoals00Result()["Away"] == away
    ]

    # Organizar as colunas
    datavisitantehtft = datavisitantehtft[
        [
            "Away",
            "Games",
            "HT_Losse_Games",
            "D/D",
            "P_D_D",
            "Odd_back_D/D",
            "Odd_lay_D/D",
            "D/E",
            "P_D_E",
            "Odd_back_D/E",
            "Odd_lay_D/E",
            "D/V",
            "P_D_V",
            "Odd_back_D/V",
            "Odd_lay_D/V",
            "HT_Draw_Games",
            "E/D",
            "P_E_D",
            "Odd_back_E/D",
            "Odd_lay_E/D",
            "E/E",
            "P_E_E",
            "Odd_back_E/E",
            "Odd_lay_E/E",
            "E/V",
            "P_E_V",
            "Odd_back_E/V",
            "Odd_lay_E/V",
            "Odd_Lay_Draw",
            "Prob_V/D",
            "HT_Win_Games",
            "V/D",
            "P_V_D",
            "Odd_back_V/D",
            "Odd_lay_V/D",
            "V/E",
            "P_V_E",
            "Odd_back_V/E",
            "Odd_lay_V/E",
            "V/V",
            "P_V_V",
            "Odd_back_V/V",
            "Odd_lay_V/V",
        ]
    ]

    # Dataframe Derrota
    dataderrotahtvisitante = datavisitantehtft[
        [
            "Away",
            "HT_Losse_Games",
            "D/D",
            "P_D_D",
            "Odd_back_D/D",
            "Odd_lay_D/D",
            "D/E",
            "P_D_E",
            "Odd_back_D/E",
            "Odd_lay_D/E",
            "D/V",
            "P_D_V",
            "Odd_back_D/V",
            "Odd_lay_D/V",
        ]
    ]

    # Dataframe Empate
    dataempatehtvisitante = datavisitantehtft[
        [
            "Away",
            "HT_Draw_Games",
            "E/D",
            "P_E_D",
            "Odd_back_E/D",
            "Odd_lay_E/D",
            "E/E",
            "P_E_E",
            "Odd_back_E/E",
            "Odd_lay_E/E",
            "E/V",
            "P_E_V",
            "Odd_back_E/V",
            "Odd_lay_E/V",
            "Odd_Lay_Draw",
            "Prob_V/D",
        ]
    ]

    # Dataframe Vitória
    datavitoriahtvisitante = datavisitantehtft[
        [
            "Away",
            "HT_Win_Games",
            "V/D",
            "P_V_D",
            "Odd_back_V/D",
            "Odd_lay_V/D",
            "V/E",
            "P_V_E",
            "Odd_back_V/E",
            "Odd_lay_V/E",
            "V/V",
            "P_V_V",
            "Odd_back_V/V",
            "Odd_lay_V/V",
        ]
    ]

    # Dataframe Informações Gerais
    datainfoaway = datavisitantehtft[
        ["Away", "Games", "HT_Losse_Games", "HT_Draw_Games", "HT_Win_Games"]
    ]

    st.dataframe(
        datainfoaway,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "Games": st.column_config.NumberColumn("Jogos"),
            "HT_Losse_Games": st.column_config.NumberColumn("Qtd. Derrotas HT"),
            "HT_Draw_Games": st.column_config.NumberColumn("Qtd. Empates HT"),
            "HT_Win_Games": st.column_config.NumberColumn("Qtd. Vitórias HT"),
        },
    )

    st.dataframe(
        datavisitantegoalsht,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Jogos"),
            "total_gols_home": st.column_config.NumberColumn("Gols Marcados HT"),
            "total_gols_away": st.column_config.NumberColumn("Gols Sofridos HT"),
            "acima_05": st.column_config.NumberColumn("Jogos Acima 0.5 Gols HT"),
            "perc_acima_05": st.column_config.NumberColumn(
                "Prob. Acima 0.5 Gols HT",
                format="%.2f%%",
            ),
            "Odd_perc_acima_05": st.column_config.NumberColumn(
                "Odd Prob. Acima 0.5 Gols HT",
                format="%.2f",
            ),
            "acima_15": st.column_config.NumberColumn("Jogos Acima 1.5 Gols HT"),
            "perc_acima_15": st.column_config.NumberColumn(
                "Prob. Acima 1.5 Gols HT",
                format="%.2f%%",
            ),
            "Odd_perc_acima_15": st.column_config.NumberColumn(
                "Odd Prob. Acima 1.5 Gols HT",
                format="%.2f",
            ),
        },
    )

    st.dataframe(
        dataderrotahtvisitante,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "HT_Losse_Games": st.column_config.NumberColumn("Qtd. Derrotas HT"),
            # Derrota -> Derrota
            "D/D": st.column_config.NumberColumn("D/D"),
            "P_D_D": st.column_config.NumberColumn("P(D/D)", format="%.2f%%"),
            "Odd_back_D/D": st.column_config.NumberColumn(
                "Odd Back D/D",
                format="%.2f",
            ),
            "Odd_lay_D/D": st.column_config.NumberColumn("Odd Lay D/D", format="%.2f"),
            # Derrota -> Empate
            "D/E": st.column_config.NumberColumn("D/E"),
            "P_D_E": st.column_config.NumberColumn("P(D/E)", format="%.2f%%"),
            "Odd_back_D/E": st.column_config.NumberColumn(
                "Odd Back D/E",
                format="%.2f",
            ),
            "Odd_lay_D/E": st.column_config.NumberColumn("Odd Lay D/E", format="%.2f"),
            # Derrota -> Vitória
            "D/V": st.column_config.NumberColumn("D/V"),
            "P_D_V": st.column_config.NumberColumn("P(D/V)", format="%.2f%%"),
            "Odd_back_D/V": st.column_config.NumberColumn(
                "Odd Back D/V",
                format="%.2f",
            ),
            "Odd_lay_D/V": st.column_config.NumberColumn("Odd Lay D/V", format="%.2f"),
        },
    )

    dataempatehtvisitante_styled = dataempatehtvisitante.style.apply(
        lambda x: [
            "background-color: #8d97a9; color: black" if (x["Prob_V/D"] > 76) else "",
        ],
        axis=1,
        subset=["Prob_V/D"],
    )

    st.dataframe(
        dataempatehtvisitante_styled,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "HT_Draw_Games": st.column_config.NumberColumn("Qtd. Empates HT"),
            # Empate -> Derrota
            "E/D": st.column_config.NumberColumn("E/D"),
            "P_E_D": st.column_config.NumberColumn("P(E/D)", format="%.2f%%"),
            "Odd_back_E/D": st.column_config.NumberColumn(
                "Odd Back E/D",
                format="%.2f",
            ),
            "Odd_lay_E/D": st.column_config.NumberColumn("Odd Lay E/D", format="%.2f"),
            # Empate -> Empate
            "E/E": st.column_config.NumberColumn("E/E"),
            "P_E_E": st.column_config.NumberColumn("P(E/E)", format="%.2f%%"),
            "Odd_back_E/E": st.column_config.NumberColumn(
                "Odd Back E/E",
                format="%.2f",
            ),
            "Odd_lay_E/E": st.column_config.NumberColumn("Odd Lay E/E", format="%.2f"),
            # Empate -> Vitória
            "E/V": st.column_config.NumberColumn("E/V"),
            "P_E_V": st.column_config.NumberColumn("P(E/V)", format="%.2f%%"),
            "Odd_back_E/V": st.column_config.NumberColumn(
                "Odd Back E/V",
                format="%.2f",
            ),
            "Odd_lay_E/V": st.column_config.NumberColumn("Odd Lay E/V", format="%.2f"),
            # Lay Draw
            "Odd_Lay_Draw": st.column_config.NumberColumn(
                "Odd Lay Draw",
                format="%.2f",
            ),
            "Prob_V/D": st.column_config.NumberColumn("Prob. V/D", format="%.2f%%"),
        },
    )

    st.dataframe(
        datavitoriahtvisitante,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "HT_Win_Games": st.column_config.NumberColumn("Qtd. Vitórias HT"),
            # Vitória -> Derrota
            "V/D": st.column_config.NumberColumn("V/D"),
            "P_V_D": st.column_config.NumberColumn("P(V/D)", format="%.2f%%"),
            "Odd_back_V/D": st.column_config.NumberColumn(
                "Odd Back V/D",
                format="%.2f",
            ),
            "Odd_lay_V/D": st.column_config.NumberColumn("Odd Lay V/D", format="%.2f"),
            # Vitória -> Empate
            "V/E": st.column_config.NumberColumn("V/E"),
            "P_V_E": st.column_config.NumberColumn("P(V/E)", format="%.2f%%"),
            "Odd_back_V/E": st.column_config.NumberColumn(
                "Odd Back V/E",
                format="%.2f",
            ),
            "Odd_lay_V/E": st.column_config.NumberColumn("Odd Lay V/E", format="%.2f"),
            # Vitória -> Vitória
            "V/V": st.column_config.NumberColumn("V/V"),
            "P_V_V": st.column_config.NumberColumn("P(V/V)", format="%.2f%%"),
            "Odd_back_V/V": st.column_config.NumberColumn(
                "Odd Back V/V",
                format="%.2f",
            ),
            "Odd_lay_V/V": st.column_config.NumberColumn("Odd Lay V/V", format="%.2f"),
        },
    )

    st.write("---")
    st.subheader("⚽ Informações do HT (0 x 0)")

    datavisitanteht00_styled = datavisitanteht00.style.apply(
        lambda x: (
            ["background-color: #8d97a9; color: black"] * len(x)
            if (x["onGoal_percent"] >= 80) and (x["Percentmmean"] > 0.80)
            else [""] * len(x)
        ),
        axis=1,
        subset=datavisitanteht00.columns,  # Aplica a todas as colunas
    )

    st.dataframe(
        datavisitanteht00_styled,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Jogos (0 x 0) HT"),
            "onGoal_count": st.column_config.NumberColumn("Jogos Com Gol FT"),
            "total_gols_home": st.column_config.NumberColumn("Gols M. no 2ºT FT"),
            "media_gols_home_ft": st.column_config.NumberColumn(
                "Média de Gols M. no 2ºT FT",
                format="%.2f",
            ),
            "total_gols_away": st.column_config.NumberColumn("Gols S. no 2ºT FT"),
            "media_gols_away_ft": st.column_config.NumberColumn(
                "Média de Gols S. no 2ºT FT",
                format="%.2f",
            ),
            "Percentmmean": st.column_config.NumberColumn(
                "Média Home - Away",
                format="%.2f",
            ),
            "onGoal_percent": st.column_config.NumberColumn(
                "Prob. de Gol no 2ºT FT",
                format="%.2f%%",
            ),
            "Odd_onGoal": st.column_config.NumberColumn(
                "Odd Prob. de Gol no 2ºT FT",
                format="%.2f",
            ),
        },
    )

    st.dataframe(
        datavisitantehtftresult,
        hide_index=True,
        width="content",
        column_config={
            "Away": st.column_config.TextColumn("Time Visitante", width="medium"),
            "total_jogos": st.column_config.NumberColumn("Total Jogos"),
            "0x0": st.column_config.NumberColumn("J. FT 0 x 0"),
            "0x0_pct": st.column_config.NumberColumn("% J. FT 0 x 0", format="%.2f%%"),
            "1x0": st.column_config.NumberColumn("J. FT 1 x 0"),
            "1x0_pct": st.column_config.NumberColumn("% J. FT 1 x 0", format="%.2f%%"),
            "0x1": st.column_config.NumberColumn("J. FT 0 x 1"),
            "0x1_pct": st.column_config.NumberColumn("% J. FT 0 x 1", format="%.2f%%"),
            "1x1": st.column_config.NumberColumn("J. FT 1 x 1"),
            "1x1_pct": st.column_config.NumberColumn("% J. FT 1 x 1", format="%.2f%%"),
            "2x0": st.column_config.NumberColumn("J. FT 2 x 0"),
            "2x0_pct": st.column_config.NumberColumn("% J. FT 2 x 0", format="%.2f%%"),
            "0x2": st.column_config.NumberColumn("J. FT 0 x 2"),
            "0x2_pct": st.column_config.NumberColumn("% J. FT 0 x 2", format="%.2f%%"),
            "other_results": st.column_config.NumberColumn("Outros Resultados"),
            "other_results_pct": st.column_config.NumberColumn(
                "Outros Resultados %",
                format="%.2f%%",
            ),
        },
    )


with tab4:
    # Selecionar data
    dateJD = st.selectbox(
        "Selecione a data:",
        load_data_today()["Date"].sort_values().unique(),
    )
    st.dataframe(
        load_data_today()[load_data_today()["Date"] == dateJD],
        hide_index=True,
        width="stretch",
    )

    jogosdiaselecionado1 = load_data_today()[load_data_today()["Date"] == dateJD][
        [
            "Date",
            "Time",
            "League",
            "Home",
            "Away",
            "Odd_H_Back",
            "Odd_A_Back",
            "Odd_CS_0x1_Lay",
        ]
    ]

    filtro1 = lambda df: (
        df[
            (  # 1º Condição 1.80-2.09 | 4.00-4.99 | 20.0+
                (df["Odd_H_Back"] >= 1.80)
                & (df["Odd_H_Back"] <= 2.09)
                & (df["Odd_A_Back"] >= 4.00)
                & (df["Odd_A_Back"] <= 4.99)
                & (df["Odd_CS_0x1_Lay"] >= 20.00)
            )
            | (  # 2º Condição 1.80-2.09 | 4.00-4.99 | 13.0-13.9
                (df["Odd_H_Back"] >= 1.80)
                & (df["Odd_H_Back"] <= 2.09)
                & (df["Odd_A_Back"] >= 4.00)
                & (df["Odd_A_Back"] <= 4.900)
                & (df["Odd_CS_0x1_Lay"] >= 13.00)
                & (df["Odd_CS_0x1_Lay"] <= 19.90)
            )
            | (  # 3º Condição 2.10-2.49 | 3.50-3.99 | 12.0-12.9
                (df["Odd_H_Back"] >= 2.10)
                & (df["Odd_H_Back"] <= 2.49)
                & (df["Odd_A_Back"] >= 3.50)
                & (df["Odd_A_Back"] <= 3.900)
                & (df["Odd_CS_0x1_Lay"] >= 12.00)
                & (df["Odd_CS_0x1_Lay"] <= 12.90)
            )
            | (  # 4º Condição 1.80-2.09 | 4.00-4.99 | 18.0-19.9
                (df["Odd_H_Back"] >= 1.80)
                & (df["Odd_H_Back"] <= 2.09)
                & (df["Odd_A_Back"] >= 4.00)
                & (df["Odd_A_Back"] <= 4.99)
                & (df["Odd_CS_0x1_Lay"] >= 18.00)
                & (df["Odd_CS_0x1_Lay"] <= 19.90)
            )
            | (  # 5° Condição 2.10-2.49 | 3.50-3.99 | 15.0-15.9
                (df["Odd_H_Back"] >= 2.10)
                & (df["Odd_H_Back"] <= 2.49)
                & (df["Odd_A_Back"] >= 3.50)
                & (df["Odd_A_Back"] <= 3.99)
                & (df["Odd_CS_0x1_Lay"] >= 15.00)
                & (df["Odd_CS_0x1_Lay"] <= 15.90)
            )
            | (  # 6º Condição 2.50-2.99 | 2.50-2.99 | 11.0-11.9
                (df["Odd_H_Back"] >= 2.50)
                & (df["Odd_H_Back"] <= 2.99)
                & (df["Odd_A_Back"] >= 2.50)
                & (df["Odd_A_Back"] <= 2.99)
                & (df["Odd_CS_0x1_Lay"] >= 11.00)
                & (df["Odd_CS_0x1_Lay"] <= 11.90)
            )
            | (  # 7º Condição 1.80-2.09 | 5.00+ | 15.0-15.9
                (df["Odd_H_Back"] >= 1.80)
                & (df["Odd_H_Back"] <= 2.09)
                & (df["Odd_A_Back"] >= 5.00)
                & (df["Odd_CS_0x1_Lay"] >= 15.00)
                & (df["Odd_CS_0x1_Lay"] <= 15.90)
            )
            | (  # 8º Condição 1.80-2.09 | 5.00+ | 14.0-14.9
                (df["Odd_H_Back"] >= 1.80)
                & (df["Odd_H_Back"] <= 2.09)
                & (df["Odd_A_Back"] >= 5.00)
                & (df["Odd_CS_0x1_Lay"] >= 14.00)
                & (df["Odd_CS_0x1_Lay"] <= 14.90)
            )
            | (  # 9° Condição 2.10-2.49 | 4.00-4.99 | 11.0-11.9
                (df["Odd_H_Back"] >= 2.10)
                & (df["Odd_H_Back"] <= 2.49)
                & (df["Odd_A_Back"] >= 4.00)
                & (df["Odd_A_Back"] <= 4.99)
                & (df["Odd_CS_0x1_Lay"] >= 11.00)
                & (df["Odd_CS_0x1_Lay"] <= 11.90)
            )
            | (  # 10º Condição 2.10-2.49 | 3.50-3.99 | 16.0-17.9
                (df["Odd_H_Back"] >= 2.10)
                & (df["Odd_H_Back"] <= 2.49)
                & (df["Odd_A_Back"] >= 3.50)
                & (df["Odd_A_Back"] <= 3.99)
                & (df["Odd_CS_0x1_Lay"] >= 16.00)
                & (df["Odd_CS_0x1_Lay"] <= 17.90)
            )
        ]
        if isinstance(df, pd.DataFrame)
        else pd.DataFrame()
    )

    jogosfiltrados1 = filtro1(jogosdiaselecionado1)

    st.write("### Lay 0x1")
    st.dataframe(
        jogosfiltrados1,
        hide_index=True,
        width="content",
    )
