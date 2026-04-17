# Guia de Criação de Bins - Estratégia Lay 0x1

Este documento descreve a lógica utilizada para categorizar os mercados e as estatísticas em "Bins" (faixas) para análise de portfolio.

## 1. Cálculo da Média de Gols Mandante
Antes de criar o bin de média, calculamos a média histórica de gols marcados pelo time em casa:

```python
# Calcular média de gols marcados em casa por cada time
avg_goals_home = datatest.groupby('Home')['Goals_H_FT'].mean().reset_index()
avg_goals_home.columns = ['Home', 'Avg_Goals_H']
datatest = datatest.merge(avg_goals_home, on='Home', how='left')
```

## 2. Lógica de Categorização (Bins)

Utilizamos a função `pd.cut` do Pandas para transformar valores contínuos em categorias discretas:

```python
import pandas as pd

# Bin: Odd Back Mandante (H)
datatest['Bin_H'] = pd.cut(
    datatest['Odd_H_Back'],
    bins=[1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 100],
    labels=['<1.3', '1.3-1.5', '1.5-1.7', '1.7-2.0', '2.1-2.5', '2.6-3.0', '3.0+']
)

# Bin: Odd Over 2.5 FT
datatest['Bin_Over'] = pd.cut(
    datatest['Odd_Over25_FT_Back'],
    bins=[0, 1.6, 1.8, 2.0, 100],
    labels=['<1.6', '1.6-1.8', '1.8-2.0', '2.0+']
)

# Bin: Odd Ambas Marcam (BTTS)
datatest['Bin_BTTS'] = pd.cut(
    datatest['Odd_BTTS_Yes_Back'],
    bins=[0, 1.6, 1.8, 2.0, 100],
    labels=['<1.6', '1.6-1.8', '1.8-2.0', '2.0+']
)

# Bin: Odd Lay 0x1 (Responsabilidade)
datatest['Bin_Lay'] = pd.cut(
    datatest['Odd_CS_0x1_Lay'],
    bins=[0, 10, 15, 20, 30, 100],
    labels=['<10', '10-15', '15-20', '20-30', '30+']
)

# Bin: Média de Gols Mandante em Casa
datatest['Bin_Avg_H'] = pd.cut(
    datatest['Avg_Goals_H'],
    bins=[0, 1.2, 1.5, 1.8, 5.0],
    labels=['<1.2', '1.2-1.5', '1.5-1.8', '1.8+']
)
```

## 3. Utilidade para Análise de Portfolio
Ao combinar esses 5 bins, criamos chaves únicas de mercado:
`('2.1-2.5', '2.0+', '1.8-2.0', '10-15', '1.2-1.5')`

Isso permite identificar quais "nichos" de jogos são lucrativos no longo prazo, filtrando automaticamente jogos com baixa expectativa de gols ou odds desajustadas.
