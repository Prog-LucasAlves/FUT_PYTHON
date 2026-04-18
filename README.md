# 📊 Análise de Estratégia: Lay 0x1 (Correct Score)

Este projeto automatiza a análise estatística para a estratégia de **Lay 0x1 (Aposta contra o placar de 0-1)** no mercado de Correct Score da Betfair, integrando dados históricos e de mercado.

## 🚀 Utilidade do App (`app_lay0x1.py`)

O aplicativo é um dashboard profissional desenvolvido em Streamlit que permite:
*   **Filtragem Diária**: Identifica jogos do dia com as melhores probabilidades para a estratégia.
*   **Veredito IA (Scoring 16 pts)**: Atribui uma nota de confiança de 0 a 16 para cada entrada.
*   **Análise de Risco no HT**: Calcula a probabilidade de o jogo "quebrar" (terminar 0x1) vindo de cenários de 0x0 ou 0x1 ao intervalo.
*   **Perfil Institucional**: Avalia se as odds da Betfair condizem com os padrões históricos de lucratividade (Golden Portfolio).

---

## 🎯 Modelo de Pontuação (16 Pontos)

O sistema utiliza um modelo de **16 critérios** para validar uma entrada. Quanto maior a pontuação, mais segura é a operação de Lay 0x1.

### 1. Padrão de Odds (Peso 5)
*   **Utilidade**: Identifica as "Faixas de Lucratividade". O sistema compara as Match Odds (Mandante/Visitante) e a Odd do Lay 0x1 para detectar desajustes matemáticos onde o lucro a longo prazo é comprovado.

### 2. Heurística Poisson HT (Peso 2)
*   **Critério**: Probabilidade matemática calculada via Distribuição de Poisson para o placar 0x1 no intervalo.
*   **Utilidade**: Pontua mais se a chance de um 0x1 precoce for baixa (< 7%), evitando jogos com gols inesperados do visitante no início.

### 3. Expectativa de Gols do Mercado (Peso 3)
*   **BTTS Yes Back < 1.90**: Indica que o mercado espera que ambos marquem (+2 pts). Se ambos marcam, o 0x1 é impossível.
*   **Over 2.5 FT Back < 1.90**: Indica expectativa de 3 ou mais gols (+1 pt). Quanto mais gols, menor a chance de um placar magro de 0x1.

### 4. Histórico Direto - H2H (Peso 1)
*   **Critério**: Frequência histórica do placar 0x1 entre os times.
*   **Utilidade**: Pontua se em menos de 10% dos confrontos históricos o resultado foi 0x1.

### 5. Defesa do Mandante - Clean Sheet (Peso 1)
*   **Critério**: Taxa de Clean Sheet do time da casa >= 25%.
*   **Utilidade**: Se o mandante tem facilidade em não sofrer gols, o risco do "1" do visitante no placar 0x1 diminui drasticamente.

### 6. Performance e Produção Ofensiva (Peso 3)
*   **PPG Mandante >= 1.60**: Superioridade técnica e controle do jogo (+1 pt).
*   **xG Mandante > 1.50**: Indica que o time da casa cria chances claras de gol (+1 pt).
*   **Custo do Gol > 1.20**: Eficiência em evitar placares magros e converter pressão em gols (+1 pt).

### 7. Confiança da Amostra (Peso 1)
*   **Utilidade**: Garante que os dados sejam estatisticamente relevantes (Amostra > 50 jogos). Se a amostra for pequena, o sistema penaliza a nota final (-1 pt).

---

## 🚦 Classificação de Recomendação
*   **11 a 16 pts**: FORTE INDICAÇÃO (Alta confiança, riscos matemáticos baixos).
*   **7 a 10 pts**: INDICAÇÃO MODERADA (Requer leitura de jogo ao vivo).
*   **Abaixo de 7 pts**: NÃO INDICADO (Risco de variância elevado).

---

## 🛠️ Requisitos
*   Python 3.x
*   Streamlit, Pandas, Plotly, Scipy
*   Dados: Betfair (Odds) + Footystats (Estatísticas avançadas)
