# Dashboard B3

## O que é esse projeto
Dashboard de análise de ações brasileiras com Python, yfinance e Streamlit.

## Stack
- Python 3.11+
- Streamlit (UI)
- yfinance (dados de mercado)
- Plotly (gráficos)
- Pandas (manipulação)

## Convenções
- Tickers B3 sempre com sufixo .SA (ex: PETR4.SA)
- Funções com type hints e docstrings
- Separar lógica de dados, análise e visualização em módulos

## Como rodar
streamlit run app.py

## Status atual
- [x] fetcher.py
- [ ] metrics.py
- [ ] charts.py
- [ ] app.py