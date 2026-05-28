# Dashboard B3

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.7-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-1.4-0066CC?style=flat-square)

Dashboard interativo para análise de ações da B3 (Bolsa de Valores do Brasil). Permite comparar múltiplos ativos, calcular métricas de risco e retorno e visualizar correlações — tudo com dados atualizados em tempo real via Yahoo Finance e taxa Selic automática via API do Banco Central.

---

## Screenshot

> _Placeholder — adicione uma captura de tela do app rodando em `docs/screenshot.png`_
>
> ![Screenshot do Dashboard](docs/screenshot.png)

---

## Funcionalidades

### Coleta de dados
- Busca histórico de preços via **yfinance** para qualquer ativo com sufixo `.SA` (ações, FIIs, BDRs)
- Suporte a múltiplos períodos: 1 mês, 3 meses, 6 meses, 1 ano, 2 anos
- Calcula automaticamente retorno diário e retorno acumulado
- Tratamento de tickers inválidos com erro tipado (`InvalidTickerError`)

### Métricas financeiras
- **Retorno acumulado** — variação percentual total do período
- **Volatilidade anualizada** — desvio padrão dos retornos diários × √252
- **Maximum Drawdown** — maior queda do pico ao vale na série histórica
- **Índice de Sharpe anualizado** — retorno excedente sobre a Selic por unidade de risco
- **Taxa Selic automática** — buscada em tempo real na API SGS do Banco Central (série 11), com fallback de 14,50% a.a. em caso de falha

### Visualizações interativas (Plotly)
- **Candlestick com volume** — preço e volume diários em dois painéis sincronizados
- **Retorno acumulado comparativo** — múltiplas linhas em um mesmo gráfico com hover unificado
- **Heatmap de correlação** — matriz de correlação de Pearson entre retornos diários, escala RdBu

### Interface (Streamlit)
- Sidebar com input de múltiplos tickers e seletor de período
- Exibe a Selic atual buscada da API na sidebar
- **Aba Visão Geral**: candlestick do ativo principal + 4 cards de métricas
- **Aba Comparativo**: gráfico de retorno acumulado + heatmap de correlação (requer ≥ 2 ativos)
- **Aba Detalhes**: tabela completa com todas as métricas para todos os ativos
- Cache de 1 hora via `@st.cache_data` para dados de mercado e métricas

---

## Stack

| Camada | Biblioteca | Versão |
|---|---|---|
| Interface | Streamlit | 1.58 |
| Gráficos | Plotly | 6.7 |
| Dados de mercado | yfinance | 1.4 |
| Manipulação de dados | Pandas | 3.0 |
| Computação numérica | NumPy | 2.4 |
| Requisições HTTP | Requests | 2.34 |
| Runtime | Python | 3.13 |

---

## Instalação e execução local

### Pré-requisitos

- Python 3.11 ou superior
- pip

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/dashboard-b3.git
cd dashboard-b3

# 2. Crie e ative o ambiente virtual
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicie o app
streamlit run app.py
```

O dashboard abre automaticamente em `http://localhost:8501`.

---

## Estrutura do projeto

```
dashboard-b3/
│
├── app.py                  # Entrypoint Streamlit — sidebar, abas e layout
│
├── data/
│   └── fetcher.py          # get_stock_data(): busca e normaliza dados via yfinance
│
├── analysis/
│   └── metrics.py          # Cálculo de métricas financeiras + busca da Selic (BCB)
│
├── components/
│   └── charts.py           # Gráficos Plotly: candlestick, retorno acumulado, heatmap
│
├── requirements.txt        # Dependências pinadas
├── CLAUDE.md               # Contexto do projeto para o Claude Code
└── README.md
```

### Responsabilidades por módulo

**`data/fetcher.py`**
Único ponto de contato com o Yahoo Finance. Retorna um `DataFrame` padronizado com colunas `Close`, `Volume`, `Return` e `Cumulative_Return`. Lança `InvalidTickerError` para tickers inválidos.

**`analysis/metrics.py`**
Funções puras que recebem um `DataFrame` do fetcher e retornam um `float`. Nenhuma chamada de rede, exceto `get_selic_atual()`, que é isolada e tem fallback explícito.

**`components/charts.py`**
Funções que recebem `DataFrame`(s) e retornam `go.Figure`. Sem estado, sem side effects. Podem ser usadas fora do contexto Streamlit.

**`app.py`**
Orquestra os três módulos acima. Não contém lógica de negócio — apenas layout, cache e tratamento de erros de UI.

---

## Próximos passos

- [ ] Adicionar gráfico de **Bollinger Bands** e médias móveis (SMA 20/50)
- [ ] Exportar tabela de métricas como `.csv` via botão de download
- [ ] Suporte a **comparação contra benchmark** (IBOVESPA / CDI)
- [ ] Página de **backtesting** de estratégias simples (cruzamento de médias)
- [ ] Testes automatizados com `pytest` para os módulos `fetcher` e `metrics`
- [ ] Deploy no **Streamlit Community Cloud**
- [ ] Adicionar suporte a **índices** (^BVSP) e **ETFs** brasileiros
