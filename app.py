import streamlit as st
import pandas as pd

from data.fetcher import get_stock_data, InvalidTickerError
from analysis.metrics import (
    calcular_retorno_acumulado,
    calcular_volatilidade_anual,
    calcular_drawdown_maximo,
    calcular_sharpe,
    get_selic_atual,
)
from components.charts import (
    grafico_candlestick,
    grafico_retorno_acumulado,
    heatmap_correlacao,
)

st.set_page_config(
    page_title="Dashboard B3",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


@st.cache_data(ttl=3600)
def carregar_dados(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        return get_stock_data(ticker, period)
    except InvalidTickerError:
        return None


@st.cache_data(ttl=3600)
def selic_atual() -> float:
    return get_selic_atual()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Dashboard B3")

    raw_input = st.text_input(
        "Tickers (separados por virgula)",
        value="PETR4.SA, WEGE3.SA, GGRC11.SA",
    )
    period = st.selectbox(
        "Periodo",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=3,
    )

    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    st.divider()
    selic = selic_atual()
    st.metric("Selic atual (a.a.)", f"{selic * 100:.2f}%")

# ── Carregamento de dados ─────────────────────────────────────────────────────

dfs: dict[str, pd.DataFrame] = {}
erros: list[str] = []

for ticker in tickers:
    df = carregar_dados(ticker, period)
    if df is None:
        erros.append(ticker)
    else:
        dfs[ticker] = df

if erros:
    st.warning(f"Tickers nao encontrados e ignorados: {', '.join(erros)}")

if not dfs:
    st.error("Nenhum ticker valido. Verifique os tickers na sidebar.")
    st.stop()

valid_tickers = list(dfs.keys())
valid_dfs = list(dfs.values())

# ── Abas ─────────────────────────────────────────────────────────────────────

aba_geral, aba_comp, aba_det = st.tabs(["Visao Geral", "Comparativo", "Detalhes"])

# ── Aba 1: Visao Geral ────────────────────────────────────────────────────────

with aba_geral:
    primeiro = valid_tickers[0]
    df0 = dfs[primeiro]

    st.subheader(primeiro)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Retorno Acumulado",
        f"{calcular_retorno_acumulado(df0) * 100:+.2f}%",
    )
    c2.metric(
        "Volatilidade Anual",
        f"{calcular_volatilidade_anual(df0) * 100:.2f}%",
    )
    c3.metric(
        "Drawdown Maximo",
        f"{calcular_drawdown_maximo(df0) * 100:.2f}%",
    )
    c4.metric(
        "Sharpe",
        f"{calcular_sharpe(df0, risk_free=selic):.4f}",
        help=f"Calculado com Selic de {selic * 100:.2f}% a.a.",
    )

    st.plotly_chart(
        grafico_candlestick(df0, primeiro),
        use_container_width=True,
    )

# ── Aba 2: Comparativo ────────────────────────────────────────────────────────

with aba_comp:
    st.plotly_chart(
        grafico_retorno_acumulado(valid_dfs, valid_tickers),
        use_container_width=True,
    )

    if len(valid_tickers) >= 2:
        st.plotly_chart(
            heatmap_correlacao(valid_dfs, valid_tickers),
            use_container_width=True,
        )
    else:
        st.info("Adicione pelo menos 2 tickers para exibir o heatmap de correlacao.")

# ── Aba 3: Detalhes ───────────────────────────────────────────────────────────

with aba_det:
    rows = []
    for ticker, df in dfs.items():
        rows.append({
            "Ticker": ticker,
            "Retorno Acumulado (%)": round(calcular_retorno_acumulado(df) * 100, 2),
            "Volatilidade Anual (%)": round(calcular_volatilidade_anual(df) * 100, 2),
            "Drawdown Maximo (%)": round(calcular_drawdown_maximo(df) * 100, 2),
            "Sharpe": round(calcular_sharpe(df, risk_free=selic), 4),
            "Ultimo Close (R$)": round(float(df["Close"].iloc[-1]), 2),
            "Vol. Medio Diario": int(df["Volume"].mean()),
            "Pregoes": len(df),
        })

    tabela = pd.DataFrame(rows).set_index("Ticker")

    st.dataframe(
        tabela.style.format({
            "Retorno Acumulado (%)": "{:+.2f}",
            "Volatilidade Anual (%)": "{:.2f}",
            "Drawdown Maximo (%)": "{:.2f}",
            "Sharpe": "{:.4f}",
            "Ultimo Close (R$)": "R$ {:.2f}",
            "Vol. Medio Diario": "{:,.0f}",
        }),
        use_container_width=True,
    )
