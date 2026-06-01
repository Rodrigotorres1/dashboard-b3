import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data.fetcher import get_stock_data, InvalidTickerError
from analysis.metrics import (
    calcular_retorno_acumulado,
    calcular_volatilidade_anual,
    calcular_drawdown_maximo,
    calcular_sharpe,
    get_selic_atual,
)
from analysis.ml_model import _FEATURES, treinar_modelo, prever_tendencia
from components.charts import (
    grafico_candlestick,
    grafico_indicadores_tecnicos,
    grafico_retorno_acumulado,
    grafico_retorno_vs_benchmark,
    grafico_rsi,
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
    except (InvalidTickerError, Exception):
        return None


@st.cache_data(ttl=3600)
def selic_atual() -> float:
    return get_selic_atual()


@st.cache_resource(ttl=3600)
def executar_ml(ticker: str) -> dict | None:
    """Treina o modelo com 2 anos de dados e retorna resultados prontos para exibição."""
    try:
        df2y = get_stock_data(ticker, period="2y")
        modelo, acuracia_cv, acuracia_std, relatorio = treinar_modelo(df2y)
        tendencia, prob = prever_tendencia(df2y, modelo)
        importancias = sorted(
            zip(_FEATURES, modelo.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        return {
            "tendencia": tendencia,
            "prob": prob,
            "acuracia_cv": acuracia_cv,
            "acuracia_std": acuracia_std,
            "importancias": importancias,
        }
    except Exception:
        return None


@st.cache_data(ttl=3600)
def calcular_metricas(df: pd.DataFrame, selic: float) -> dict:
    """Calcula todas as métricas de uma vez para evitar recomputação entre abas."""
    return {
        "retorno": calcular_retorno_acumulado(df),
        "volatilidade": calcular_volatilidade_anual(df),
        "drawdown": calcular_drawdown_maximo(df),
        "sharpe": calcular_sharpe(df, risk_free=selic),
    }


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

df_ibovespa = carregar_dados("^BVSP", period)

if erros:
    st.warning(f"Tickers nao encontrados e ignorados: {', '.join(erros)}")

if not dfs:
    st.error("Nenhum ticker valido. Verifique os tickers na sidebar.")
    st.stop()

valid_tickers = list(dfs.keys())
valid_dfs = list(dfs.values())

# ── Abas ─────────────────────────────────────────────────────────────────────

aba_geral, aba_comp, aba_det, aba_ml = st.tabs(
    ["Visao Geral", "Comparativo", "Detalhes", "Machine Learning"]
)

# ── Aba 1: Visao Geral ────────────────────────────────────────────────────────

with aba_geral:
    primeiro = valid_tickers[0]
    df0 = dfs[primeiro]
    m0 = calcular_metricas(df0, selic)

    st.subheader(primeiro)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno Acumulado", f"{m0['retorno'] * 100:+.2f}%")
    c2.metric("Volatilidade Anual", f"{m0['volatilidade'] * 100:.2f}%")
    c3.metric("Drawdown Maximo", f"{m0['drawdown'] * 100:.2f}%")
    c4.metric(
        "Sharpe",
        f"{m0['sharpe']:.4f}",
        help=f"Calculado com Selic de {selic * 100:.2f}% a.a.",
    )

    st.markdown("#### Preço")
    st.plotly_chart(
        grafico_candlestick(df0, primeiro),
        use_container_width=True,
    )

    st.markdown("#### Indicadores Tecnicos")
    st.plotly_chart(
        grafico_indicadores_tecnicos(df0, primeiro),
        use_container_width=True,
    )

    st.markdown("#### RSI (14)")
    st.plotly_chart(
        grafico_rsi(df0, primeiro),
        use_container_width=True,
    )

# ── Aba 2: Comparativo ────────────────────────────────────────────────────────

with aba_comp:
    st.plotly_chart(
        grafico_retorno_acumulado(valid_dfs, valid_tickers),
        use_container_width=True,
    )

    if df_ibovespa is not None:
        st.plotly_chart(
            grafico_retorno_vs_benchmark(valid_dfs, valid_tickers, df_ibovespa),
            use_container_width=True,
        )
    else:
        st.warning("Nao foi possivel carregar o Ibovespa (^BVSP). Grafico de benchmark indisponivel.")

    if len(valid_tickers) >= 2:
        st.plotly_chart(
            heatmap_correlacao(valid_dfs, valid_tickers),
            use_container_width=True,
        )
    else:
        st.info("Adicione pelo menos 2 tickers para exibir o heatmap de correlacao.")

# ── Aba 3: Detalhes ──────────────────────────────────────────────────────────


with aba_det:
    rows = []
    for ticker, df in dfs.items():
        m = calcular_metricas(df, selic)
        rows.append({
            "Ticker": ticker,
            "Retorno Acumulado (%)": round(m["retorno"] * 100, 2),
            "Volatilidade Anual (%)": round(m["volatilidade"] * 100, 2),
            "Drawdown Maximo (%)": round(m["drawdown"] * 100, 2),
            "Sharpe": round(m["sharpe"], 4),
            "Ultimo Close (R$)": round(float(df["Close"].iloc[-1]), 2),
            "Vol. Medio Diario": int(df["Volume"].mean()),
            "Pregões": len(df),
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

# ── Aba 4: Machine Learning ───────────────────────────────────────────────────

with aba_ml:
    primeiro = valid_tickers[0]

    st.subheader(f"Previsão de Tendencia — Random Forest")
    st.caption(f"Modelo treinado com 2 anos de dados de {primeiro}")

    with st.spinner(f"Treinando modelo para {primeiro}..."):
        resultado_ml = executar_ml(primeiro)

    if resultado_ml is None:
        st.error("Nao foi possivel treinar o modelo. Verifique se ha dados suficientes.")
    else:
        tendencia = resultado_ml["tendencia"]
        prob = resultado_ml["prob"]
        acuracia_cv = resultado_ml["acuracia_cv"]
        acuracia_std = resultado_ml["acuracia_std"]
        importancias = resultado_ml["importancias"]

        cor_tendencia = "#26a69a" if tendencia == "Alta" else "#ef5350"
        seta = "↑" if tendencia == "Alta" else "↓"

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Tendencia (proximos 5 pregões)",
            f"{seta} {tendencia}",
            help="Direcao prevista pelo RandomForest com base nas features tecnicas.",
        )
        c2.metric(
            "Probabilidade",
            f"{prob * 100:.1f}%",
            help="Confianca da previsão: media dos votos das 200 arvores.",
        )
        c3.metric(
            "Acuracia CV (TimeSeriesSplit)",
            f"{acuracia_cv * 100:.1f}% ± {acuracia_std * 100:.1f}%",
            help="Media e desvio padrao da acuracia em 5 folds temporais. "
                 "Avaliacao sem data leakage.",
        )

        st.markdown("#### Importancia das Features")

        feats = [f for f, _ in importancias]
        imps = [i for _, i in importancias]
        cores = [
            "#64b5f6" if imp >= sorted(imps, reverse=True)[2] else "#546e7a"
            for imp in imps
        ]

        fig_imp = go.Figure(go.Bar(
            x=imps[::-1],
            y=feats[::-1],
            orientation="h",
            marker_color=cores[::-1],
            text=[f"{v:.1%}" for v in imps[::-1]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            height=420,
            xaxis=dict(title="Importancia", tickformat=".0%"),
            yaxis=dict(title=""),
            margin=dict(l=10, r=60, t=20, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.warning(
            "Previsão para fins educacionais. Nao constitui recomendacao de investimento.",
            icon="⚠️",
        )
