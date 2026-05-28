import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.fetcher import get_stock_data


def grafico_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Retorna candlestick de fechamento com volume como subgráfico inferior.

    Como get_stock_data retorna apenas Close e Volume (sem OHLC completo),
    o candlestick é construído com Open=Close_anterior, High=max(O,C)*1.001,
    Low=min(O,C)*0.999, dando aparência visual correta do movimento diário.

    Args:
        df: DataFrame retornado por get_stock_data, com colunas Close e Volume.
        ticker: Nome do ativo exibido no título e no eixo.

    Returns:
        Figure com dois painéis: candlestick (70%) e volume (30%).
    """
    close = df["Close"]
    open_ = close.shift(1).fillna(close)
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999

    colors_vol = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(close, open_)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=open_,
            high=high,
            low=low,
            close=close,
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=colors_vol,
            showlegend=False,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Preco e Volume",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
    )
    fig.update_yaxes(title_text="Preco (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def grafico_retorno_acumulado(dfs: list[pd.DataFrame], tickers: list[str]) -> go.Figure:
    """
    Retorna linha de retorno acumulado para múltiplos ativos no mesmo painel.

    Permite comparar visualmente a evolução percentual de cada ativo
    a partir de uma base zero no início do período.

    Args:
        dfs: Lista de DataFrames retornados por get_stock_data.
        tickers: Lista de nomes correspondentes a cada DataFrame.

    Returns:
        Figure com uma linha por ativo e formatação de eixo em porcentagem.
    """
    fig = go.Figure()

    for df, ticker in zip(dfs, tickers):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cumulative_Return"] * 100,
                mode="lines",
                name=ticker,
                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}%<extra>" + ticker + "</extra>",
            )
        )

    fig.update_layout(
        title="Retorno Acumulado Comparativo",
        yaxis_title="Retorno Acumulado (%)",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

    return fig


def heatmap_correlacao(dfs: list[pd.DataFrame], tickers: list[str]) -> go.Figure:
    """
    Retorna heatmap de correlação de Pearson entre os retornos diários dos ativos.

    Correlação calculada sobre os retornos (não preços) para medir
    co-movimento genuíno entre os ativos, independente de tendência.

    Args:
        dfs: Lista de DataFrames retornados por get_stock_data.
        tickers: Lista de nomes correspondentes a cada DataFrame.

    Returns:
        Figure com heatmap simétrico de correlações no intervalo [-1, 1].
    """
    returns = pd.DataFrame(
        {ticker: df["Return"] for ticker, df in zip(tickers, dfs)}
    )
    corr = returns.corr()

    text = [[f"{v:.2f}" for v in row] for row in corr.values]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=tickers,
            y=tickers,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlacao"),
        )
    )

    fig.update_layout(
        title="Correlacao de Retornos Diarios",
        template="plotly_dark",
        height=450,
        width=500,
    )

    return fig


if __name__ == "__main__":
    tickers = ["PETR4.SA", "WEGE3.SA", "GGRC11.SA"]

    print("Carregando dados...")
    dfs = [get_stock_data(t, period="1y") for t in tickers]

    print("\nTestando grafico_candlestick...")
    for df, ticker in zip(dfs, tickers):
        fig = grafico_candlestick(df, ticker)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2
        print(f"  {ticker}: OK — {len(fig.data)} traces (candlestick + volume)")

    print("\nTestando grafico_retorno_acumulado...")
    fig = grafico_retorno_acumulado(dfs, tickers)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(tickers)
    print(f"  OK — {len(fig.data)} linhas ({', '.join(tickers)})")

    print("\nTestando heatmap_correlacao...")
    fig = heatmap_correlacao(dfs, tickers)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].z.shape == (len(tickers), len(tickers))
    import numpy as np
    corr_matrix = np.array(fig.data[0].z)
    print(f"  OK — matriz {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
    print("\n  Correlacoes:")
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j > i:
                print(f"    {t1} x {t2}: {corr_matrix[i][j]:.4f}")

    print("\nTodos os graficos gerados com sucesso.")
