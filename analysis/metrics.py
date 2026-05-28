import numpy as np
import pandas as pd
import requests
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.fetcher import get_stock_data

_BCB_SELIC_URL = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados/ultimos/1?formato=json"
)
_SELIC_FALLBACK = 0.145


def get_selic_atual() -> float:
    """
    Busca a taxa Selic diária mais recente na API do Banco Central do Brasil
    e a converte para taxa anual efetiva.

    A série SGS 11 retorna a Selic acumulada no dia (% a.d.). A função
    anualiza pelo critério de juros compostos com 252 dias úteis.

    Returns:
        Taxa Selic anual como decimal (ex: 0.1450 = 14,50% a.a.).
        Em caso de falha na requisição ou parsing, retorna o fallback 0.145.
    """
    try:
        response = requests.get(_BCB_SELIC_URL, timeout=5)
        response.raise_for_status()
        selic_diaria = float(response.json()[0]["valor"]) / 100
        return float((1 + selic_diaria) ** 252 - 1)
    except Exception:
        return _SELIC_FALLBACK


def calcular_retorno_acumulado(df: pd.DataFrame) -> float:
    """
    Retorna o retorno acumulado total do período.

    Usa o último valor de Cumulative_Return, que representa
    a variação percentual desde o primeiro pregão até o último.

    Args:
        df: DataFrame retornado por get_stock_data, com coluna Cumulative_Return.

    Returns:
        Retorno acumulado como decimal (0.15 = 15%).
    """
    return float(df["Cumulative_Return"].iloc[-1])


def calcular_volatilidade_anual(df: pd.DataFrame) -> float:
    """
    Retorna a volatilidade anualizada com base nos retornos diários.

    Calcula o desvio padrão dos retornos diários e anualiza
    multiplicando por sqrt(252), número convencional de pregões/ano.

    Args:
        df: DataFrame retornado por get_stock_data, com coluna Return.

    Returns:
        Volatilidade anualizada como decimal (0.30 = 30%).
    """
    retornos = df["Return"].dropna()
    return float(retornos.std() * np.sqrt(252))


def calcular_drawdown_maximo(df: pd.DataFrame) -> float:
    """
    Retorna o maximum drawdown do período.

    Mede a maior queda percentual do pico até o vale ao longo
    da série histórica de preços de fechamento.

    Args:
        df: DataFrame retornado por get_stock_data, com coluna Close.

    Returns:
        Maximum drawdown como decimal negativo (-0.25 = -25%).
    """
    close = df["Close"]
    pico_acumulado = close.cummax()
    drawdown = (close - pico_acumulado) / pico_acumulado
    return float(drawdown.min())


def calcular_sharpe(df: pd.DataFrame, risk_free: float | None = None) -> float:
    """
    Retorna o Índice de Sharpe anualizado.

    Mede o retorno excedente por unidade de risco. Busca a Selic atual
    via API do Banco Central; usa 14,50% a.a. como fallback se a API falhar.

    Args:
        df: DataFrame retornado por get_stock_data, com coluna Return.
        risk_free: Taxa livre de risco anual (decimal). Se None, busca
                   automaticamente via get_selic_atual().

    Returns:
        Índice de Sharpe. Valores > 1 são considerados bons.
    """
    if risk_free is None:
        risk_free = get_selic_atual()
    retornos = df["Return"].dropna()
    risk_free_diario = (1 + risk_free) ** (1 / 252) - 1
    excesso = retornos - risk_free_diario
    if excesso.std() == 0:
        return 0.0
    return float((excesso.mean() / excesso.std()) * np.sqrt(252))


if __name__ == "__main__":
    ticker = "PETR4.SA"

    print("Buscando Selic atual na API do Banco Central...")
    selic = get_selic_atual()
    origem = "API BCB" if selic != _SELIC_FALLBACK else "fallback"
    print(f"Selic anual ({origem}): {selic*100:.4f}% a.a.")

    print(f"\nCarregando dados de {ticker}...")
    df = get_stock_data(ticker, period="1y")

    retorno = calcular_retorno_acumulado(df)
    volatilidade = calcular_volatilidade_anual(df)
    drawdown = calcular_drawdown_maximo(df)
    sharpe = calcular_sharpe(df, risk_free=selic)

    print(f"\n{'='*45}")
    print(f"  Metricas — {ticker} (ultimo ano)")
    print(f"{'='*45}")
    print(f"  Retorno acumulado  : {retorno*100:+.2f}%")
    print(f"  Volatilidade anual : {volatilidade*100:.2f}%")
    print(f"  Drawdown maximo    : {drawdown*100:.2f}%")
    print(f"  Sharpe (Selic {selic*100:.2f}%): {sharpe:.4f}")
    print(f"{'='*45}")
