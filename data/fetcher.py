import yfinance as yf
import pandas as pd

from data.cache import carregar_cache, salvar_dados


class InvalidTickerError(Exception):
    """Raised when a ticker returns no data from yfinance."""


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """Baixa dados brutos do yfinance e retorna DataFrame normalizado."""
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        raise InvalidTickerError(
            f"Ticker '{ticker}' retornou dados vazios. Verifique se o ticker é válido."
        )

    df = raw[["Close", "Volume"]].copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns.name = None

    df["Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Return"]).cumprod() - 1
    df.index.name = "Date"
    return df


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Retorna dados históricos do ativo, usando cache SQLite quando disponível.

    Tenta primeiro o cache local (data/cache.db). Se o cache estiver ausente
    ou com mais de 1 hora, busca do Yahoo Finance e atualiza o cache.

    Args:
        ticker: Stock ticker with .SA suffix (e.g. 'PETR4.SA').
        period: yfinance period string ('1mo', '3mo', '6mo', '1y', '2y', '5y').

    Returns:
        DataFrame with columns: Close, Volume, Return, Cumulative_Return.

    Raises:
        InvalidTickerError: If ticker returns no data or is invalid.
    """
    cached = carregar_cache(ticker, period)
    if cached is not None:
        return cached

    df = _fetch_yfinance(ticker, period)
    salvar_dados(ticker, period, df)
    return df


if __name__ == "__main__":
    tickers = ["WEGE3.SA", "PETR4.SA", "GGRC11.SA"]

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"{'='*60}")
        try:
            df = get_stock_data(ticker, period="1y")
            print(f"Periodo:       {df.index[0].date()} - {df.index[-1].date()}")
            print(f"Registros:     {len(df)}")
            print(f"Ultimo Close:  R$ {df['Close'].iloc[-1]:.2f}")
            print(f"Retorno total: {df['Cumulative_Return'].iloc[-1]*100:.2f}%")
            print(f"Vol. medio:    {int(df['Volume'].mean()):,}")
            print("\nÚltimas 5 linhas:")
            print(df.tail().to_string())
        except InvalidTickerError as e:
            print(f"ERRO: {e}")

    # Test invalid ticker
    print(f"\n{'='*60}")
    print("Ticker inválido: XYZABC99.SA")
    print(f"{'='*60}")
    try:
        get_stock_data("XYZABC99.SA")
    except InvalidTickerError as e:
        print(f"ERRO capturado: {e}")
