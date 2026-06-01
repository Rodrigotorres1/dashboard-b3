import sqlite3
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "cache.db"
_CACHE_TTL = timedelta(hours=1)

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS cotacoes (
        ticker            TEXT NOT NULL,
        period            TEXT NOT NULL,
        date              TEXT NOT NULL,
        open              REAL,
        high              REAL,
        low               REAL,
        close             REAL NOT NULL,
        volume            REAL NOT NULL,
        return            REAL,
        cumulative_return REAL,
        updated_at        TEXT NOT NULL,
        PRIMARY KEY (ticker, period, date)
    )
"""

_CREATE_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_ticker_period_updated
    ON cotacoes (ticker, period, updated_at)
"""


def init_db() -> None:
    """
    Cria o banco SQLite e a tabela 'cotacoes' se ainda não existirem.

    Seguro para chamar múltiplas vezes — usa CREATE IF NOT EXISTS.
    O banco é criado em data/cache.db relativo ao próprio módulo.
    """
    with sqlite3.connect(_DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)


def salvar_dados(ticker: str, period: str, df: pd.DataFrame) -> None:
    """
    Persiste o DataFrame no banco, sobrescrevendo linhas existentes para o mesmo
    (ticker, period, date) via INSERT OR REPLACE.

    Os campos open, high e low são armazenados como NULL porque get_stock_data
    retorna apenas Close e Volume. O updated_at é marcado como UTC agora.

    Args:
        ticker: Código do ativo (ex: 'PETR4.SA').
        period: Período da consulta (ex: '1y').
        df: DataFrame retornado por get_stock_data.
    """
    init_db()
    updated_at = datetime.now(timezone.utc).isoformat()

    rows = [
        (
            ticker,
            period,
            str(date.date()),
            None,
            None,
            None,
            float(row["Close"]),
            float(row["Volume"]),
            float(row["Return"]) if pd.notna(row["Return"]) else None,
            float(row["Cumulative_Return"]) if pd.notna(row["Cumulative_Return"]) else None,
            updated_at,
        )
        for date, row in df.iterrows()
    ]

    with sqlite3.connect(_DB_PATH) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO cotacoes
                (ticker, period, date, open, high, low, close, volume,
                 return, cumulative_return, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def carregar_cache(ticker: str, period: str) -> pd.DataFrame | None:
    """
    Retorna o DataFrame cacheado se os dados tiverem menos de 1 hora.

    Verifica o updated_at mais recente para o par (ticker, period). Se o cache
    estiver dentro do TTL, reconstrói e retorna o DataFrame no mesmo formato
    que get_stock_data. Caso contrário retorna None.

    Args:
        ticker: Código do ativo.
        period: Período da consulta.

    Returns:
        DataFrame com colunas Close, Volume, Return, Cumulative_Return indexado
        por Date, ou None se o cache estiver ausente ou expirado.
    """
    init_db()

    with sqlite3.connect(_DB_PATH) as conn:
        row = conn.execute(
            "SELECT MAX(updated_at) FROM cotacoes WHERE ticker = ? AND period = ?",
            (ticker, period),
        ).fetchone()

        if row[0] is None:
            return None

        updated_at = datetime.fromisoformat(row[0])
        if datetime.now(timezone.utc) - updated_at > _CACHE_TTL:
            return None

        rows = conn.execute(
            """
            SELECT date, close, volume, return, cumulative_return
            FROM cotacoes
            WHERE ticker = ? AND period = ?
            ORDER BY date ASC
            """,
            (ticker, period),
        ).fetchall()

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Date", "Close", "Volume", "Return", "Cumulative_Return"])
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[s]")
    df["Volume"] = df["Volume"].astype("int64")
    df = df.set_index("Date")
    df.index.name = "Date"
    return df
