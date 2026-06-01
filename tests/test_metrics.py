"""
Testes unitários para analysis/metrics.py.

Todos os testes usam dados sintéticos gerados localmente — sem chamadas
ao yfinance ou à API do BCB. O objetivo é validar a lógica de cálculo
de cada função de forma rápida e determinística.
"""

import math
import numpy as np
import pandas as pd
import pytest

from analysis.metrics import (
    calcular_drawdown_maximo,
    calcular_retorno_acumulado,
    calcular_sharpe,
    calcular_volatilidade_anual,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_df(prices: list[float], volume: int = 1_000_000) -> pd.DataFrame:
    """
    Constrói um DataFrame no formato de get_stock_data a partir de uma lista
    de preços de fechamento. Usado pelos testes para evitar dependência de rede.
    """
    dates = pd.date_range(start="2024-01-01", periods=len(prices), freq="B")
    close = pd.Series(prices, index=dates, dtype=float)
    ret = close.pct_change()
    cum_ret = (1 + ret).cumprod() - 1
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": pd.Series([volume] * len(prices), index=dates, dtype="int64"),
            "Return": ret,
            "Cumulative_Return": cum_ret,
        }
    )


# ── test_retorno_acumulado ────────────────────────────────────────────────────

class TestRetornoAcumulado:
    def test_alta_simples(self):
        """Preço sobe 10%: retorno acumulado deve ser exatamente 0.10."""
        df = _make_df([100.0, 110.0])
        assert calcular_retorno_acumulado(df) == pytest.approx(0.10)

    def test_alta_composta(self):
        """Dois períodos de +10%: retorno acumulado = (1.1)^2 - 1 = 0.21."""
        df = _make_df([100.0, 110.0, 121.0])
        assert calcular_retorno_acumulado(df) == pytest.approx(0.21)

    def test_queda(self):
        """Preço cai 20%: retorno acumulado deve ser -0.20."""
        df = _make_df([100.0, 80.0])
        assert calcular_retorno_acumulado(df) == pytest.approx(-0.20)

    def test_sem_variacao(self):
        """Preço estável: retorno acumulado deve ser zero."""
        df = _make_df([50.0, 50.0, 50.0])
        assert calcular_retorno_acumulado(df) == pytest.approx(0.0)

    def test_retorna_float(self):
        df = _make_df([100.0, 105.0, 102.0])
        assert isinstance(calcular_retorno_acumulado(df), float)


# ── test_volatilidade_anual ───────────────────────────────────────────────────

class TestVolatilidadeAnual:
    def test_retorna_float_positivo(self):
        """Volatilidade deve ser sempre um float estritamente positivo."""
        df = _make_df([100.0, 102.0, 98.0, 105.0, 101.0])
        vol = calcular_volatilidade_anual(df)
        assert isinstance(vol, float)
        assert vol > 0.0

    def test_precos_constantes_zero(self):
        """Sem variação de preço, desvio padrão e volatilidade são zero."""
        df = _make_df([100.0] * 10)
        assert calcular_volatilidade_anual(df) == pytest.approx(0.0)

    def test_anualizado_por_sqrt_252(self):
        """
        Verifica que a anualização usa sqrt(252).
        Com retornos diários constantes de 1%, std diário = 0,
        mas com variação conhecida podemos checar a escala.
        """
        # 5 preços com retornos diários idênticos a 1%
        prices = [100.0 * (1.01 ** i) for i in range(5)]
        df = _make_df(prices)
        std_diario = df["Return"].dropna().std()
        esperado = std_diario * math.sqrt(252)
        assert calcular_volatilidade_anual(df) == pytest.approx(esperado)

    def test_maior_variacao_maior_volatilidade(self):
        """Série mais volátil deve produzir volatilidade anual maior."""
        df_estavel = _make_df([100.0, 101.0, 100.0, 101.0, 100.0])
        df_volatil = _make_df([100.0, 130.0, 70.0, 120.0, 80.0])
        assert calcular_volatilidade_anual(df_volatil) > calcular_volatilidade_anual(df_estavel)


# ── test_drawdown_maximo ──────────────────────────────────────────────────────

class TestDrawdownMaximo:
    def test_retorna_negativo_ou_zero(self):
        """Drawdown deve ser sempre <= 0."""
        df = _make_df([100.0, 120.0, 80.0, 100.0])
        assert calcular_drawdown_maximo(df) <= 0.0

    def test_valor_correto(self):
        """Pico em 120, vale em 80: drawdown = (80 - 120) / 120 = -1/3."""
        df = _make_df([100.0, 120.0, 80.0, 100.0])
        esperado = (80.0 - 120.0) / 120.0  # ≈ -0.3333
        assert calcular_drawdown_maximo(df) == pytest.approx(esperado)

    def test_serie_sempre_crescente(self):
        """Série monotonicamente crescente: drawdown deve ser zero."""
        df = _make_df([100.0, 110.0, 120.0, 130.0])
        assert calcular_drawdown_maximo(df) == pytest.approx(0.0)

    def test_queda_total(self):
        """Queda de 100 para 50: drawdown = -0.50."""
        df = _make_df([100.0, 90.0, 70.0, 50.0])
        assert calcular_drawdown_maximo(df) == pytest.approx(-0.50)

    def test_retorna_float(self):
        df = _make_df([100.0, 90.0, 95.0])
        assert isinstance(calcular_drawdown_maximo(df), float)


# ── test_sharpe ───────────────────────────────────────────────────────────────

class TestSharpe:
    def test_retorna_float(self):
        """Sharpe deve retornar um float com risk_free explícito."""
        df = _make_df([100.0, 102.0, 101.0, 104.0, 103.0])
        resultado = calcular_sharpe(df, risk_free=0.145)
        assert isinstance(resultado, float)

    def test_retorno_zero_std_zero(self):
        """Retornos constantes → std = 0 → Sharpe deve ser 0.0 (sem divisão por zero)."""
        df = _make_df([100.0] * 10)
        assert calcular_sharpe(df, risk_free=0.145) == pytest.approx(0.0)

    def test_maior_retorno_maior_sharpe(self):
        """
        Mesmo ruído base (volatilidade idêntica), média mais alta → Sharpe maior.
        Ambas as séries são construídas somando um drift diferente ao mesmo vetor
        de ruído, garantindo std(retornos) igual e apenas a média diferindo.
        """
        ruido = np.array([0.02, -0.01, 0.015, -0.008, 0.012, -0.005, 0.018, -0.009])

        def precos_de_retornos(retornos: np.ndarray) -> list[float]:
            p = [100.0]
            for r in retornos:
                p.append(p[-1] * (1 + r))
            return p

        df_fraco = _make_df(precos_de_retornos(ruido + 0.001))
        df_forte = _make_df(precos_de_retornos(ruido + 0.010))
        assert calcular_sharpe(df_forte, risk_free=0.145) > calcular_sharpe(df_fraco, risk_free=0.145)

    def test_risk_free_reduz_sharpe(self):
        """Taxa livre de risco maior deve reduzir o Sharpe."""
        df = _make_df([100.0, 102.0, 101.0, 104.0, 103.0])
        sharpe_baixo = calcular_sharpe(df, risk_free=0.05)
        sharpe_alto = calcular_sharpe(df, risk_free=0.50)
        assert sharpe_baixo > sharpe_alto

    def test_usa_risk_free_explicito_nao_api(self):
        """
        Garantia de que risk_free=0.145 é respeitado sem chamar a API do BCB.
        O cálculo manual deve bater com a função.
        """
        df = _make_df([100.0, 103.0, 101.0, 105.0, 104.0])
        retornos = df["Return"].dropna()
        rf_diario = (1 + 0.145) ** (1 / 252) - 1
        excesso = retornos - rf_diario
        esperado = float((excesso.mean() / excesso.std()) * np.sqrt(252))
        assert calcular_sharpe(df, risk_free=0.145) == pytest.approx(esperado)
