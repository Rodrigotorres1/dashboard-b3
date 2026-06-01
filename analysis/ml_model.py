import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, train_test_split


_FEATURES = [
    "rsi_14",
    "sma_20",
    "sma_50",
    "bb_upper",
    "bb_lower",
    "return_1d",
    "return_5d",
    "return_10d",
    "volume_zscore",
    "volatilidade_5d",
    "distancia_sma20",
]


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói a tabela de features e o target a partir do DataFrame do fetcher.

    Indicadores calculados:
    - RSI 14 (Wilder EWM)
    - SMA 20 e SMA 50
    - Bollinger superior e inferior (SMA20 ± 2σ)
    - Retorno de 1, 5 e 10 dias
    - Volume normalizado por z-score (média e desvio do período inteiro)
    - volatilidade_5d: desvio padrão dos retornos dos últimos 5 dias
    - distancia_sma20: distância percentual do preço em relação à SMA 20
    - Target: 1 se Close(t+5) > Close(t), 0 caso contrário

    As primeiras 50 linhas e as últimas 5 são descartadas porque contêm
    NaN obrigatório de indicadores e do target prospectivo, respectivamente.

    Args:
        df: DataFrame retornado por get_stock_data com colunas Close e Volume.

    Returns:
        DataFrame com as colunas em _FEATURES mais a coluna 'target',
        sem nenhum NaN, indexado por data.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    volume = df["Volume"]

    # RSI 14 — método de Wilder
    delta = close.diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.ewm(com=13, min_periods=14).mean()
    media_perda = perda.ewm(com=13, min_periods=14).mean()
    rs = media_ganho / media_perda
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Médias móveis e Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["sma_20"] = sma20
    out["sma_50"] = close.rolling(50).mean()
    out["bb_upper"] = sma20 + 2 * std20
    out["bb_lower"] = sma20 - 2 * std20

    # Retornos passados
    retorno_diario = close.pct_change(1)
    out["return_1d"] = retorno_diario
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    # Volume z-score (normalizado sobre toda a série)
    out["volume_zscore"] = (volume - volume.mean()) / volume.std()

    # Volatilidade realizada dos últimos 5 dias
    out["volatilidade_5d"] = retorno_diario.rolling(5).std()

    # Distância percentual do preço para a SMA20 — captura mean reversion
    out["distancia_sma20"] = (close - sma20) / sma20

    # Target: 1 se subiu após 5 pregões, 0 se caiu ou ficou igual
    out["target"] = (close.shift(-5) > close).astype(int)

    out = out.dropna()

    return out


def treinar_modelo(
    df: pd.DataFrame,
) -> tuple[RandomForestClassifier, float, float, str]:
    """
    Treina um RandomForestClassifier e avalia com StratifiedKFold de 5 folds.

    O modelo usa class_weight='balanced' para corrigir o viés gerado por
    desbalanceamento entre classes Alta/Baixa. A avaliação por TimeSeriesSplit
    respeita a ordem cronológica dos dados — cada fold treina apenas em dados
    anteriores ao período de validação, eliminando o data leakage temporal.

    O modelo final é retreinado sobre 70% dos dados (split temporal) para que
    prever_tendencia() use o modelo com os dados mais recentes no teste.

    Args:
        df: DataFrame retornado por get_stock_data.

    Returns:
        Tupla (modelo, acuracia_cv_media, relatorio_classificacao) onde:
        - modelo: RandomForestClassifier ajustado sobre os dados de treino.
        - acuracia_cv_media: média das acurácias nos 5 folds de validação.
        - acuracia_cv_std: desvio padrão das acurácias entre os folds.
        - relatorio_classificacao: classification_report no split de teste temporal.

    Raises:
        ValueError: Se o DataFrame tiver menos de 100 linhas após preparar features.
    """
    features_df = preparar_features(df)

    if len(features_df) < 100:
        raise ValueError(
            f"Apenas {len(features_df)} amostras após preparar features. "
            "Use um período maior (mínimo recomendado: 1 ano)."
        )

    X = features_df[_FEATURES].values
    y = features_df["target"].values

    # Avaliação com TimeSeriesSplit — sem data leakage temporal
    tscv = TimeSeriesSplit(n_splits=5)
    acuracias_cv = []

    for treino_idx, val_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[treino_idx], y[treino_idx])
        acuracias_cv.append(accuracy_score(y[val_idx], clf.predict(X[val_idx])))

    acuracia_cv_media = float(np.mean(acuracias_cv))
    acuracia_cv_std = float(np.std(acuracias_cv))

    # Modelo final treinado no split temporal 70/30
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.30, shuffle=False
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X_treino, y_treino)

    y_pred = modelo.predict(X_teste)
    relatorio = classification_report(y_teste, y_pred, target_names=["Baixa", "Alta"])

    return modelo, acuracia_cv_media, acuracia_cv_std, relatorio


def prever_tendencia(
    df: pd.DataFrame,
    modelo: RandomForestClassifier,
) -> tuple[str, float]:
    """
    Prevê a tendência para os próximos 5 pregões com base no último ponto da série.

    Usa a linha mais recente com todas as features válidas para fazer a predição.
    O target da última linha é descartado (é prospectivo por definição).

    Args:
        df: DataFrame retornado por get_stock_data, com dados recentes.
        modelo: RandomForestClassifier retornado por treinar_modelo.

    Returns:
        Tupla (tendencia, probabilidade) onde:
        - tendencia: "Alta" ou "Baixa".
        - probabilidade: confiança da previsão entre 0.0 e 1.0.
    """
    features_df = preparar_features(df)
    ultima_linha = features_df[_FEATURES].iloc[[-1]].values

    classe = int(modelo.predict(ultima_linha)[0])
    probabilidades = modelo.predict_proba(ultima_linha)[0]
    prob = float(probabilidades[classe])

    tendencia = "Alta" if classe == 1 else "Baixa"
    return tendencia, prob


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.fetcher import get_stock_data

    ticker = "PETR4.SA"
    print(f"Carregando dados de {ticker}...")
    df = get_stock_data(ticker, period="2y")

    print("\n--- Features ---")
    features_df = preparar_features(df)
    print(f"Amostras totais    : {len(features_df)}")
    print(f"Features           : {_FEATURES}")
    baixa = int((features_df["target"] == 0).sum())
    alta = int((features_df["target"] == 1).sum())
    print(f"Distribuicao target: Baixa={baixa}  Alta={alta}")
    print("\nUltimas 3 linhas:")
    print(features_df.tail(3).to_string())

    print("\n--- Treinamento (TimeSeriesSplit 5 folds) ---")
    modelo, acuracia_cv, acuracia_std, relatorio = treinar_modelo(df)
    print(f"Acuracia CV media  : {acuracia_cv*100:.2f}%  (+/- {acuracia_std*100:.2f}%)")
    print(f"\nRelatorio de classificacao (split temporal 70/30):\n{relatorio}")

    print("--- Previsao ---")
    tendencia, prob = prever_tendencia(df, modelo)
    print(f"Tendencia (proximos 5 pregoes): {tendencia}")
    print(f"Probabilidade                 : {prob*100:.1f}%")

    print("\n--- Importancia das features ---")
    importancias = sorted(
        zip(_FEATURES, modelo.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for feat, imp in importancias:
        barra = "#" * int(imp * 40)
        print(f"  {feat:<20} {imp:.4f}  {barra}")
