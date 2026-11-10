# ==========================================================
# FACTOR INVESTING NO MERCADO ACIONARIO BRASILEIRO
# Carteira multifatorial vs Ibovespa e CDI
#
# Etapas:
# 1. Baixa volatilidade
# 2. Valor (P/VP + ROE)
# 3. Qualidade (margem líquida)
# 4. Momento (retorno de 12 meses)
# 5. Carteira multifatorial e comparação com Ibovespa e CDI
#
# Instalação:
# pip install yfinance pandas numpy requests matplotlib
#
# Execute:
# python factor_investing_completo.py
# ==========================================================

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# 0. CONFIGURAÇÕES
# ----------------------------------------------------------

DATA_INICIO_FORMACAO = "2023-03-28"
DATA_INICIO_CARTEIRA = "2024-03-28"
DATA_FIM_CARTEIRA = "2026-07-02"  # Yahoo usa data final exclusiva
MAX_GAP_DIAS = 7

ACOES_UNIVERSO = [
    "ABEV3", "ALOS3", "ALPA4", "ARZZ3", "ASAI3", "AURE3", "AZUL4", "B3SA3",
    "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BEEF3", "BHIA3", "BPAC11", "BRKM5",
    "CCRO3", "CIEL3", "CMIG4", "CMIN3", "COGN3", "CRFB3", "CSAN3", "CSNA3",
    "CVCB3", "CYRE3", "DXCO3", "ELET3", "ELET6", "EMBR3", "ENEV3", "ENGI11",
    "EQTL3", "EZTC3", "FLRY3", "GOAU4", "HAPV3", "HYPE3", "IGTI11", "IRBR3",
    "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LREN3", "LWSA3", "MGLU3", "MRFG3",
    "MRVE3", "MULT3", "NTCO3", "PCAR3", "PETR3", "PETR4", "PETZ3", "PRIO3",
    "RADL3", "RAIL3", "RAIZ4", "RDOR3", "RECV3", "RENT3", "RRRP3", "SANB11",
    "SBSP3", "SLCE3", "SMTO3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "TRPL4",
    "UGPA3", "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVA3", "WEGE3", "YDUQ3",
]

CODIGOS_ATUAIS = {
    "RRRP3": "BRAV3", "CCRO3": "MOTV3", "ELET3": "AXIA3", "ELET6": "AXIA6",
    "TRPL4": "ISAE4", "EMBR3": "EMBJ3", "ARZZ3": "AZZA3", "MRFG3": "MBRF3",
}

PASTA_CACHE = "cache_factor_investing"
os.makedirs(PASTA_CACHE, exist_ok=True)


def ticker_atual(ticker):
    return CODIGOS_ATUAIS.get(ticker, ticker)


def normalizar_indice(df):
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def baixar_precos(tickers, inicio, fim, arquivo):
    """Baixa preços ajustados e guarda cache local."""
    tickers = list(dict.fromkeys(tickers))
    caminho = os.path.join(PASTA_CACHE, arquivo)

    if os.path.exists(caminho):
        try:
            cache = normalizar_indice(
                pd.read_csv(caminho, index_col=0, parse_dates=True)
            )
            if set(tickers).issubset(cache.columns):
                return cache[tickers]
        except Exception:
            print(f"[AVISO] Cache {arquivo} inválido; baixando novamente.")

    dados = yf.download(
        tickers, start=inicio, end=fim, auto_adjust=True,
        progress=False, threads=False
    )
    if dados.empty:
        raise ValueError(f"Não foi possível baixar os preços: {arquivo}")

    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" in dados.columns.get_level_values(0):
            precos = dados["Close"]
        else:
            precos = dados.xs("Close", axis=1, level=1)
    else:
        precos = dados[["Close"]].copy()
        precos.columns = [tickers[0]]

    if isinstance(precos, pd.Series):
        precos = precos.to_frame(name=tickers[0])

    precos = normalizar_indice(precos).reindex(columns=tickers)
    precos.to_csv(caminho)
    return precos


# ----------------------------------------------------------
# 1. BAIXA VOLATILIDADE
# ----------------------------------------------------------

def ranking_baixa_volatilidade():
    print("\n" + "=" * 60)
    print("FATOR 1 - BAIXA VOLATILIDADE")
    print("=" * 60)

    tickers = [ticker_atual(t) + ".SA" for t in ACOES_UNIVERSO]
    precos = baixar_precos(
        tickers, DATA_INICIO_FORMACAO, "2024-03-29",
        "precos_baixa_volatilidade.csv"
    )
    precos = precos.loc[:, precos.notna().sum() >= 200]

    volatilidade = (
        precos.pct_change(fill_method=None).std() * np.sqrt(252) * 100
    )
    ranking = volatilidade.dropna().sort_values().rename_axis("Ticker").reset_index()
    ranking["Ticker"] = ranking["Ticker"].str.replace(".SA", "", regex=False)
    ranking = ranking.rename(columns={0: "Volatilidade anual (%)"})
    ranking["Volatilidade anual (%)"] = ranking[
        "Volatilidade anual (%)"
    ].round(2)

    print(f"Ações analisadas: {len(ranking)} de {len(tickers)}")
    print(ranking.head(10).to_string(index=False))
    return ranking


# ----------------------------------------------------------
# 2. VALOR (P/VP + ROE)
# ----------------------------------------------------------

UNIVERSO_VALOR = [
    "ABEV3", "ALOS3", "ALPA4", "ARZZ3", "ASAI3", "AURE3", "AZUL4", "B3SA3",
    "BBDC3", "BBDC4", "BBSE3", "BEEF3", "BHIA3", "BPAC11", "BRAP4", "BRKM5",
    "CCRO3", "CIEL3", "CMIG4", "CMIN3", "COGN3", "CRFB3", "CSAN3", "CSNA3",
    "CVCB3", "CYRE3", "DXCO3", "ELET3", "ELET6", "EMBR3", "ENEV3", "ENGI11",
    "EQTL3", "EZTC3", "FLRY3", "GOAU4", "HAPV3", "HYPE3", "IGTI11", "IRBR3",
    "ITSA4", "ITUB4", "JBSS3", "KLBN11", "LREN3", "LWSA3", "MGLU3", "MRFG3",
    "MRVE3", "MULT3", "NTCO3", "PCAR3", "PETR3", "PETR4", "PETZ3", "PRIO3",
    "RADL3", "RAIL3", "RAIZ4", "RDOR3", "RECV3", "RENT3", "RRRP3", "SANB11",
    "SBSP3", "SLCE3", "SMTO3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "TRPL4",
    "UGPA3", "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVA3", "WEGE3", "YDUQ3",
]


def ranking_valor():
    print("\n" + "=" * 60)
    print("FATOR 2 - VALOR (P/VP + ROE)")
    print("=" * 60)

    tickers = [ticker_atual(t) + ".SA" for t in UNIVERSO_VALOR]
    precos = baixar_precos(
        tickers, "2024-03-20", "2024-04-02", "precos_valor.csv"
    )
    precos_data = precos.loc[:pd.Timestamp("2024-03-28")]
    if precos_data.empty:
        raise ValueError("Sem preços para o cálculo do fator valor.")
    precos_data = precos_data.ffill().iloc[-1]

    linhas = []
    for ticker in tickers:
        try:
            preco = precos_data.get(ticker)
            if pd.isna(preco):
                continue
            info = yf.Ticker(ticker).info
            vpa = info.get("bookValue")
            roe = info.get("returnOnEquity")
            if vpa and roe and vpa > 0 and roe > 0:
                pvp = float(preco) / float(vpa)
                if pvp > 0:
                    linhas.append({
                        "Ticker": ticker.replace(".SA", ""),
                        "P/VP": pvp,
                        "ROE": float(roe)
                    })
        except Exception:
            continue

    df = pd.DataFrame(linhas)
    if df.empty:
        print("[AVISO] Dados insuficientes para o ranking de valor.")
        return df

    df["Rank P/VP"] = df["P/VP"].rank(ascending=True)
    df["Rank ROE"] = df["ROE"].rank(ascending=False)
    df["Pontuação"] = df["Rank P/VP"] + df["Rank ROE"]
    df = df.sort_values("Pontuação").head(20).reset_index(drop=True)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------
# 3. QUALIDADE (MARGEM LÍQUIDA)
# ----------------------------------------------------------

def obter_tickers_b3():
    payload = {
        "language": "pt-br", "pageNumber": 1, "pageSize": 200,
        "index": "IBOV", "segment": "1"
    }
    import base64
    import json

    token = base64.b64encode(json.dumps(payload).encode()).decode()
    url = (
        "https://sistemaswebb3-listados.b3.com.br/indexProxy/"
        f"indexCall/GetPortfolioDay/{token}"
    )
    resposta = requests.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30
    )
    resposta.raise_for_status()
    return sorted({item["cod"].strip() for item in resposta.json()["results"]})


def ranking_qualidade():
    print("\n" + "=" * 60)
    print("FATOR 3 - QUALIDADE (MARGEM LÍQUIDA)")
    print("=" * 60)

    try:
        tickers = obter_tickers_b3()
    except Exception as erro:
        print(f"[AVISO] Consulta à B3 falhou: {erro}")
        tickers = ACOES_UNIVERSO

    inicio = pd.Timestamp("2024-03-28")
    fim = pd.Timestamp("2026-01-01")
    linhas = []

    for ticker in tickers:
        try:
            demonstrativos = yf.Ticker(f"{ticker}.SA").financials
            if demonstrativos.empty:
                continue

            margens = []
            for data in demonstrativos.columns:
                data = pd.to_datetime(data)
                if not inicio <= data <= fim:
                    continue
                if (
                    "Total Revenue" not in demonstrativos.index
                    or "Net Income" not in demonstrativos.index
                ):
                    continue

                receita = demonstrativos.loc["Total Revenue", data]
                lucro = demonstrativos.loc["Net Income", data]
                if pd.notna(receita) and receita > 0 and pd.notna(lucro):
                    margens.append(float(lucro / receita) * 100)

            if margens:
                linhas.append({
                    "Ticker": ticker,
                    "Margem líquida média (%)": np.mean(margens)
                })
        except Exception:
            continue

    df = pd.DataFrame(linhas)
    if df.empty:
        print("[AVISO] Dados insuficientes para o ranking de qualidade.")
        return df

    df = df.sort_values(
        "Margem líquida média (%)", ascending=False
    ).head(20).reset_index(drop=True)
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------
# 4. MOMENTO (RETORNO DE 12 MESES)
# ----------------------------------------------------------

def ranking_momento():
    print("\n" + "=" * 60)
    print("FATOR 4 - MOMENTO")
    print("=" * 60)

    try:
        universo = obter_tickers_b3()
        origem = "composição consultada na B3"
    except Exception as erro:
        print(f"[AVISO] Consulta à B3 falhou: {erro}")
        universo = ACOES_UNIVERSO
        origem = "universo de referência do estudo"

    tickers = [ticker_atual(t) + ".SA" for t in universo]
    inicio = datetime.strptime("28/03/2023", "%d/%m/%Y")
    fim = datetime.strptime("28/03/2024", "%d/%m/%Y")

    dados = baixar_precos(
        tickers,
        inicio.strftime("%Y-%m-%d"),
        (fim + timedelta(days=MAX_GAP_DIAS + 1)).strftime("%Y-%m-%d"),
        "precos_momento.csv"
    )

    linhas = []
    for ticker in dados.columns:
        serie = dados[ticker].dropna()
        inicial = serie[serie.index >= pd.Timestamp(inicio)]
        final = serie[serie.index >= pd.Timestamp(fim)]
        if inicial.empty or final.empty:
            continue

        data_i, data_f = inicial.index[0], final.index[0]
        if (data_i - pd.Timestamp(inicio)).days > MAX_GAP_DIAS:
            continue
        if (data_f - pd.Timestamp(fim)).days > MAX_GAP_DIAS:
            continue

        preco_i, preco_f = float(inicial.iloc[0]), float(final.iloc[0])
        if preco_i > 0:
            linhas.append({
                "Ticker": ticker.replace(".SA", ""),
                "Momento (%)": (preco_f / preco_i - 1) * 100
            })

    df = pd.DataFrame(linhas)
    if df.empty:
        print("[AVISO] Dados insuficientes para o ranking de momento.")
        return df

    df = df.sort_values("Momento (%)", ascending=False).head(20).reset_index(drop=True)
    print(f"Universo: {origem}")
    print(df.to_string(index=False))
    return df


# ----------------------------------------------------------
# 5. CARTEIRA MULTIFATORIAL
# Composição final adotada no documento do trabalho.
# ----------------------------------------------------------

FATORES_CARTEIRA = {
    "Baixa volatilidade": ["TAEE11", "AURE3", "BBSE3", "ITSA4"],
    "Valor": ["PETR4", "CYRE3", "VBBR3", "BBDC3"],
    "Qualidade": ["ITSA4", "CXSE3", "BBSE3", "BRAP4"],
    "Momento": ["CEAB3", "POMO4", "TEND3", "YDUQ3"],
}


def montar_pesos():
    posicoes = [
        ticker + ".SA"
        for lista in FATORES_CARTEIRA.values()
        for ticker in lista
    ]
    pesos = pd.Series(
        1 / len(posicoes), index=posicoes
    ).groupby(level=0).sum()

    print("\n" + "=" * 60)
    print("COMPOSIÇÃO DA CARTEIRA")
    print("=" * 60)
    for fator, lista in FATORES_CARTEIRA.items():
        print(f"{fator}: {', '.join(lista)}")

    print("\nPesos dos ativos (%):")
    print((pesos * 100).round(2).to_string())
    print(f"Total: {pesos.sum():.2%}")
    return pesos


# ----------------------------------------------------------
# 6. RENTABILIDADE E PERFORMANCE
# ----------------------------------------------------------

def baixar_cdi(datas):
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        "?formato=json&dataInicial=29/03/2024&dataFinal=01/07/2026"
    )
    resposta = requests.get(url, timeout=30)
    resposta.raise_for_status()
    dados = pd.DataFrame(resposta.json())
    if dados.empty:
        raise ValueError("Banco Central não retornou dados do CDI.")

    dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y")
    taxas = dados.set_index("data")["valor"].astype(float) / 100
    cdi = (1 + taxas).cumprod()
    return cdi.reindex(datas).ffill().fillna(1.0)


def baixar_ibovespa(inicio, fim):
    dados = yf.download(
        "^BVSP", start=inicio, end=fim, auto_adjust=True,
        progress=False, threads=False
    )
    if dados.empty:
        raise ValueError("Não foi possível baixar o Ibovespa.")

    if isinstance(dados.columns, pd.MultiIndex):
        serie = dados["Close"].squeeze()
    else:
        serie = dados["Close"]

    return normalizar_indice(serie.to_frame(name="Ibovespa"))["Ibovespa"].dropna()


def calcular_metricas(serie, cdi_diario, anos):
    retornos = serie.pct_change().dropna()
    total = serie.iloc[-1] / serie.iloc[0] - 1
    anualizado = (1 + total) ** (1 / anos) - 1
    volatilidade = retornos.std() * np.sqrt(252)
    drawdown = (serie / serie.cummax() - 1).min()

    excesso = retornos - cdi_diario.reindex(retornos.index)
    desvio = excesso.std()
    sharpe = (
        excesso.mean() / desvio * np.sqrt(252)
        if pd.notna(desvio) and desvio != 0 else np.nan
    )
    return [total, anualizado, volatilidade, drawdown, sharpe]


def analisar_performance(pesos):
    print("\n" + "=" * 60)
    print("RENTABILIDADE E PERFORMANCE")
    print("=" * 60)

    precos = baixar_precos(
        list(pesos.index), DATA_INICIO_CARTEIRA,
        DATA_FIM_CARTEIRA, "precos_carteira.csv"
    )
    precos = precos[pesos.index].dropna(how="all").ffill()

    if precos.empty or precos.iloc[0].isna().any():
        raise ValueError(
            "Há ativos sem cotação na data inicial da carteira. "
            "Verifique os tickers e o cache."
        )

    carteira = (precos / precos.iloc[0] * pesos).sum(axis=1)

    ibov = baixar_ibovespa(DATA_INICIO_CARTEIRA, DATA_FIM_CARTEIRA)
    ibov = ibov.reindex(precos.index).ffill()
    if ibov.isna().any():
        raise ValueError("Existem datas sem cotação do Ibovespa.")
    ibov = ibov / ibov.iloc[0]

    cdi = baixar_cdi(precos.index)
    cdi_diario = cdi.pct_change().fillna(0)
    anos = (precos.index[-1] - precos.index[0]).days / 365.25

    resultado = pd.DataFrame(
        {
            "Carteira": calcular_metricas(carteira, cdi_diario, anos),
            "Ibovespa": calcular_metricas(ibov, cdi_diario, anos),
        },
        index=[
            "Retorno acumulado", "Retorno anualizado",
            "Volatilidade anual", "Drawdown máximo", "Sharpe"
        ],
    )

    cdi_total = cdi.iloc[-1] - 1
    cdi_anual = (1 + cdi_total) ** (1 / anos) - 1

    print(f"Período: {precos.index[0].date()} a {precos.index[-1].date()}")
    print("\nIndicadores (%):")
    print((resultado.iloc[:4] * 100).round(2).to_string())
    print(
        f"\nSharpe - Carteira: {resultado.loc['Sharpe', 'Carteira']:.2f}"
        f" | Ibovespa: {resultado.loc['Sharpe', 'Ibovespa']:.2f}"
    )
    print(f"CDI acumulado: {cdi_total:.2%}")
    print(f"CDI anualizado: {cdi_anual:.2%}")

    resultado.to_csv("resultado_factor_investing.csv", encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.plot(carteira, label="Carteira Multifatorial", linewidth=2)
    plt.plot(ibov, label="Ibovespa", linewidth=2)
    plt.plot(cdi, label="CDI", linestyle="--", linewidth=2)
    plt.title("Carteira Multifatorial vs Ibovespa vs CDI")
    plt.xlabel("Data")
    plt.ylabel("Valor normalizado (base 1,0)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafico_carteira_vs_ibov.png", dpi=150)
    plt.show()


# ----------------------------------------------------------
# 7. EXECUÇÃO
# ----------------------------------------------------------

def main():
    print("INICIANDO O ESTUDO DE FACTOR INVESTING")

    ranking_baixa_volatilidade()
    ranking_valor()
    ranking_qualidade()
    ranking_momento()

    pesos = montar_pesos()
    analisar_performance(pesos)

    print("\nExecução concluída.")
    print("Arquivos gerados:")
    print("- resultado_factor_investing.csv")
    print("- grafico_carteira_vs_ibov.png")
    print(f"- caches na pasta '{PASTA_CACHE}'")


if __name__ == "__main__":
    main()
