# Ap2_factor.investing

# Factor Investing no Mercado Acionário Brasileiro

## Objetivo
Este trabalho analisa uma carteira multifatorial baseada em quatro fatores de investimento: baixa volatilidade, valor, qualidade e momento.

O objetivo é comparar o desempenho da carteira com o Ibovespa e o CDI no período analisado.

## Metodologia
A carteira utiliza quatro fatores, cada um representando 25% do capital investido. Cada fator é composto por quatro ações, com pesos iguais por posição. Quando uma ação aparece em mais de um fator, seus pesos são somados.

A estratégia utiliza a metodologia buy and hold, sem rebalanceamento durante o período de análise.

## Fontes de dados
- Yahoo Finance: preços históricos e informações financeiras.
- Banco Central do Brasil: série histórica do CDI.
- B3: informações sobre a composição do mercado acionário.

## Bibliotecas
- yfinance
- pandas
- numpy
- requests
- matplotlib

## Como executar
1. Instale as bibliotecas necessárias:

   `pip install -r requirements.txt`

2. Execute o arquivo principal:

   `python factor_investing_completo.py`

O programa apresenta os rankings dos fatores, a composição da carteira, os indicadores de desempenho e o gráfico comparativo com o Ibovespa e o CDI.

## Integrantes
Preencher com os nomes dos integrantes do grupo.

## Instituição
IBMEC
