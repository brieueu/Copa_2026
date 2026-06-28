# Model Card — Copa 2026 Predictor v0.2.0

## Dados
Usa Kaggle para estrutura do torneio/elencos, OpenFootball para resultados/fixtures da Copa 2026, World Football Elo e international-results para forma recente e ratings.

## Método
A v0.2.0 mantém Monte Carlo com 100.000 torneios, mas troca a força majoritariamente estática por `model_strength_v0_2`: Elo dinâmico pós-fase de grupos, força de elenco EA FC, forma recente, sede/local e prior externo v0.1.0.

## Pênaltis
O mata-mata separa vitória em 90/120 minutos, empate após prorrogação e disputa de pênaltis. O bracket contém `knockout_resolution`, `won_by_penalties`, `p_penalty_home` e `p_penalty_away`.

## Limitações
Resultado é previsão probabilística, não fato. Pesos de sede/local e forma são hipóteses iniciais e devem ser calibrados em backtests futuros.
