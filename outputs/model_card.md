# Model card — Previsor Copa 2026 autocontido

## Objetivo
Produzir previsões probabilísticas para a Copa do Mundo FIFA 2026 em um notebook único.

## Modelos usados
- Elo/FIFA ranking: peso 30%
- Poisson de gols: peso 30%
- Proxy não linear/gradient boosting: peso 25%
- Estrutural/econômico: peso 15%

## Limitações
- Os dados são embutidos e aproximados, porque o diretório original de dados foi removido.
- O chaveamento de mata-mata é uma aproximação balanceada, não a tabela oficial completa da FIFA.
- Lesões, convocações definitivas, mando específico, clima e forma imediatamente anterior ao torneio não estão modelados.
- O resultado é uma previsão probabilística, não um resultado oficial.
