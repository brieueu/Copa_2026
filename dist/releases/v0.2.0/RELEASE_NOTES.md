# Copa 2026 Predictor v0.2.0

## Resumo

A v0.2.0 atualiza o modelo do Copa 2026 Predictor para usar Elo dinâmico pós-fase de grupos, forma recente, sede/local e modelagem explícita de pênaltis no mata-mata. A simulação Monte Carlo continua com 100.000 torneios e seed 42.

## Bases utilizadas

- `Data/raw/external/v0.2.0_dynamic_elo/openfootball_worldcup_2026.json`: resultados e fixtures reais da Copa 2026.
- `Data/raw/external/v0.2.0_dynamic_elo/international_results.csv`: histórico internacional para forma recente.
- `Data/raw/external/v0.2.0_dynamic_elo/international_shootouts.csv`: histórico de disputas por pênaltis.
- `Data/raw/external/v0.2.0_dynamic_elo/world_football_elo_*.tsv`: ratings, resultados, fixtures e mapeamentos World Football Elo.
- Kaggle datasets já usados na v0.1.0 para estrutura do torneio, elenco, ratings e probabilidades de referência.

A proveniência dos downloads externos está em `Data/raw/external/v0.2.0_dynamic_elo/DOWNLOAD_MANIFEST.json`.

## Diferenças v0.1.0 → v0.2.0

- força estática substituída por `model_strength_v0_2`;
- Elo recalculado jogo a jogo com margem de vitória;
- forma recente dos últimos 10 jogos e dos últimos 6 meses;
- vantagem de país-sede/região adicionada ao cálculo de partida;
- pênaltis separados de vitória em 90/120 minutos;
- novos outputs, gráficos, model card e README.

## Sucessos e limitações da v0.1.0

A v0.1.0 funcionou como baseline reprodutível e visual, mas tinha baixa sensibilidade a resultados reais, placares largos, forma recente, sede/local e pênaltis. A v0.2.0 corrige esses pontos de forma inicial e auditável.

## Como reproduzir

```bash
python -m pytest -q
python tools/build_actual_2026_results.py
python tools/dynamic_elo.py --matches Data/processed/actual_2026_matches.csv --out Data/processed/dynamic_elo_after_group_stage.csv
python tools/build_recent_form_features.py
python tools/run_fast_100k_simulation.py
python tools/create_professional_readme_charts.py
python -m pytest -q
```

## Packages anexáveis

- `copa_2026_v0.2.0_bases_de_dados.zip`
- `copa_2026_v0.2.0_graficos.zip`
- `copa_2026_v0.2.0_outputs_modelo.zip`
- `SHA256SUMS.txt`

## Checksums

Consulte `dist/releases/v0.2.0/SHA256SUMS.txt`.

## Publicação

Nenhum push, tag ou GitHub Release foi executado nesta etapa.
