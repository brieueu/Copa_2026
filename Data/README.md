# Data — Copa 2026

Esta pasta foi recriada para armazenar dados reais baixados do Kaggle e bases processadas para o notebook `Copa_2026_Data_Pipeline_e_Simulacao.ipynb`.

## Fontes Kaggle usadas

1. `justdhia/ea-sports-fc-26-player-ratings`
   - ratings de jogadores EA Sports FC 26.

2. `afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings`
   - ratings Elo históricos por seleção.

3. `samandarabdujabbar/ea-sports-fc-25-complete-player-stats-and-analysis`
   - estatísticas completas de jogadores EA Sports FC 25.

4. `pranishkessi/fifa-world-cup-2026-prediction-simulator`
   - grupos, fixtures, probabilidades de partidas, probabilidades por seleção e slots de chaveamento da Copa 2026.

## Estrutura

```text
Data/
├── dataset_manifest.json
├── raw/
│   └── kaggle/
│       └── ... datasets baixados ...
└── processed/
    ├── copa_2026_master_team_dataset.csv
    ├── copa_2026_group_stage_simulated_matches.csv
    ├── copa_2026_group_stage_tables.csv
    ├── copa_2026_best_thirds.csv
    ├── copa_2026_resolved_bracket_simulation.csv
    ├── copa_2026_base_combinada.xlsx
    └── simulation_summary.json
```

## Notebook principal atualizado

O notebook principal do projeto é:

`Vencedor_Copa_2026_Notebook.ipynb`

Ele consome a base processada em `Data/processed/copa_2026_master_team_dataset.csv`, roda simulações Monte Carlo e gera gráficos em `outputs/`.

O notebook `Copa_2026_Data_Pipeline_e_Simulacao.ipynb` permanece como auditoria/ingestão dos dados Kaggle.

## Observação metodológica

A base combinada cruza dados por nome de seleção normalizado. O chaveamento é resolvido a partir dos slots do dataset `pranishkessi/fifa-world-cup-2026-prediction-simulator`. A simulação é probabilística, usa seed fixa (`42`) e não representa resultado oficial.

Na versão atualizada do notebook principal, a força das seleções dá peso maior à qualidade individual do elenco (EA FC 26, EA FC 25, valor de mercado e ratings por setor) para lidar melhor com confrontos intercontinentais pouco frequentes no novo formato de 48 seleções.
