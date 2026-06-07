# Model Card — Copa 2026 Predictor Atualizado

## Dados
Usa datasets Kaggle em `Data/raw/kaggle` e base combinada em `Data/processed/copa_2026_master_team_dataset.csv`.

## Método
Simulação Monte Carlo estocástica com força por seleção baseada em Elo, rankings, ratings EA FC 26, EA FC 25, valor de mercado, ataque, defesa e goleiros.

## Ponto metodológico central
Para o formato de 48 seleções, a qualidade individual recebe peso alto porque várias seleções de África, Ásia e Oceania terão poucos confrontos recentes contra potências europeias e sul-americanas. O multiplicador intercontinental aumenta a influência da diferença técnica quando o histórico direto é provavelmente escasso.

## Limitações
Resultado é previsão probabilística, não resultado oficial. Lesões, convocações definitivas, forma imediatamente anterior ao torneio e decisões oficiais finais de chaveamento podem alterar as probabilidades. Ratings de jogos e valor de mercado são proxies imperfeitos de qualidade futebolística.
