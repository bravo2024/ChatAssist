# ChatAssist

Classifies support-chat intents and predicts whether a conversation will resolve without a human handoff.

Trains four classifiers on synthetic customer service conversation data to predict whether a chat interaction will resolve successfully. Features include intent confidence, context relevance, sentiment score, escalation status, and topic category.

## Results (holdout)

Best model (Logistic Regression):

| Metric | Value |
|---|---|
| ROC AUC | 0.731 |
| Gini | 0.463 |
| KS Statistic | 0.477 |
| F1 Score | 0.493 |
| Accuracy | 0.704 |

5-fold CV AUC: 0.725 ± 0.029.

## Run it

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Tabs

| Tab | What it does |
|---|---|
| **Explorer** | Conversation log, resolution distribution, feature overview |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Intent Analysis** | Topic distribution pie chart, resolution rate by topic, sentiment by outcome |
| **Insights** | Intent confidence distributions, context relevance analysis, escalation patterns |

## The data behind it

Synthetic conversation data engineered to mirror customer service chat patterns: intent categories, sentiment scores, context relevance, escalation flags, and resolution outcomes.

### Layout

```
ChatAssist/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## License

MIT
