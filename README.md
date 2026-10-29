# FraudShield

FraudShield is an educational web app for exploring synthetic payment-risk scoring. A Flask server provides a browser dashboard and JSON API. A scikit-learn logistic regression model scores generated transactions; a threshold slider shows how changing the alert cutoff affects results on a held-out test set.

This is a local demo using synthetic data, not Visa data or a production fraud system.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The dashboard trains a model on startup, displays evaluation metrics, and streams generated transactions.

## Model and evaluation

- The app generates 20,000 synthetic transactions with known labels and a deliberately rare fraud class.
- It splits the data into 80% training and 20% test sets using stratification, preserving the fraud share in both splits.
- Logistic regression is fit only on the training split. The test split is held out for evaluation.
- The slider changes the probability-score threshold used to flag examples as possible fraud.
- Precision, recall, F1, false-positive rate, and the confusion matrix update from held-out predictions. ROC-AUC uses continuous model scores and stays constant as the threshold changes.
- The dataset and metrics are reproducible from a fixed random seed. They only measure how this model performs on this synthetic generator; they do not establish real-world performance.

The model uses log purchase amount, log distance from recent activity, recent attempt count, new-merchant flag, country mismatch, and e-commerce flag. Balanced class weights help the learner account for the rare positive class; the resulting model scores are not presented as calibrated real-world fraud probabilities.

## API

- `GET /api/health` — server status and split sizes.
- `GET /api/evaluation?threshold=0.25` — held-out metrics at a threshold from 0.01 to 0.99.
- `GET /api/transactions/generate?threshold=0.25` — create and score one synthetic transaction.
- `POST /api/score` — score a supplied JSON transaction.

Example request:

```bash
curl -X POST http://127.0.0.1:5000/api/score \
  -H 'Content-Type: application/json' \
  -d '{"amount":1250,"distance_km":1200,"attempts_10m":4,"new_merchant":true,"country_mismatch":false,"channel":"E-commerce"}'
```

## Resume bullet template

> Built a Flask transaction-risk app with scikit-learn logistic regression; evaluated on a stratified 20% synthetic holdout set and visualized precision, recall, F1, ROC-AUC, and confusion-matrix tradeoffs across decision thresholds.

Only add specific metric values after observing the app, and describe the data as synthetic. Do not present these results as Visa or production performance.
