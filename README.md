# FraudShield

An educational payment-risk demo with a live transaction dashboard, a small JSON scoring API, and a reproducible synthetic batch evaluation. It uses generated data and transparent rules; it is not a production fraud detection system and does not use Visa data or Visa systems.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). The dashboard streams generated transactions. Use **Run 1,000 txn batch** to see reproducible synthetic evaluation metrics; change the count through the API for up to 10,000 records.

## API

- `GET /api/health` — readiness and demo mode.
- `POST /api/transactions/generate` — generate, score, and return one synthetic transaction.
- `POST /api/score` — score a supplied transaction.
- `POST /api/evaluate` — evaluate a reproducibly seeded synthetic batch.

Example score request:

```bash
curl -X POST http://127.0.0.1:5000/api/score \
  -H 'Content-Type: application/json' \
  -d '{"amount":1250,"distance_km":1200,"attempts_10m":4,"new_merchant":true,"country_mismatch":false}'
```

The scoring rules add points for high amount, unusual distance, repeated attempts, a new merchant, and country mismatch. Scores from 0–34 are approved, 35–69 sent for review, and 70–99 declined. The API response includes the risk score, decision, reasons, and measured local scoring time.

## Metrics and limitations

The batch creates synthetic fraud and non-fraud examples with deliberately different feature distributions. `precision` and `recall` treat **declined** as a positive fraud detection; transactions sent to review are not counted as detected. `false_positive_rate` is the share of synthetic non-fraud transactions declined. Latency is a local Python rule-scoring measurement and does not include network or browser time. Results are seed-reproducible, but describe only this generated dataset and these heuristic rules. They should not be presented as real-world performance or production latency.

This demo intentionally avoids storing real cardholder information. Do not enter real payment or personal data.

## Suggested resume line after running the batch

> Built a Python/Flask transaction-risk scoring API and live dashboard; evaluated transparent fraud rules on 1,000 reproducible synthetic transactions and reported precision, recall, false-positive rate, and local scoring latency.

Replace the batch size and any metric values only with values you observed from the app; describe the records as synthetic and latency as local. This project demonstrates API design, input validation, explainable decision rules, UI integration, and evaluation—not machine-learning model training or payment-network deployment.
