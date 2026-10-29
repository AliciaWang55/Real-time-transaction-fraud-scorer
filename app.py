"""FraudShield: a synthetic-data fraud scoring and evaluation demo."""

# Standard-library helpers create repeatable data and measure scorer time.
from __future__ import annotations

import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone

# Flask serves the dashboard and JSON API. scikit-learn trains and evaluates
# the supervised model using a stratified train/test split.
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
RECENT = deque(maxlen=100)
FEATURE_NAMES = ["log_amount", "log_distance", "attempts_10m", "new_merchant", "country_mismatch", "e_commerce"]
DATA_SEED = 20261029
DATA_SIZE = 20000
DEFAULT_THRESHOLD = 0.85


# Make fake transactions with a hidden label. Fraud examples are uncommon and
# tend to have different feature patterns, but the distributions overlap.
def make_transaction(rng: random.Random | None = None) -> dict:
    rng = rng or random
    fraud = rng.random() < 0.015
    if fraud:
        amount = round(rng.choice([rng.lognormvariate(7.1, 0.65), rng.lognormvariate(2.4, 0.5)]), 2)
        distance = round(rng.expovariate(1 / 1150))
        attempts = rng.choices([1, 2, 3, 4, 5, 6], [0.10, 0.13, 0.19, 0.22, 0.20, 0.16])[0]
        new_merchant = rng.random() < 0.62
        country_mismatch = rng.random() < 0.32
        channel = rng.choices(["Card present", "E-commerce", "Digital wallet"], [0.12, 0.73, 0.15])[0]
    else:
        amount = round(min(rng.lognormvariate(3.8, 0.9), 1800), 2)
        distance = round(rng.expovariate(1 / 48))
        attempts = rng.choices([1, 2, 3], [0.91, 0.08, 0.01])[0]
        new_merchant = rng.random() < 0.14
        country_mismatch = rng.random() < 0.006
        channel = rng.choices(["Card present", "E-commerce", "Digital wallet"], [0.56, 0.32, 0.12])[0]
    return {
        "id": uuid.uuid4().hex[:10].upper(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "amount": amount,
        "currency": "USD",
        "merchant": rng.choice(["Northstar Market", "Metro Fuel", "CloudNine Travel", "Daily Goods", "Arcade Supply", "Harbor Eats"]),
        "channel": channel,
        "distance_km": distance,
        "attempts_10m": attempts,
        "new_merchant": new_merchant,
        "country_mismatch": country_mismatch,
        "_label_fraud": fraud,
    }


# Convert transaction fields into numeric features in a fixed order so model
# training, live scoring, and held-out evaluation all use the same inputs.
def transaction_features(tx: dict) -> list[float]:
    return [
        float(np.log1p(max(0, float(tx.get("amount", 0))))),
        float(np.log1p(max(0, float(tx.get("distance_km", 0))))),
        float(tx.get("attempts_10m", 1)),
        float(bool(tx.get("new_merchant", False))),
        float(bool(tx.get("country_mismatch", False))),
        float(tx.get("channel") == "E-commerce"),
    ]


# Create a seeded synthetic dataset, hold out a stratified 20% test set, and
# fit logistic regression only on the other 80%. Test scores are never used
# to fit the model. Balanced class weights help with the rare fraud examples.
def train_model() -> dict:
    rng = random.Random(DATA_SEED)
    transactions = [make_transaction(rng) for _ in range(DATA_SIZE)]
    features = np.asarray([transaction_features(tx) for tx in transactions], dtype=float)
    labels = np.asarray([int(tx["_label_fraud"]) for tx in transactions], dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=DATA_SEED
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=DATA_SEED, solver="liblinear"),
    )
    classifier.fit(x_train, y_train)
    fraud_scores = classifier.predict_proba(x_test)[:, 1]
    return {
        "model": classifier,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "fraud_scores": fraud_scores,
        "fraud_rate": float(labels.mean()),
    }


MODEL_DATA = train_model()
MODEL = MODEL_DATA["model"]


# Calculate threshold-dependent metrics from held-out scores. A transaction
# counts as flagged when its model score is at or above the chosen threshold.
def evaluation_metrics(threshold: float) -> dict:
    threshold = min(0.99, max(0.01, float(threshold)))
    actual = MODEL_DATA["y_test"]
    scores = MODEL_DATA["fraud_scores"]
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    return {
        "threshold": threshold,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "test_size": int(len(actual)),
        "test_fraud_count": int(actual.sum()),
        "flagged_count": int(predicted.sum()),
        "false_positive_rate": float(fp / max(1, fp + tn)),
    }


# Score one live transaction with the trained model and keep the recent result
# in memory for this server session. The demo uses the current threshold to
# mark higher-scoring transactions for review or decline.
def process(tx: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    started = time.perf_counter()
    risk_score = float(MODEL.predict_proba([transaction_features(tx)])[0][1])
    if risk_score >= min(0.85, threshold + 0.30):
        decision = "declined"
    elif risk_score >= threshold:
        decision = "review"
    else:
        decision = "approved"
    reasons = []
    if tx.get("amount", 0) >= 500:
        reasons.append("High purchase amount")
    if tx.get("distance_km", 0) >= 300:
        reasons.append("Purchase far from recent activity")
    if tx.get("attempts_10m", 1) >= 3:
        reasons.append("Several attempts in a short period")
    if tx.get("new_merchant"):
        reasons.append("First purchase at this merchant")
    if tx.get("country_mismatch"):
        reasons.append("Merchant country differs from card profile")
    result = {
        **tx,
        "risk_score": round(risk_score * 100, 1),
        "decision": decision,
        "reasons": reasons or ["No notable risk signals"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    result.pop("_label_fraud", None)
    RECENT.appendleft(result)
    return result


# Main page markup and styles. The embedded JavaScript below updates the
# dashboard, requests live transactions, and refreshes test metrics as the
# threshold slider moves.
PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FraudShield | Model evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f6f9;--ink:#162338;--muted:#718096;--line:#e5eaf0;--blue:#185adb;--green:#07845b;--amber:#bc7200;--red:#c23a45}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px 'DM Sans',sans-serif}.top{height:66px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 max(calc((100vw - 1180px)/2),24px)}.brand{display:flex;align-items:center;gap:11px;font-weight:700;font-size:18px}.shield{background:#eaf0ff;color:var(--blue);height:34px;width:34px;border-radius:10px;display:grid;place-items:center;font-size:18px}.live{color:var(--green);font-size:11px;font-weight:700}.live:before{content:'';display:inline-block;background:#17a673;width:7px;height:7px;border-radius:50%;margin-right:7px}.wrap{max-width:1180px;margin:30px auto;padding:0 24px}.eyebrow{text-transform:uppercase;color:var(--muted);letter-spacing:1.2px;font-size:10px;font-weight:700}.heading{display:flex;justify-content:space-between;align-items:end;margin-bottom:18px}.heading h1{font-size:26px;letter-spacing:-.7px;margin:5px 0}.sub{color:var(--muted);margin:0}.panel,.card{background:#fff;border:1px solid var(--line);border-radius:11px}.card{padding:16px 18px}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:14px}.label{font-size:11px;color:var(--muted);font-weight:600}.value{font-size:23px;font-weight:700;margin:8px 0 3px;letter-spacing:-.5px}.hint{font-size:10px;color:var(--muted)}.green{color:var(--green)}.orange{color:var(--amber)}.red{color:var(--red)}.panel{padding:18px 20px}.row{display:grid;grid-template-columns:1.1fr .9fr;gap:14px}.panelhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}.panel h2{font-size:14px;margin:0}.small{font-size:11px;color:var(--muted)}.split{font-family:'DM Mono',monospace;font-size:10px;background:#f3f6fb;padding:6px 9px;border-radius:6px;color:#4b5c74}.sliderbox{background:#f7f9fc;border:1px solid var(--line);border-radius:9px;padding:14px 15px;margin-bottom:14px}.sliderhead{display:flex;justify-content:space-between;align-items:center;font-weight:600;font-size:12px;margin-bottom:9px}.sliderhead strong{font:500 12px 'DM Mono';color:var(--blue)}input[type=range]{width:100%;accent-color:var(--blue)}.rangeends{display:flex;justify-content:space-between;font-size:10px;color:var(--muted)}.tradeoff{font-size:11px;line-height:1.55;color:var(--muted);margin:9px 0 0}.matrix{display:grid;grid-template-columns:24px 1fr 1fr;gap:5px;align-items:stretch;margin:6px auto 0;max-width:390px}.mhead{font-size:9px;text-align:center;color:var(--muted);padding:5px}.axis{writing-mode:vertical-rl;transform:rotate(180deg);font-size:9px;color:var(--muted);display:flex;align-items:center;justify-content:center}.cell{min-height:88px;border-radius:8px;display:flex;flex-direction:column;align-items:center;justify-content:center}.cell strong{font:600 23px 'DM Mono'}.cell span{font-size:10px;margin-top:3px}.tn,.tp{background:#e7f5ee;color:#08724e}.fp{background:#fff0df;color:#a85e00}.fn{background:#fbe8e9;color:#b6323b}.foot{font-size:10px;color:var(--muted);line-height:1.6;margin:14px 2px}.stream{margin-top:14px}.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;padding:9px 8px;border-bottom:1px solid var(--line)}td{padding:11px 8px;border-bottom:1px solid #f0f2f5;font-size:11px}.mono{font-family:'DM Mono',monospace;font-size:10px}.pill{padding:4px 8px;border-radius:99px;font-weight:700;font-size:9px;text-transform:uppercase}.approved{background:#e5f6ef;color:#06734e}.review{background:#fff2dc;color:#a35d00}.declined{background:#fce9eb;color:#b02f3b}.toast{position:fixed;bottom:20px;right:20px;background:#17263c;color:#fff;padding:12px 16px;border-radius:8px;font-size:12px;opacity:0;transition:.2s}.toast.show{opacity:1}@media(max-width:900px){.cards{grid-template-columns:repeat(3,1fr)}.row{grid-template-columns:1fr}}@media(max-width:620px){.cards{grid-template-columns:repeat(2,1fr)}.heading{align-items:start;gap:10px;flex-direction:column}}
</style></head><body>
<header class="top"><div class="brand"><div class="shield">⬡</div>FraudShield</div><div class="live">SYNTHETIC MODEL · LOCAL DEMO</div></header>
<main class="wrap"><div class="heading"><div><div class="eyebrow">Risk operations · model evaluation</div><h1>Threshold lab</h1><p class="sub">Explore fraud alerts on a held-out test set, then watch the live synthetic stream.</p></div><div class="split">Logistic regression · stratified 80 / 20 split</div></div>
<section class="cards"><div class="card"><div class="label">Precision</div><div class="value green" id="precision">—</div><div class="hint">Flagged transactions that were fraud</div></div><div class="card"><div class="label">Recall</div><div class="value orange" id="recall">—</div><div class="hint">Fraud cases successfully flagged</div></div><div class="card"><div class="label">F1 score</div><div class="value" id="f1">—</div><div class="hint">Balance of precision and recall</div></div><div class="card"><div class="label">ROC-AUC</div><div class="value" id="auc">—</div><div class="hint">Ranking quality across thresholds</div></div><div class="card"><div class="label">False positive rate</div><div class="value red" id="fpr">—</div><div class="hint">Legitimate purchases flagged</div></div></section>
<section class="row"><div class="panel"><div class="panelhead"><h2>Choose a fraud-alert threshold</h2><span class="split" id="testInfo">Held-out test set</span></div><div class="sliderbox"><div class="sliderhead"><span>Flag as possible fraud at or above</span><strong id="thresholdText">85%</strong></div><input id="threshold" type="range" min="0.01" max="0.95" step="0.01" value="0.85"><div class="rangeends"><span>Lower threshold · more alerts</span><span>Higher threshold · fewer alerts</span></div><p class="tradeoff">Lower the threshold to catch more fraud (higher recall), with more legitimate purchases flagged. Raise it to reduce false alarms, while some fraud may be missed. ROC-AUC stays fixed because it measures ranking across all thresholds.</p></div><div class="panelhead"><h2>Confusion matrix</h2><span class="small" id="flagged">—</span></div><div class="matrix"><div></div><div class="mhead">Predicted legitimate</div><div class="mhead">Predicted fraud</div><div class="axis">Legitimate</div><div class="cell tn"><strong id="tn">0</strong><span>True negative · legit cleared</span></div><div class="cell fp"><strong id="fp">0</strong><span>False positive · false alarm</span></div><div class="axis">Fraud</div><div class="cell fn"><strong id="fn">0</strong><span>False negative · missed fraud</span></div><div class="cell tp"><strong id="tp">0</strong><span>True positive · fraud caught</span></div></div><p class="foot">Model is trained only on the 80% training split. These metrics are computed from the untouched 20% test split. The synthetic fraud prevalence is <strong id="fraudRate">—</strong>; results are not evidence of real-world payment performance.</p></div>
<div class="panel"><div class="panelhead"><h2>Model setup</h2><span class="small">Reproducible experiment</span></div><p class="tradeoff"><strong>Data:</strong> generated transactions with known synthetic labels and a deliberately rare fraud class.</p><p class="tradeoff"><strong>Features:</strong> log purchase amount, log distance, recent attempt count, new-merchant flag, country mismatch, and e-commerce flag.</p><p class="tradeoff"><strong>Model:</strong> scikit-learn LogisticRegression with balanced class weights. It learns feature weights from the training split.</p><p class="tradeoff"><strong>Evaluation:</strong> precision, recall, F1, and confusion matrix update from test-set scores as you move the slider. ROC-AUC uses the continuous scores.</p><p class="tradeoff"><strong>Important:</strong> synthetic data is designed for a demo. A model can learn generator patterns that do not match real customer behavior.</p></div></section>
<section class="panel stream"><div class="panelhead"><h2>Live synthetic transaction stream</h2><span class="small" id="streamStatus">Starting…</span></div><div class="tablewrap"><table><thead><tr><th>Transaction</th><th>Merchant</th><th>Amount</th><th>Model score</th><th>Decision</th><th>Signal</th></tr></thead><tbody id="rows"><tr><td colspan="6" class="small">Generating sample activity…</td></tr></tbody></table></div></section>
<p class="foot">Educational prototype only. Uses generated transaction data, not Visa data. The live activity stream and held-out evaluation use synthetic examples. Model scores are not calibrated estimates of real fraud probability.</p>
</main><div class="toast" id="toast"></div>
<script>
const fmt=n=>Number(n).toLocaleString(); let streamOn=true;
function showMetrics(d){document.getElementById('precision').textContent=(100*d.precision).toFixed(1)+'%';document.getElementById('recall').textContent=(100*d.recall).toFixed(1)+'%';document.getElementById('f1').textContent=d.f1.toFixed(3);document.getElementById('auc').textContent=d.roc_auc.toFixed(3);document.getElementById('fpr').textContent=(100*d.false_positive_rate).toFixed(2)+'%';document.getElementById('thresholdText').textContent=Math.round(d.threshold*100)+'%';document.getElementById('tn').textContent=fmt(d.confusion_matrix[0][0]);document.getElementById('fp').textContent=fmt(d.confusion_matrix[0][1]);document.getElementById('fn').textContent=fmt(d.confusion_matrix[1][0]);document.getElementById('tp').textContent=fmt(d.confusion_matrix[1][1]);document.getElementById('flagged').textContent=fmt(d.flagged_count)+' flagged · '+fmt(d.test_size)+' test records';document.getElementById('testInfo').textContent=fmt(d.test_size)+' held-out records';document.getElementById('fraudRate').textContent=(100*d.test_fraud_count/d.test_size).toFixed(2)+'%';}
let timer;async function updateMetrics(){let t=document.getElementById('threshold').value;try{let res=await fetch('/api/evaluation?threshold='+t);showMetrics(await res.json())}catch(e){}}
document.getElementById('threshold').addEventListener('input',()=>{document.getElementById('thresholdText').textContent=Math.round(document.getElementById('threshold').value*100)+'%';clearTimeout(timer);timer=setTimeout(updateMetrics,100)});
function addRow(d){let tr=document.createElement('tr'),why=d.reasons[0];tr.innerHTML=`<td class="mono">${d.id}</td><td>${d.merchant}</td><td>$${Number(d.amount).toFixed(2)}</td><td class="mono">${d.risk_score.toFixed(1)}%</td><td><span class="pill ${d.decision}">${d.decision}</span></td><td title="${d.reasons.join('; ')}">${why.length>34?why.slice(0,32)+'…':why}</td>`;let b=document.getElementById('rows');b.prepend(tr);while(b.children.length>8)b.lastChild.remove();document.getElementById('streamStatus').textContent='New sample just scored'}
async function tick(){if(!streamOn)return;try{let t=document.getElementById('threshold').value,r=await fetch('/api/transactions/generate?threshold='+t),d=await r.json();addRow(d)}catch(e){document.getElementById('streamStatus').textContent='API connection unavailable'}}
updateMetrics();tick();setInterval(tick,1200);
</script></body></html>'''


# Render the single-page dashboard in the browser.
@app.get("/")
def index():
    return render_template_string(PAGE)


# Generate and score one fake transaction for the live stream.
@app.get("/api/transactions/generate")
def generate():
    try:
        threshold = float(request.args.get("threshold", DEFAULT_THRESHOLD))
    except ValueError:
        threshold = DEFAULT_THRESHOLD
    return jsonify(process(make_transaction(), threshold))


# Return metrics for a selected threshold, using only the held-out test set.
@app.get("/api/evaluation")
def evaluation():
    try:
        threshold = float(request.args.get("threshold", DEFAULT_THRESHOLD))
    except ValueError:
        return jsonify({"error": "threshold must be a number between 0.01 and 0.99"}), 400
    return jsonify(evaluation_metrics(threshold))


# Score a user-supplied transaction JSON object with the trained classifier.
@app.post("/api/score")
def score():
    tx = request.get_json(silent=True) or {}
    required = {"amount", "distance_km", "attempts_10m"}
    if not required.issubset(tx):
        return jsonify({"error": "Required fields: amount, distance_km, attempts_10m"}), 400
    try:
        if float(tx["amount"]) < 0 or float(tx["distance_km"]) < 0 or int(tx["attempts_10m"]) < 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "amount and distance_km must be nonnegative; attempts_10m must be a positive integer"}), 400
    tx.setdefault("id", uuid.uuid4().hex[:10].upper())
    tx.setdefault("merchant", "Example Merchant")
    tx.setdefault("channel", "E-commerce")
    tx.setdefault("currency", "USD")
    return jsonify(process(tx))


# Let deployment tools or a person quickly check whether the demo is running.
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "synthetic-logistic-regression", "train_size": len(MODEL_DATA["y_train"]), "test_size": len(MODEL_DATA["y_test"])})


# Start the local server when app.py is run directly.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
