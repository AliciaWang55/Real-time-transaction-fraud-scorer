"""FraudShield: an educational, synthetic transaction-risk demo."""
from __future__ import annotations

import math
import random
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
RECENT = deque(maxlen=100)
COUNTS = {"approved": 0, "review": 0, "declined": 0, "fraud": 0, "caught": 0, "false_positive": 0}


def make_transaction(rng: random.Random | None = None) -> dict:
    rng = rng or random
    fraud = rng.random() < 0.075
    if fraud:
        amount = round(rng.choice([rng.uniform(450, 2400), rng.uniform(2, 20)]), 2)
        distance = round(rng.uniform(900, 5000))
        attempts = rng.randint(2, 7)
        new_merchant = rng.random() < 0.76
        country_mismatch = rng.random() < 0.55
    else:
        amount = round(min(rng.expovariate(1 / 62), 850) + 4, 2)
        distance = round(rng.expovariate(1 / 42))
        attempts = rng.choices([1, 2, 3], [0.86, 0.12, 0.02])[0]
        new_merchant = rng.random() < 0.13
        country_mismatch = rng.random() < 0.012
    return {"id": uuid.uuid4().hex[:10].upper(), "timestamp": datetime.now(timezone.utc).isoformat(),
            "amount": amount, "currency": "USD", "merchant": rng.choice(["Northstar Market", "Metro Fuel", "CloudNine Travel", "Daily Goods", "Arcade Supply", "Harbor Eats"]),
            "channel": rng.choice(["Card present", "E-commerce", "Digital wallet"]), "distance_km": distance,
            "attempts_10m": attempts, "new_merchant": new_merchant, "country_mismatch": country_mismatch,
            "_label_fraud": fraud}


def score_transaction(tx: dict) -> dict:
    points, reasons = 0, []
    amount = float(tx.get("amount", 0))
    distance = float(tx.get("distance_km", 0))
    attempts = int(tx.get("attempts_10m", 1))
    if amount >= 1000:
        points += 35; reasons.append("Unusually high amount (≥ $1,000)")
    elif amount >= 500:
        points += 22; reasons.append("High amount (≥ $500)")
    if distance >= 800:
        points += 30; reasons.append("Purchase far from recent activity (≥ 800 km)")
    elif distance >= 300:
        points += 16; reasons.append("Purchase outside usual area (≥ 300 km)")
    if attempts >= 5:
        points += 28; reasons.append("Repeated attempts in the last 10 minutes")
    elif attempts >= 3:
        points += 15; reasons.append("Several attempts in the last 10 minutes")
    if tx.get("new_merchant"):
        points += 12; reasons.append("First purchase at this merchant")
    if tx.get("country_mismatch"):
        points += 25; reasons.append("Merchant country differs from card profile")
    score = min(99, points)
    decision = "declined" if score >= 70 else "review" if score >= 35 else "approved"
    return {"risk_score": score, "decision": decision, "reasons": reasons or ["No notable risk signals"]}


def process(tx: dict) -> dict:
    started = time.perf_counter()
    result = {**tx, **score_transaction(tx)}
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result.pop("_label_fraud", None)
    RECENT.appendleft(result)
    return result


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FraudShield | Risk Operations</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f6f9;--ink:#162338;--muted:#718096;--line:#e5eaf0;--blue:#185adb;--green:#07845b;--amber:#bc7200;--red:#c23a45}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px 'DM Sans',sans-serif}.top{height:66px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 max(calc((100vw - 1200px)/2),24px)}.brand{display:flex;align-items:center;gap:11px;font-weight:700;font-size:18px;letter-spacing:-.4px}.shield{background:#eaf0ff;color:var(--blue);height:34px;width:34px;border-radius:10px;display:grid;place-items:center;font-size:18px}.live{color:var(--green);font-size:12px;font-weight:600}.live:before{content:'';display:inline-block;background:#17a673;width:7px;height:7px;border-radius:50%;margin-right:7px;box-shadow:0 0 0 3px #d8f4e9}.wrap{max-width:1200px;margin:34px auto;padding:0 24px}.eyebrow{text-transform:uppercase;color:var(--muted);letter-spacing:1.3px;font-size:10px;font-weight:700}.heading{display:flex;justify-content:space-between;align-items:end;margin:0 0 22px}.heading h1{font-size:27px;letter-spacing:-.8px;margin:5px 0}.sub{color:var(--muted);margin:0}.buttons{display:flex;gap:9px}.btn{border:1px solid var(--line);background:white;color:var(--ink);font:600 12px 'DM Sans';padding:10px 14px;border-radius:8px;cursor:pointer}.btn.primary{background:var(--blue);color:white;border-color:var(--blue)}.btn:hover{filter:brightness(.97)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card,.panel{background:#fff;border:1px solid var(--line);border-radius:11px}.card{padding:17px 19px}.cardlabel{font-size:11px;color:var(--muted);font-weight:600}.value{font-size:25px;font-weight:700;margin:9px 0 3px;letter-spacing:-.7px}.hint{font-size:11px;color:var(--muted)}.green{color:var(--green)}.orange{color:var(--amber)}.red{color:var(--red)}.grid{display:grid;grid-template-columns:1.6fr 1fr;gap:15px;margin-top:15px}.panel{padding:19px 20px}.panelhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.panel h2{font-size:14px;margin:0}.small{font-size:11px;color:var(--muted)}table{width:100%;border-collapse:collapse;text-align:left}th{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;font-weight:700;padding:9px 8px;border-bottom:1px solid var(--line)}td{padding:12px 8px;border-bottom:1px solid #f0f2f5;font-size:11px}tr:last-child td{border:0}.mono{font-family:'DM Mono',monospace;font-size:10px}.pill{padding:4px 8px;border-radius:99px;font-weight:700;font-size:9px;text-transform:uppercase;letter-spacing:.3px}.approved{background:#e5f6ef;color:#06734e}.review{background:#fff2dc;color:#a35d00}.declined{background:#fce9eb;color:#b02f3b}.barrow{margin:18px 0}.barlabel{display:flex;justify-content:space-between;font-size:11px;margin-bottom:7px}.bar{height:7px;background:#edf0f4;border-radius:10px;overflow:hidden}.bar span{height:100%;display:block;border-radius:10px;transition:width .3s}.approvebar{background:#16a675}.reviewbar{background:#e4a32c}.declinebar{background:#d95762}.thresholds{margin-top:24px;padding-top:16px;border-top:1px solid var(--line);font-size:11px;color:var(--muted);line-height:1.8}.thresholds strong{color:var(--ink)}.foot{color:var(--muted);font-size:10px;margin-top:18px;line-height:1.6}.toast{position:fixed;bottom:20px;right:20px;background:#17263c;color:#fff;padding:12px 16px;border-radius:8px;font-size:12px;opacity:0;transition:.2s}.toast.show{opacity:1}@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.heading{align-items:start;gap:15px;flex-direction:column}.tablewrap{overflow:auto}.top{padding:0 20px}}
</style></head><body><header class="top"><div class="brand"><div class="shield">⬡</div>FraudShield</div><div class="live">DEMO STREAM ACTIVE</div></header><main class="wrap"><div class="heading"><div><div class="eyebrow">Risk operations · synthetic environment</div><h1>Transaction monitoring</h1><p class="sub">A transparent, rules-based view of incoming card activity.</p></div><div class="buttons"><button class="btn" onclick="runBatch(1000)">Run 1,000 txn batch</button><button class="btn primary" onclick="toggleStream()" id="streamBtn">Pause stream</button></div></div>
<section class="cards"><div class="card"><div class="cardlabel">Transactions scored</div><div class="value" id="total">0</div><div class="hint">Since this session began</div></div><div class="card"><div class="cardlabel">Approval rate</div><div class="value green" id="approval">—</div><div class="hint">Approved / total</div></div><div class="card"><div class="cardlabel">Flagged for review</div><div class="value orange" id="reviewCount">0</div><div class="hint">Requires analyst attention</div></div><div class="card"><div class="cardlabel">Median scoring latency</div><div class="value" id="latency">—</div><div class="hint">Local API response time</div></div></section>
<section class="grid"><div class="panel"><div class="panelhead"><h2>Recent transactions</h2><span class="small" id="updated">Waiting for first transaction</span></div><div class="tablewrap"><table><thead><tr><th>Transaction</th><th>Merchant</th><th>Amount</th><th>Risk</th><th>Decision</th><th>Signal</th></tr></thead><tbody id="rows"><tr><td colspan="6" class="small">Starting synthetic stream…</td></tr></tbody></table></div></div><div class="panel"><div class="panelhead"><h2>Decision distribution</h2><span class="small">Session total</span></div><div class="barrow"><div class="barlabel"><span>Approved</span><strong id="aN">0</strong></div><div class="bar"><span class="approvebar" id="aBar" style="width:0"></span></div></div><div class="barrow"><div class="barlabel"><span>Review</span><strong id="rN">0</strong></div><div class="bar"><span class="reviewbar" id="rBar" style="width:0"></span></div></div><div class="barrow"><div class="barlabel"><span>Declined</span><strong id="dN">0</strong></div><div class="bar"><span class="declinebar" id="dBar" style="width:0"></span></div></div><div class="thresholds"><strong>Decision thresholds</strong><br>0–34 · Approve &nbsp; 35–69 · Review &nbsp; 70–99 · Decline<br>Signals: amount, distance, retry frequency, merchant novelty, country mismatch</div></div></section>
<div class="foot">Educational prototype using generated transactions and transparent heuristic rules. Decisions are illustrative; this is not a production fraud model. Synthetic labels support offline evaluation only.</div></main><div class="toast" id="toast"></div>
<script>let counts={approved:0,review:0,declined:0},latencies=[],running=true,busy=false;const fmt=n=>Number(n).toLocaleString();function notify(s){let t=document.getElementById('toast');t.textContent=s;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2600)}function paint(data){let d=data.decision;counts[d]++;latencies.push(data.latency_ms);if(latencies.length>101)latencies.shift();let total=counts.approved+counts.review+counts.declined;document.getElementById('total').textContent=fmt(total);document.getElementById('approval').textContent=(counts.approved/total*100).toFixed(1)+'%';document.getElementById('reviewCount').textContent=fmt(counts.review);let sorted=[...latencies].sort((a,b)=>a-b);document.getElementById('latency').textContent=sorted[Math.floor(sorted.length/2)].toFixed(2)+' ms';[['a',counts.approved],['r',counts.review],['d',counts.declined]].forEach(([k,n])=>{document.getElementById(k+'N').textContent=fmt(n);document.getElementById(k+'Bar').style.width=(n/total*100)+'%'});let tr=document.createElement('tr');let reason=data.reasons[0];tr.innerHTML=`<td class="mono">${data.id}</td><td>${data.merchant}</td><td>$${Number(data.amount).toFixed(2)}</td><td><span class="mono ${d==='declined'?'red':d==='review'?'orange':'green'}">${data.risk_score}/99</span></td><td><span class="pill ${d}">${d}</span></td><td title="${data.reasons.join('; ')}">${reason.length>31?reason.slice(0,29)+'…':reason}</td>`;let body=document.getElementById('rows');body.prepend(tr);while(body.children.length>8)body.lastChild.remove();document.getElementById('updated').textContent='Updated just now'}async function tick(){if(!running||busy)return;try{let r=await fetch('/api/transactions/generate',{method:'POST'}),d=await r.json();paint(d)}catch(e){notify('API connection lost')}}async function runBatch(n){if(busy)return;busy=true;notify('Scoring '+fmt(n)+' synthetic transactions…');try{let r=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({count:n})}),d=await r.json();counts=d.decisions;latencies=[d.p50_latency_ms];renderSummary(d);notify('Batch complete · '+d.total+' transactions · '+d.p50_latency_ms.toFixed(2)+' ms p50')}finally{busy=false}}function renderSummary(d){let total=d.total;document.getElementById('total').textContent=fmt(total);document.getElementById('approval').textContent=(d.decisions.approved/total*100).toFixed(1)+'%';document.getElementById('reviewCount').textContent=fmt(d.decisions.review);document.getElementById('latency').textContent=d.p50_latency_ms.toFixed(2)+' ms';[['a',d.decisions.approved],['r',d.decisions.review],['d',d.decisions.declined]].forEach(([k,n])=>{document.getElementById(k+'N').textContent=fmt(n);document.getElementById(k+'Bar').style.width=(n/total*100)+'%'});document.getElementById('rows').innerHTML=d.sample.map(x=>`<tr><td class="mono">${x.id}</td><td>${x.merchant}</td><td>$${x.amount.toFixed(2)}</td><td><span class="mono">${x.risk_score}/99</span></td><td><span class="pill ${x.decision}">${x.decision}</span></td><td title="${x.reasons.join('; ')}">${x.reasons[0]}</td></tr>`).join('');document.getElementById('updated').textContent='Batch evaluation complete'}function toggleStream(){running=!running;document.getElementById('streamBtn').textContent=running?'Pause stream':'Resume stream'}setInterval(tick,850);</script></body></html>'''


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/transactions/generate")
def generate():
    return jsonify(process(make_transaction()))


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
    tx.setdefault("merchant", "Example Merchant"); tx.setdefault("channel", "E-commerce"); tx.setdefault("currency", "USD")
    return jsonify(process(tx))


@app.post("/api/evaluate")
def evaluate():
    body = request.get_json(silent=True) or {}
    try:
        count = max(1, min(10000, int(body.get("count", 1000))))
    except (ValueError, TypeError):
        return jsonify({"error": "count must be an integer from 1 to 10000"}), 400
    rng = random.Random(20261028)
    results, decisions, latencies = [], {"approved": 0, "review": 0, "declined": 0}, []
    tp = fp = fn = 0
    for i in range(count):
        tx = make_transaction(rng)
        label = tx.pop("_label_fraud")
        result = {**tx, **score_transaction(tx)}
        start = time.perf_counter_ns()
        score_transaction(tx)
        latency = (time.perf_counter_ns() - start) / 1_000_000
        latencies.append(latency)
        decisions[result["decision"]] += 1
        predicted = result["decision"] == "declined"
        tp += predicted and label; fp += predicted and not label; fn += (not predicted) and label
        if i < 8: results.append({**result, "latency_ms": round(latency, 3)})
    latencies.sort()
    p50 = latencies[len(latencies)//2]; p95 = latencies[int((len(latencies)-1)*.95)]
    return jsonify({"total": count, "decisions": decisions, "fraud_count": tp+fn,
        "precision": round(tp/(tp+fp), 4) if tp+fp else 0,
        "recall": round(tp/(tp+fn), 4) if tp+fn else 0,
        "false_positive_rate": round(fp/max(1, count-(tp+fn)), 4),
        "p50_latency_ms": round(p50, 4), "p95_latency_ms": round(p95, 4), "sample": results,
        "evaluation_note": "Fraud = declined. Review outcomes count as not caught for this conservative metric."})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "mode": "synthetic-demo", "recent_transactions": len(RECENT)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
