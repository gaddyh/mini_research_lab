# 📊 Mini Research Lab — Signal Validation Engine

A Python-based system for evaluating trading ideas as **statistical hypotheses** and determining whether they survive real market conditions.

---

## 🚀 What This Project Does

Most trading tools try to **find profitable strategies**.

This system does something different:

> It tests whether an idea is **real, stable, and generalizable** — or just noise.

---

## 🧠 Core Concept

Every idea goes through a structured pipeline:

```
Idea → Signal → Variations → Statistical Test → Score → Stability → Decision
```

Instead of asking:

> “Does this work?”

We ask:

> “Does this survive reality?”

---

## 🔍 Example Insights

### ✅ Volatility Clustering

```bash
python cli.py --symbols AAPL MSFT SPY --family volatility_clustering
```

Output:

```
CROSS-SYMBOL SUMMARY

VOLATILITY CLUSTERING:
  3/3 → PROMOTE
  ✔ Strong across market
  Confidence: HIGH
```

📌 Interpretation:
- Significant across all assets  
- Consistent direction  
- Meaningful explanatory power (R² up to ~0.12)  

➡️ This is a **real market structure**

---

### ⚠️ Mean Reversion (Short-Term Only)

```bash
python cli.py --symbols AAPL --family mean_reversion \
  --start-date 2020-01-01 --end-date 2023-12-31
```

Output:

```
MEAN REVERSION:
  0/1 → PROMOTE
  1/1 → REFINE
  ⚠ Signal is weak or regime-dependent
  Confidence: MEDIUM
```

📌 Interpretation:
- Strong only at 1-day horizon  
- Rapid decay across longer windows  
- Not stable over time  

➡️ This is **conditional / regime-dependent behavior**

---

## 🧩 Supported Signal Families

- Mean Reversion  
- Momentum  
- Volatility Clustering  
- Moving Average Distance  

Each family is tested across multiple parameter variations (e.g. lookbacks).

---

## 📊 Statistical Approach

For each signal variation:

- Linear regression (OLS)
- Coefficient (direction & strength)
- P-value (statistical significance)
- R² (explanatory power)

Signals are evaluated based on:

- Direction consistency  
- Significance ratio  
- Effect size  
- Cross-symbol validation  

---

## 🧪 Cross-Symbol Validation

The system tests signals across multiple assets:

```
AAPL → PROMOTE
MSFT → PROMOTE
SPY  → PROMOTE
```

Then aggregates:

```
3/3 → PROMOTE → Strong across market
```

📌 This answers:

> Is this a **real market behavior**, or just noise in one stock?

---

## 🧠 Why This Matters

Many signals look good because:

- They are overfit  
- They only work in one time period  
- They only work on one asset  

This system explicitly tests:

✔ Stability across time  
✔ Robustness across parameters  
✔ Generalization across assets  

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt
```

```bash
python cli.py --symbols AAPL MSFT SPY --family volatility_clustering
```

---

## 📁 Output

Results are saved to:

```
reports/
  ├── tables/
  ├── figures/
  └── *.json
```

---

## 🧠 Summary

This project shifts the focus from:

```
“finding strategies”
```

to:

```
validating ideas
```

---

## 📌 Final Thought

> A signal that doesn’t survive time and across assets is not a signal — it’s noise.
