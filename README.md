# Mini Research Lab

A lightweight framework to practice **describe tables** and **single linear regression summaries** on financial data.

Goal:
- Build intuition, not just run models
- Learn how tables can **mislead you**
- Repeat the same analysis pattern across multiple ideas

---

## What this project is

A small environment to run many **mini experiments** of the form:

Y = α + βX + ε

Where:
- X = one signal (recent return, volatility, distance from mean…)
- Y = future outcome (next return, next volatility…)

---

## Core Learning Focus

This project is NOT about:
- finding alpha immediately
- building production models

This project IS about:
- reading `describe()` deeply
- interpreting regression summaries correctly
- spotting traps:
  - significant but useless
  - high R² but unstable
  - skewed distributions
  - outliers dominating std

---

## Project Structure
mini_research_lab/
├── data/ # raw and processed data
├── notebooks/ # one idea per notebook
├── reports/ # saved figures and tables
├── src/mini_research_lab/
│ ├── data_loader.py
│ ├── features.py
│ ├── experiment_specs.py
│ ├── summaries.py
│ ├── plotting.py
│ └── lab.py


---

## Setup

```bash
# Install in editable mode (recommended for development)
pip install -e .

# Or install directly
pip install mini-research-lab

from mini_research_lab import (
    download_prices,
    add_return_features,
    MiniResearchLab,
    default_experiments,
    plot_experiment_bundle,
)

prices = download_prices("AAPL", start="2018-01-01")
df = add_return_features(prices)

lab = MiniResearchLab(df)

spec = default_experiments()[0]

result = lab.run_experiment(spec.x_col, spec.y_col)

print(result["x_describe"].to_dict())
print(result["y_describe"].to_dict())
print(result["regression"].to_dict())

print(result["model"].summary())

plot_experiment_bundle(df, spec.x_col, spec.y_col, title_prefix=spec.name)

