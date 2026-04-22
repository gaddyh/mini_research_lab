from pathlib import Path
import pandas as pd

from src.mini_research_lab import (
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

# Save table summaries to reports/tables
tables_dir = Path("reports/tables")
tables_dir.mkdir(parents=True, exist_ok=True)

# Save x variable summary
x_summary = pd.DataFrame([result["x_describe"].to_dict()]).T
x_summary.to_csv(tables_dir / f"{spec.name}_x_summary.csv")

# Save y variable summary
y_summary = pd.DataFrame([result["y_describe"].to_dict()]).T
y_summary.to_csv(tables_dir / f"{spec.name}_y_summary.csv")

# Save regression summary
reg_summary = pd.DataFrame([result["regression"].to_dict()]).T
reg_summary.to_csv(tables_dir / f"{spec.name}_regression.csv")

# Save full model summary as text
with open(tables_dir / f"{spec.name}_model_summary.txt", "w") as f:
    f.write(str(result["model"].summary()))

print(result["x_describe"].to_dict())
print(result["y_describe"].to_dict())
print(result["regression"].to_dict())
print(result["model"].summary())

# Save plots to reports/figures
plot_experiment_bundle(df, spec.x_col, spec.y_col, title_prefix=spec.name)