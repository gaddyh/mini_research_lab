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

# Save table summaries to reports/tables/{spec.name}
tables_dir = Path("reports/tables") / spec.name
tables_dir.mkdir(parents=True, exist_ok=True)

# Save X summary to separate file
with open(tables_dir / "x_summary.txt", "w") as f:
    f.write("=== X Variable Summary ===\n")
    x_dict = result["x_describe"].to_dict()
    for key, value in x_dict.items():
        f.write(f"{key}: {value}\n")

# Save Y summary to separate file
with open(tables_dir / "y_summary.txt", "w") as f:
    f.write("=== Y Variable Summary ===\n")
    y_dict = result["y_describe"].to_dict()
    for key, value in y_dict.items():
        f.write(f"{key}: {value}\n")

# Save regression summary to separate file
with open(tables_dir / "regression.txt", "w") as f:
    f.write("=== Regression Summary ===\n")
    reg_dict = result["regression"].to_dict()
    for key, value in reg_dict.items():
        f.write(f"{key}: {value}\n")

# Save full model summary as text
with open(tables_dir / "model_summary.txt", "w") as f:
    f.write(str(result["model"].summary()))

print(result["x_describe"].to_dict())
print(result["y_describe"].to_dict())
print(result["regression"].to_dict())
print(result["model"].summary())

# Save plots to reports/figures/{spec.name}
figures_dir = Path("reports/figures") / spec.name
saved_files = plot_experiment_bundle(
    df, spec.x_col, spec.y_col, 
    title_prefix="",  # No prefix since we're using folder structure
    output_dir=figures_dir
)

print(f"Results saved to:")
print(f"  Tables: {tables_dir}")
print(f"  Figures: {figures_dir}")