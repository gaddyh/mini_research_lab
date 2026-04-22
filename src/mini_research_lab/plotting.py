from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram(series: pd.Series, title: str, bins: int = 50, save_path: str | Path | None = None) -> Path:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty after dropping NaN values.")

    plt.figure(figsize=(8, 4.5))
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(series.name or "value")
    plt.ylabel("count")
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{series.name or 'histogram'}_hist.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_boxplot(series: pd.Series, title: str, save_path: str | Path | None = None) -> Path:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty after dropping NaN values.")

    plt.figure(figsize=(7, 2.8))
    plt.boxplot(s, vert=False)
    plt.title(title)
    plt.xlabel(series.name or "value")
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{series.name or 'boxplot'}_box.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_scatter_with_fit(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    alpha: float = 0.35,
    save_path: str | Path | None = None,
) -> Path:
    temp = df[[x_col, y_col]].dropna().copy()
    if temp.empty:
        raise ValueError("No rows left after dropping NaN values.")

    x = temp[x_col].to_numpy()
    y = temp[y_col].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(7.5, 5))
    plt.scatter(x, y, alpha=alpha)
    plt.plot(x_line, y_line)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{x_col}_vs_{y_col}_scatter.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_series(
    series: pd.Series,
    title: str,
    save_path: str | Path | None = None,
) -> Path:
    s = series.dropna()
    if s.empty:
        raise ValueError("Series is empty after dropping NaN values.")

    plt.figure(figsize=(10, 4.5))
    plt.plot(s.index, s.values)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel(series.name or "value")
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{series.name or 'series'}_plot.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


def plot_experiment_bundle(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title_prefix: str = "",
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    prefix = f"{title_prefix} — " if title_prefix else ""
    output_dir = Path(output_dir)
    
    saved_files = []
    
    # X variable plots
    saved_files.append(plot_histogram(
        df[x_col], 
        f"{prefix}{x_col} histogram", 
        save_path=output_dir / f"{prefix}{x_col}_histogram.png"
    ))
    saved_files.append(plot_boxplot(
        df[x_col], 
        f"{prefix}{x_col} boxplot", 
        save_path=output_dir / f"{prefix}{x_col}_boxplot.png"
    ))

    # Y variable plots
    saved_files.append(plot_histogram(
        df[y_col], 
        f"{prefix}{y_col} histogram", 
        save_path=output_dir / f"{prefix}{y_col}_histogram.png"
    ))
    saved_files.append(plot_boxplot(
        df[y_col], 
        f"{prefix}{y_col} boxplot", 
        save_path=output_dir / f"{prefix}{y_col}_boxplot.png"
    ))

    # Scatter plot
    saved_files.append(plot_scatter_with_fit(
        df=df,
        x_col=x_col,
        y_col=y_col,
        title=f"{prefix}{x_col} vs {y_col}",
        save_path=output_dir / f"{prefix}{x_col}_vs_{y_col}_scatter.png"
    ))
    
    return saved_files