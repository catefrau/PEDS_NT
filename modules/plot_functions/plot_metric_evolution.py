
import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(csv_path=None, csv_text=None):
    if csv_path is not None:
        return pd.read_csv(csv_path)
    if csv_text is not None:
        return pd.read_csv(io.StringIO(csv_text.strip()))
    raise ValueError("Provide either csv_path or csv_text.")


def add_epoch0(df, epoch0_values=None):
    if not epoch0_values:
        return df
    row = {col: None for col in df.columns}
    row['epoch'] = 0
    for k, v in epoch0_values.items():
        if k in row:
            row[k] = v
    return pd.concat([pd.DataFrame([row]), df], ignore_index=True)


def plot_metric_evolution(
    csv_path=None,
    csv_text=None,
    metrics=None,
    x_col='epoch',
    epoch0_values=None,
    layout='subplots',
    output_path='output/metrics_evolution.png',
    title='Metric evolution over epochs',
):
    df = load_csv(csv_path=csv_path, csv_text=csv_text)

    if metrics is None or len(metrics) == 0:
        raise ValueError('Pass at least one metric name.')
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(f'Metrics not found in CSV: {missing}')
    if x_col not in df.columns:
        raise ValueError(f"x_col '{x_col}' not found in CSV.")

    df = add_epoch0(df, epoch0_values=epoch0_values)
    x = df[x_col]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if layout == 'same':
        fig, ax = plt.subplots(figsize=(10, 5))
        for metric in metrics:
            ax.plot(x, df[metric], marker='o', linewidth=2, label=metric)
        ax.set_xlabel(x_col)
        ax.set_ylabel('value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)), sharex=True)
        if len(metrics) == 1:
            axes = [axes]
        for ax, metric in zip(axes, metrics):
            ax.plot(x, df[metric], marker='o', linewidth=2)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            best_idx = df[metric].idxmin()
            if pd.notna(df.loc[best_idx, metric]):
                ax.scatter(df.loc[best_idx, x_col], df.loc[best_idx, metric], s=60)
                ax.annotate(
                    f"best: {df.loc[best_idx, metric]:.4g} @ {x_col}={df.loc[best_idx, x_col]}",
                    (df.loc[best_idx, x_col], df.loc[best_idx, metric]),
                    textcoords='offset points',
                    xytext=(8, 8),
                )
            ax.set_title(metric)
        axes[-1].set_xlabel(x_col)
        fig.suptitle(title)
        fig.tight_layout()

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot any metric evolution from a training CSV.')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--metrics', nargs='+', required=True, help='Columns to plot')
    parser.add_argument('--xcol', type=str, default='epoch', help='Column for x-axis')
    parser.add_argument('--layout', type=str, default='subplots', choices=['subplots', 'same'])
    parser.add_argument('--output', type=str, default='output/metrics_evolution.png')
    args = parser.parse_args()

    plot_metric_evolution(
        csv_path=args.csv,
        metrics=args.metrics,
        x_col=args.xcol,
        layout=args.layout,
        output_path=args.output,
    )
