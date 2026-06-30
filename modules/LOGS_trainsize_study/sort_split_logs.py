from pathlib import Path
import pandas as pd
import sys

ROOT_SUBFOLDER = "."
TARGET_NAME = "split_log.csv"
SPLIT_ORDER = ["train", "val", "test"]


def sort_one_csv(csv_path: Path):
    df = pd.read_csv(csv_path)

    required = {"sample_idx", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}")

    df["split"] = pd.Categorical(df["split"], categories=SPLIT_ORDER, ordered=True)
    df = df.sort_values(["split", "sample_idx"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)


def main():
    start_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root = start_dir / ROOT_SUBFOLDER

    if not root.exists() or not root.is_dir():
        print(f"Folder not found: {root}")
        sys.exit(1)

    csv_files = sorted(root.rglob(TARGET_NAME))

    if not csv_files:
        print(f'No files named "{TARGET_NAME}" found under {root}')
        return

    print(f"Found {len(csv_files)} file(s).")
    for csv_file in csv_files:
        try:
            sort_one_csv(csv_file)
            print(f"Sorted: {csv_file}")
        except Exception as e:
            print(f"Skipped: {csv_file} -> {e}")


if __name__ == "__main__":
    main()
