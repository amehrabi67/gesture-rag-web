from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KNN model from point_history.csv.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=7)
    args = parser.parse_args()

    x_rows, y_rows = [], []
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y_rows.append(int(row[0]))
            x_rows.append([float(v) for v in row[1:]])

    clf = KNeighborsClassifier(n_neighbors=args.k, weights="distance")
    clf.fit(np.asarray(x_rows, np.float32), np.asarray(y_rows, np.int32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved {args.out} from {len(y_rows)} rows")


if __name__ == "__main__":
    main()
