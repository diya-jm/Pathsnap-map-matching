"""Build the ten-feature Review 2 matrices and trip-grouped fold manifest."""
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from pathsnap import FEATURES

CACHE = Path(".cache")
RESULTS = Path("results")
N_SPLITS = 3


def _load_split(split):
    files = sorted((CACHE / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No prepared {split} trips found under {CACHE / split}; run `python prepare.py` first."
        )
    xs, ys, group_parts, trip_rows = [], [], [], {}
    for file in files:
        with np.load(file) as saved:
            x = saved["x"]
            ids = saved["ids"]
            aligned_route = saved["y"]
        if x.ndim != 3 or x.shape[-1] != len(FEATURES):
            raise ValueError(
                f"{file.stem}: expected candidate features shaped (points, candidates, "
                f"{len(FEATURES)}), got {x.shape}"
            )
        candidate_y = (ids == aligned_route[:, None]).astype(np.uint8)
        if candidate_y.shape != x.shape[:2]:
            raise ValueError(f"{file.stem}: candidate labels do not match candidate rows")
        flat_x = x.reshape(-1, len(FEATURES))
        flat_y = candidate_y.reshape(-1)
        xs.append(flat_x)
        ys.append(flat_y)
        group_parts.append(np.full(len(flat_y), file.stem, dtype="U8"))
        trip_rows[file.stem] = len(flat_y)

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(group_parts), trip_rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build():
    RESULTS.mkdir(exist_ok=True)
    X, y, groups, trip_rows = _load_split("train")
    test_files = sorted((CACHE / "student_test").glob("*.npz"))
    if len(trip_rows) != 80 or len(test_files) != 10:
        raise ValueError(
            f"Expected 80 training and 10 student-test trips; found {len(trip_rows)} and {len(test_files)}"
        )

    dataset_rows = [dict(
        split="train", trips=len(trip_rows), candidate_rows=len(y),
        feature_columns=X.shape[1], positive_labels=int(y.sum()),
        negative_labels=int(len(y) - y.sum()), group_count=len(np.unique(groups)),
        included_in_cv="yes",
    ), dict(
        split="student_test", trips=len(test_files), candidate_rows="",
        feature_columns=len(FEATURES), positive_labels="", negative_labels="",
        group_count=len(test_files), included_in_cv="no; held separate",
    )]
    _write_csv(RESULTS / "review2_dataset_summary.csv", dataset_rows)

    fold_assignments = {}
    fold_summary = []
    for fold, (fit_idx, val_idx) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(X, y, groups=groups), start=1
    ):
        fit_ids = set(np.unique(groups[fit_idx]).tolist())
        val_ids = set(np.unique(groups[val_idx]).tolist())
        overlap = fit_ids & val_ids
        if overlap:
            raise ValueError(f"Fold {fold} has trip IDs in both fit and validation: {sorted(overlap)}")
        for trip_id in val_ids:
            fold_assignments[trip_id] = fold
        fold_summary.append(dict(
            fold=fold, fit_trips=len(fit_ids), validation_trips=len(val_ids),
            fit_candidate_rows=len(fit_idx),
            validation_candidate_rows=len(val_idx),
            overlapping_trip_ids=len(overlap),
        ))

    if set(fold_assignments) != set(trip_rows):
        raise ValueError("Fold assignment did not cover every training trip exactly once")
    _write_csv(RESULTS / "review2_fold_assignments.csv", [
        dict(trip=trip_id, validation_fold=fold_assignments[trip_id],
             candidate_rows=trip_rows[trip_id])
        for trip_id in sorted(trip_rows)
    ])
    _write_csv(RESULTS / "review2_fold_summary.csv", fold_summary)

    print(f"FEATURES ({len(FEATURES)}): {', '.join(FEATURES)}")
    print(f"X shape: {X.shape}; y shape: {y.shape}; groups shape: {groups.shape}")
    print(f"Positive labels: {int(y.sum())}; negative labels: {int(len(y) - y.sum())}")
    print(f"Training trip groups: {len(np.unique(groups))}; student-test trips held separate: {len(test_files)}")
    for row in fold_summary:
        print(
            f"Fold {row['fold']}: fit={row['fit_trips']} trips, "
            f"validation={row['validation_trips']} trips, "
            f"overlap={row['overlapping_trip_ids']}"
        )
    return X, y, groups, fold_assignments


if __name__ == "__main__":
    build()
