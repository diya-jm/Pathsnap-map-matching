"""Chiranthan V | Review 1: inspect candidate-road features on a training trip.

From the repository's pathsnap directory: python review1_features.py
"""
import numpy as np

from src.features import FEATURE_COLUMNS, R, add_features, candidates, crop_arcs, project
from src.load import load_trip


def main():
    trip_id = "00000001"
    track, nodes, arcs, _ = load_trip("train", trip_id)
    gps_xy, nodes_xy = project(track, nodes)
    nearby_arcs = crop_arcs(gps_xy, nodes_xy, arcs)
    roads = candidates(gps_xy, nodes_xy, arcs, nearby_arcs)
    rows = add_features(roads, gps_xy, track.t.to_numpy())
    if not np.isfinite(rows[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite feature in the training trip")

    print(f"Training trip {trip_id}: {len(track)} observations, {R:g} m candidate radius")
    print(f"Feature matrix: {len(rows)} candidate rows x {len(FEATURE_COLUMNS)} columns")
    print("Columns:", ", ".join(FEATURE_COLUMNS))
    print("\nObservation 0: candidates ordered by distance (metres)")
    first = rows.loc[rows.pt == 0].sort_values(["dist", "arc_id"])
    print(first[["arc_id", *FEATURE_COLUMNS]].round(3).to_string(index=False))

    # Find the same segment in both directions at one moving observation.
    # Selecting a pair for illustration does not use the reference route.
    cols = ["pt", "arc_id", "u", "v", "dist", "cos_h", "t"]
    moving = rows.loc[(rows.speed > 1) & (rows.u != rows.v), cols]
    forward = moving.loc[moving.cos_h > 0.9]
    reverse = moving.loc[moving.cos_h < -0.9]
    pairs = forward.merge(reverse, left_on=["pt", "u", "v"],
                          right_on=["pt", "v", "u"], suffixes=("_a", "_b"))
    if pairs.empty:
        print("\nNo opposite-direction example met the heading and speed criteria.")
        return
    pair = pairs.sort_values(["pt", "dist_a", "arc_id_a", "arc_id_b"]).iloc[0]
    print(f"\nDirection example at observation {int(pair.pt)}:")
    for side in ["a", "b"]:
        print(f"  Arc {int(pair['arc_id_' + side])}: "
              f"{pair['dist_' + side]:.2f} m away, "
              f"heading cosine {pair['cos_h_' + side]:+.3f}, "
              f"fraction along {pair['t_' + side]:.3f}")


if __name__ == "__main__":
    main()
