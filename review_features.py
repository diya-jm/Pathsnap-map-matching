"""Show how raw GPS and road geometry become features for Review 1."""
from pathlib import Path

import numpy as np

from pathsnap import FEATURES, candidates, read_trip


def main():
    trip_id = "00000001"
    folder = Path("data") / "Student folder" / "train" / trip_id
    trip = read_trip(folder)
    ids, features = candidates(trip)
    print(f"PathSnap | Feature engineering | Training trip {trip_id}")
    print(f"Shape: {features.shape} (observations, candidate roads, features)")
    print("Features:", ", ".join(FEATURES))
    lon, lat, seconds = trip["gps"][0]
    print(f"\nObservation 0: longitude {lon:.6f}, latitude {lat:.6f}, time {seconds:g} s")
    print(f"Heading available: {bool(features[0, 0, FEATURES.index('heading_available')])}")
    print(f"{'Road ID':>8} {'From':>8} {'To':>8} {'Distance (m)':>14} {'Heading cosine':>16} {'Fraction along':>16}")
    for arc, row in zip(ids[0], features[0]):
        start, end = trip["arcs"][arc]
        print(f"{arc:8d} {start:8d} {end:8d} {row[0]:14.2f} {row[2]:16.3f} {row[5]:16.3f}")

    cache = Path(".cache/train") / f"{trip_id}.npz"
    if cache.is_file():
        with np.load(cache) as saved:
            np.testing.assert_array_equal(ids, saved["ids"])
            np.testing.assert_allclose(features, saved["x"])
        print("\nVerified: recomputed candidates and features match the prepared trip.")


if __name__ == "__main__":
    main()
