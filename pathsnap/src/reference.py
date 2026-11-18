"""Create candidate labels from the ordered reference route."""

import numpy as np


def labels(candidates, route, max_forward=50):
    """Label the nearest reachable reference-road candidate for each point.

    ``route`` is ordered. A candidate is eligible when its arc occurs at or
    after the last matched route position and no more than ``max_forward``
    route steps ahead. Candidates are considered by point and distance, so
    the first eligible candidate is the nearest one. Points without an
    eligible candidate receive no positive label and count as candidate
    misses in feature preparation.

    The returned integer array has one entry per input candidate row.
    """
    route_positions = {}
    for position, arc_id in enumerate(route):
        route_positions.setdefault(int(arc_id), []).append(position)

    ordered = candidates.assign(_row=np.arange(len(candidates))).sort_values(
        ["pt", "dist"], kind="stable"
    )
    y = np.zeros(len(candidates), dtype=np.int8)
    route_cursor = 0

    for _, point_candidates in ordered.groupby("pt", sort=True):
        for row, arc_id in zip(point_candidates["_row"], point_candidates["arc_id"]):
            positions = route_positions.get(int(arc_id), ())
            match = next(
                (pos for pos in positions
                 if route_cursor <= pos <= route_cursor + max_forward),
                None,
            )
            if match is not None:
                y[row] = 1
                route_cursor = match
                break

    return y
