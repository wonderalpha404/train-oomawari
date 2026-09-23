from __future__ import annotations

import argparse
import csv
from pathlib import Path

DATA = Path(__file__).parent / "data" / "tokyo_near_zone.csv"


class Edge:
    __slots__ = ("u", "v", "km", "line")

    def __init__(self, u: str, v: str, km: float, line: str) -> None:
        self.u = u
        self.v = v
        self.km = km
        self.line = line


_PREFIXES = (
    "(武蔵)", "(横)", "(岸)", "(中)", "(川)", "(篠)", "(北)",
    "(成)", "(両)", "(烏)", "(信)", "(房)", "(総)", "(臨)",
    "（臨）",
)


def clean_station(name: str) -> str:
    for prefix in _PREFIXES:
        name = name.replace(prefix, "")
    return name


def load_graph(path: Path = DATA) -> tuple[list[str], list[Edge], dict[str, int]]:
    edges: list[Edge] = []
    names: set[str] = set()

    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 4:
                raise ValueError(f"invalid row: {row!r}")
            a, b, km, line = row
            a = clean_station(a)
            b = clean_station(b)
            edges.append(Edge(a, b, float(km), line))
            names.add(a)
            names.add(b)

    stations = sorted(names)
    index = {name: i for i, name in enumerate(stations)}
    return stations, edges, index


def longest_simple_path(
    start: str, target: str, stations: list[str], edges: list[Edge], index: dict[str, int]
) -> tuple[float, list[tuple[str, str, float, str]]]:
    """Solve the exact maximum-weight simple s-t path as a MILP.

    y(u,v) is a binary directed-edge variable.  Each visited internal
    station has exactly one incoming and one outgoing selected arc;
    start has only one outgoing and target only one incoming.

    A commodity-flow constraint forces all selected vertices to belong
    to the same s-t path, excluding disconnected cycles.
    """
    if start not in index:
        raise KeyError(f"unknown station: {start}")
    if target not in index:
        raise KeyError(f"unknown station: {target}")
    if start == target:
        return 0.0, []

    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n = len(stations)
    m = len(edges)
    s = index[start]
    t = index[target]

    # Two directed orientations per undirected railway section.
    D = 2 * m
    z0 = D
    g0 = D + n
    N = D + n + D

    def yidx(ei: int, direction: int) -> int:
        return 2 * ei + direction

    def gidx(ei: int, direction: int) -> int:
        return g0 + 2 * ei + direction

    # Directed incidence lists.
    incoming: list[list[int]] = [[] for _ in range(n)]
    outgoing: list[list[int]] = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        u = index[e.u]
        v = index[e.v]
        a = yidx(i, 0)      # u -> v
        b = yidx(i, 1)      # v -> u
        outgoing[u].append(a)
        incoming[v].append(a)
        outgoing[v].append(b)
        incoming[u].append(b)

    # Sparse linear constraints.
    # 1: each physical section can be used in at most one direction.
    # 2: visited vertices have exactly one incoming/outgoing arc, except s/t.
    # 3: one-unit-per-visited-vertex commodity flow.
    # 4: flow capacity g <= (n-1)y.
    rows = 2 * m + 2 * n + n + 2 * m
    A = lil_matrix((rows, N), dtype=float)
    lb = np.full(rows, -np.inf)
    ub = np.full(rows, np.inf)
    r = 0

    for i in range(m):
        A[r, yidx(i, 0)] = 1
        A[r, yidx(i, 1)] = 1
        ub[r] = 1
        r += 1

    for v in range(n):
        # Outgoing selected arc count.
        for a in outgoing[v]:
            A[r, a] = 1
        if v not in (t,):
            A[r, z0 + v] = -1
            lb[r] = ub[r] = 0
        else:
            lb[r] = ub[r] = 0
        r += 1

        # Incoming selected arc count.
        for a in incoming[v]:
            A[r, a] = 1
        if v not in (s,):
            A[r, z0 + v] = -1
            lb[r] = ub[r] = 0
        else:
            lb[r] = ub[r] = 0
        r += 1

    # Commodity flow: source sends one unit to every other visited vertex.
    for v in range(n):
        if v == s:
            for a in outgoing[v]:
                A[r, g0 + a] = 1
            for a in incoming[v]:
                A[r, g0 + a] = -1
            for w in range(n):
                if w != s:
                    A[r, z0 + w] = -1
            lb[r] = ub[r] = 0
        else:
            for a in incoming[v]:
                A[r, g0 + a] = 1
            for a in outgoing[v]:
                A[r, g0 + a] = -1
            A[r, z0 + v] = -1
            lb[r] = ub[r] = 0
        r += 1

    M = n - 1
    for i in range(m):
        for direction in (0, 1):
            yi = yidx(i, direction)
            gi = gidx(i, direction)
            A[r, gi] = 1
            A[r, yi] = -M
            ub[r] = 0
            r += 1

    assert r == rows

    c = np.zeros(N)
    for i, e in enumerate(edges):
        c[yidx(i, 0)] = -e.km
        c[yidx(i, 1)] = -e.km

    integrality = np.zeros(N)
    integrality[:D] = 1
    integrality[z0:g0] = 1

    lower = np.zeros(N)
    upper = np.full(N, np.inf)
    upper[:D] = 1
    upper[z0:g0] = 1
    upper[g0:] = M

    # Start and target must be visited.
    lower[z0 + s] = upper[z0 + s] = 1
    lower[z0 + t] = upper[z0 + t] = 1

    result = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        options={"time_limit": 300, "mip_rel_gap": 0},
    )

    if not result.success:
        raise RuntimeError(
            f"MILP solver failed: status={result.status}, message={result.message}"
        )

    x = result.x
    selected: dict[int, int] = {}
    for i, e in enumerate(edges):
        for direction in (0, 1):
            if x[yidx(i, direction)] > 0.5:
                u = index[e.u] if direction == 0 else index[e.v]
                v = index[e.v] if direction == 0 else index[e.u]
                selected[u] = v

    route_edges: list[tuple[str, str, float, str]] = []
    current = s
    distance = 0.0
    seen = {s}

    while current != t:
        if current not in selected:
            raise RuntimeError("MILP solution could not be reconstructed as an s-t path")
        nxt = selected[current]
        if nxt in seen:
            raise RuntimeError("MILP reconstruction contains a cycle")
        seen.add(nxt)

        chosen = None
        for e in edges:
            eu = index[e.u]
            ev = index[e.v]
            if (eu == current and ev == nxt) or (eu == nxt and ev == current):
                # The physical edge must be selected in the corresponding direction.
                direction = 0 if eu == current else 1
                if x[yidx(edges.index(e), direction)] > 0.5:
                    chosen = e
                    break
        if chosen is None:
            raise RuntimeError("selected edge was not found during reconstruction")

        route_edges.append((chosen.u if index[chosen.u] == current else chosen.v,
                            chosen.v if index[chosen.u] == current else chosen.u,
                            chosen.km, chosen.line))
        distance += chosen.km
        current = nxt

    return distance, route_edges


def format_result(start: str, target: str, distance: float, edges) -> str:
    stations = [start] + [e[1] for e in edges]
    lines = [
        f"{start} -> {target}",
        f"最大大回り距離: {distance:.1f} km",
        f"駅数: {len(stations)}",
        "経路:",
        " -> ".join(stations),
        "",
        "区間:",
    ]
    for a, b, km, line in edges:
        lines.append(f"  {a} - {b}  {km:.1f} km  [{line}]")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="東京近郊区間の最大大回り経路を探索")
    parser.add_argument("start", help="出発駅")
    parser.add_argument("target", help="到着駅")
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()

    stations, edges, index = load_graph(args.data)
    distance, route_edges = longest_simple_path(
        args.start, args.target, stations, edges, index
    )
    print(format_result(args.start, args.target, distance, route_edges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
