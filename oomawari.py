from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path


DATA = Path(__file__).parent / "data" / "tokyo_near_zone.csv"


@dataclass(frozen=True)
class Edge:
    to: str
    km: float
    line: str


def load_graph(path: Path = DATA) -> dict[str, list[Edge]]:
    graph: dict[str, list[Edge]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if len(row) != 4:
                raise ValueError(f"invalid row: {row!r}")
            a, b, km, line = row
            # The source dataset prefixes duplicate station names with a route hint,
            # e.g. (武蔵)三郷. User-facing station names should remain plain.
            a = a.replace("(武蔵)", "").replace("(横)", "").replace("(岸)", "").replace("(中)", "").replace("(川)", "").replace("(篠)", "").replace("(北)", "").replace("(成)", "").replace("(両)", "").replace("(烏)", "").replace("(信)", "").replace("(房)", "").replace("(総)", "").replace("(臨)", "")
            b = b.replace("(武蔵)", "").replace("(横)", "").replace("(岸)", "").replace("(中)", "").replace("(川)", "").replace("(篠)", "").replace("(北)", "").replace("(成)", "").replace("(両)", "").replace("(烏)", "").replace("(信)", "").replace("(房)", "").replace("(総)", "").replace("(臨)", "")
            graph.setdefault(a, []).append(Edge(b, float(km), line))
            graph.setdefault(b, []).append(Edge(a, float(km), line))
    return graph


def _upper_bound(graph, current: str, target: str, visited: set[str], distance: float) -> float:
    # Relaxation: a simple path uses degree <= 2 at every remaining vertex.
    # Taking the two heaviest unused incident edges at every vertex and
    # dividing their total by two is therefore an admissible upper bound.
    remaining = set(graph) - visited
    remaining.add(current)
    total = distance
    for v in remaining:
        weights = [
            e.km for e in graph[v]
            if e.to in remaining and not (v == current and e.to in visited)
        ]
        weights.sort(reverse=True)
        total += sum(weights[:2]) / 2.0
    # The target must still be reachable from the current state.
    if target not in remaining:
        return float("-inf")
    return total


def longest_simple_path(start: str, target: str, graph: dict[str, list[Edge]]) -> tuple[float, list[tuple[str, str, float, str]]]:
    if start not in graph:
        raise KeyError(f"unknown station: {start}")
    if target not in graph:
        raise KeyError(f"unknown station: {target}")
    if start == target:
        return 0.0, []

    # A cheap initial solution gives branch-and-bound a useful threshold.
    def shortest_path() -> tuple[float, list[str]]:
        pq = [(0.0, start, [start])]
        best = {start: 0.0}
        while pq:
            d, v, path = heapq.heappop(pq)
            if v == target:
                return d, path
            if d != best[v]:
                continue
            for e in graph[v]:
                nd = d + e.km
                if nd < best.get(e.to, float("inf")):
                    best[e.to] = nd
                    heapq.heappush(pq, (nd, e.to, path + [e.to]))
        raise ValueError("stations are disconnected")

    best_distance, best_nodes = shortest_path()
    best_edges: list[tuple[str, str, float, str]] = []
    for a, b in zip(best_nodes, best_nodes[1:]):
        edge = next(e for e in graph[a] if e.to == b)
        best_edges.append((a, b, edge.km, edge.line))

    visited = {start}
    path_nodes = [start]

    # Long edges first usually finds a strong solution early.
    def dfs(v: str, distance: float) -> None:
        nonlocal best_distance, best_edges
        if v == target:
            if distance > best_distance:
                best_distance = distance
                best_edges = []
                for a, b in zip(path_nodes, path_nodes[1:]):
                    e = next(e for e in graph[a] if e.to == b)
                    best_edges.append((a, b, e.km, e.line))
            return

        if _upper_bound(graph, v, target, visited, distance) <= best_distance:
            return

        candidates = [e for e in graph[v] if e.to not in visited]
        candidates.sort(key=lambda e: e.km, reverse=True)

        for e in candidates:
            visited.add(e.to)
            path_nodes.append(e.to)
            dfs(e.to, distance + e.km)
            path_nodes.pop()
            visited.remove(e.to)

    dfs(start, 0.0)
    return best_distance, best_edges


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

    graph = load_graph(args.data)
    distance, edges = longest_simple_path(args.start, args.target, graph)
    print(format_result(args.start, args.target, distance, edges))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
