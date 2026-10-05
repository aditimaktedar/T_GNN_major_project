"""Utilities for building temporal drug-drug interaction graphs."""

from typing import List, Tuple, Dict, Any, Optional


def build_edge_index_from_pairs(
    pairs: List[Tuple[int, int]],
    bidirectional: bool = True
) -> List[Tuple[int, int]]:
    """Converts a list of drug pair indices into a graph edge index list."""
    edges = set()
    for u, v in pairs:
        edges.add((u, v))
        if bidirectional:
            edges.add((v, u))
    return sorted(list(edges))


def build_patient_regimen_graph(
    drug_indices: List[int],
    include_self_loops: bool = False
) -> List[Tuple[int, int]]:
    """
    Builds a complete co-medication graph between all drugs in a patient's active regimen.
    """
    edges = []
    n = len(drug_indices)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = drug_indices[i], drug_indices[j]
            edges.append((u, v))
            edges.append((v, u))
        if include_self_loops:
            u = drug_indices[i]
            edges.append((u, u))
    return edges


def build_temporal_regimen_graph(
    prescriptions: List[Dict[str, Any]],
    drug_to_idx: Dict[str, int],
    window_hours: float = 24.0
) -> List[Tuple[int, int]]:
    """
    Builds edges between drugs co-administered within a given temporal window.
    Each prescription dict is expected to contain 'drug_id' and 'timestamp_hours'.
    """
    valid = []
    for rx in prescriptions:
        did = str(rx.get("drug_id", ""))
        ts = float(rx.get("timestamp_hours", 0.0))
        if did in drug_to_idx:
            valid.append((drug_to_idx[did], ts))

    edges = set()
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            u, t_u = valid[i]
            v, t_v = valid[j]
            if abs(t_u - t_v) <= window_hours:
                edges.add((u, v))
                edges.add((v, u))

    return sorted(list(edges))


def validate_edge_index(
    edge_index: List[Tuple[int, int]],
    num_nodes: int
) -> bool:
    """Verifies that all edge endpoint indices are strictly within [0, num_nodes - 1]."""
    for src, dst in edge_index:
        if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
            return False
    return True
