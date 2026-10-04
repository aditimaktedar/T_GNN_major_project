"""Tests for temporal graph construction and utilities."""

from src.graph.graph_builder import (
    build_edge_index_from_pairs,
    build_patient_regimen_graph,
    build_temporal_regimen_graph,
    validate_edge_index
)


def test_build_edge_index_from_pairs():
    pairs = [(0, 1), (2, 3)]
    edges = build_edge_index_from_pairs(pairs, bidirectional=True)
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (2, 3) in edges
    assert (3, 2) in edges
    assert len(edges) == 4


def test_build_patient_regimen_graph():
    meds = [5, 10, 15]
    edges = build_patient_regimen_graph(meds, include_self_loops=True)
    # 3 pairs * 2 directions = 6 edges + 3 self loops = 9
    assert len(edges) == 9
    assert (5, 10) in edges
    assert (10, 5) in edges
    assert (5, 5) in edges


def test_temporal_regimen_graph():
    drug_to_idx = {"A": 0, "B": 1, "C": 2}
    prescriptions = [
        {"drug_id": "A", "timestamp_hours": 0.0},
        {"drug_id": "B", "timestamp_hours": 12.0},
        {"drug_id": "C", "timestamp_hours": 48.0},
    ]
    edges = build_temporal_regimen_graph(prescriptions, drug_to_idx, window_hours=24.0)
    # A and B are within 12h <= 24h
    # C is 48h (36h from B, 48h from A) -> no edge to C
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert len(edges) == 2


def test_validate_edge_index():
    assert validate_edge_index([(0, 1), (1, 2)], num_nodes=3) is True
    assert validate_edge_index([(0, 3)], num_nodes=3) is False
    assert validate_edge_index([(-1, 0)], num_nodes=3) is False


if __name__ == "__main__":
    test_build_edge_index_from_pairs()
    test_build_patient_regimen_graph()
    test_temporal_regimen_graph()
    test_validate_edge_index()
    print("All graph builder tests passed!")
