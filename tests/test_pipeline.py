"""Unit tests for the Member B pipeline using tiny synthetic fixtures only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.build_dataset import build_ml_dataset
from src.data.clean_mimic import clean_prescriptions
from src.data.drug_pairs import canonicalize_pair, construct_drug_pairs
from src.data.exceptions import ConfigurationError, MissingInputError
from src.data.labels import integrate_labels
from src.data.mimic import load_prescriptions
from src.data.normalize_drugs import normalize_drugs, prepare_drugs_pending_rxnorm
from src.data.splits import split_patients, validate_patient_splits
from src.data.twosides import load_twosides
from src.features.pubchem import retrieve_smiles
from src.features.rdkit_features import featurize_smiles, morgan_fingerprint, parse_smiles


def test_pair_canonicalization_and_self_pair_removal() -> None:
    assert canonicalize_pair("b", "a") == ("a", "b")
    assert canonicalize_pair("a", "a") is None
    assert canonicalize_pair("", "a") is None


def test_duplicate_pair_removal(tmp_path: Path) -> None:
    prescriptions = pd.DataFrame(
        {
            "patient_id": [1, 1, 1],
            "drug_name_norm": ["alpha", "beta", "alpha"],
            "start_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "end_time": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-01-09"]),
        }
    )
    pairs, stats = construct_drug_pairs(
        prescriptions,
        pair_rule="interval_overlap",
        output_path=tmp_path / "pairs.parquet",
        stats_path=tmp_path / "pairs.json",
    )
    assert list(pairs["pair_key"].unique()) == ["alpha||beta"]
    assert stats["n_pairs"] == 1
    assert stats["duplicate_pair_records_removed"] >= 0


def test_same_admission_rule_and_missing_rule_error(tmp_path: Path) -> None:
    prescriptions = pd.DataFrame(
        {
            "patient_id": [1, 1, 1],
            "admission_id": [10, 10, 11],
            "drug_name_norm": ["alpha", "beta", "gamma"],
        }
    )
    pairs, _ = construct_drug_pairs(
        prescriptions,
        pair_rule="same_admission",
        output_path=tmp_path / "pairs.parquet",
        stats_path=tmp_path / "pairs.json",
    )
    assert len(pairs) == 1
    assert pairs.iloc[0]["pair_key"] == "alpha||beta"
    with pytest.raises(ConfigurationError):
        construct_drug_pairs(
            pd.DataFrame({"patient_id": [1], "drug_name_norm": ["alpha"]}),
            pair_rule="interval_overlap",
            output_path=tmp_path / "bad.parquet",
            stats_path=tmp_path / "bad.json",
        )


def test_normalization_and_unmapped_drugs(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.csv"
    mapping.write_text("source_drug,rxnorm_id\nalpha,100\n", encoding="utf-8")
    prescriptions = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "drug_name_raw": ["Alpha", "Beta"],
            "drug_name_norm": ["alpha", "beta"],
        }
    )
    normalized, stats = normalize_drugs(
        prescriptions,
        mapping_path=mapping,
        output_path=tmp_path / "norm.parquet",
        stats_path=tmp_path / "norm.json",
    )
    assert stats["mapped_unique_drugs"] == 1
    assert stats["unmapped_unique_drugs"] == 1
    assert "beta" in stats["unmapped_drugs"]
    assert pd.isna(normalized.loc[normalized["drug_name_norm"] == "beta", "rxnorm_id"].iloc[0])
    with pytest.raises(MissingInputError):
        normalize_drugs(prescriptions, mapping_path=tmp_path / "missing.csv")


def test_label_joining_unknown_not_automatic_negative(tmp_path: Path) -> None:
    mimic_pairs = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "drug_a": ["a", "a"],
            "drug_b": ["b", "c"],
            "pair_key": ["a||b", "a||c"],
        }
    )
    ddi = pd.DataFrame({"pair_key": ["a||b"], "source": ["twosides"]})
    labeled, stats = integrate_labels(
        mimic_pairs,
        ddi,
        negative_mode="none",
        output_path=tmp_path / "labels.parquet",
        stats_path=tmp_path / "labels.json",
    )
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 0
    assert stats["unknown_unmatched_count"] == 1
    assert set(labeled["label"]) == {"positive", "unknown"}


def test_pubchem_cache_uses_mocked_responses(tmp_path: Path) -> None:
    calls = []

    class FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "PropertyTable": {
                    "Properties": [{"CID": 702, "CanonicalSMILES": "CCO", "Title": "ethanol"}]
                }
            }

    def fake_fetch(url: str, timeout: int) -> FakeResponse:
        calls.append(url)
        return FakeResponse()

    first, _ = retrieve_smiles(
        ["ethanol"],
        cache_dir=tmp_path / "pubchem",
        output_path=tmp_path / "chem1.parquet",
        stats_path=tmp_path / "chem1.json",
        fetch_fn=fake_fetch,
        pause_seconds=0,
    )
    second, stats = retrieve_smiles(
        ["ethanol"],
        cache_dir=tmp_path / "pubchem",
        output_path=tmp_path / "chem2.parquet",
        stats_path=tmp_path / "chem2.json",
        fetch_fn=fake_fetch,
        pause_seconds=0,
    )
    assert first.iloc[0]["smiles"] == "CCO"
    assert second.iloc[0]["smiles"] == "CCO"
    assert stats["n_fetched"] == 0
    assert stats["n_cache_hits"] == 1
    assert len(calls) == 1


def test_invalid_smiles_and_fingerprint_determinism(tmp_path: Path) -> None:
    chemicals = pd.DataFrame(
        {
            "query": ["ethanol", "bad", "missing"],
            "smiles": ["CCO", "not-a-smiles", None],
        }
    )
    features, stats = featurize_smiles(
        chemicals,
        output_path=tmp_path / "feat.parquet",
        stats_path=tmp_path / "feat.json",
        radius=2,
        n_bits=64,
    )
    assert stats["valid_smiles"] == 1
    assert stats["invalid_smiles"] == 1
    assert stats["missing_smiles"] == 1
    assert stats["feature_dimension"] == 64
    assert features.loc[features["smiles_valid"], "fingerprint"].iloc[0] is not None
    assert features.loc[features["skip_reason"] == "invalid_smiles", "fingerprint"].iloc[0] is None
    mol = parse_smiles("CCO")
    assert morgan_fingerprint(mol, radius=2, n_bits=64) == morgan_fingerprint(mol, radius=2, n_bits=64)


def test_patient_split_leakage_detection_and_determinism(tmp_path: Path) -> None:
    table = pd.DataFrame({"patient_id": [f"p{i}" for i in range(20)] * 2})
    first, _ = split_patients(
        table,
        seed=7,
        output_path=tmp_path / "splits1.csv",
        stats_path=tmp_path / "s1.json",
    )
    second, _ = split_patients(
        table,
        seed=7,
        output_path=tmp_path / "splits2.csv",
        stats_path=tmp_path / "s2.json",
    )
    assert first.equals(second)
    validate_patient_splits(first)
    leaked = pd.DataFrame({"patient_id": ["p1", "p1"], "split": ["train", "test"]})
    with pytest.raises(ConfigurationError):
        validate_patient_splits(leaked)


def test_dataset_assembly(tmp_path: Path) -> None:
    pairs = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "drug_a": ["a", "a"],
            "drug_b": ["b", "c"],
            "pair_key": ["a||b", "a||c"],
            "label": ["positive", "unknown"],
        }
    )
    splits = pd.DataFrame({"patient_id": ["p1", "p2"], "split": ["train", "test"]})
    features = pd.DataFrame(
        {
            "drug_id": ["a", "b", "c"],
            "smiles": ["C", "CC", "CCC"],
            "smiles_valid": [True, True, True],
            "fingerprint": [[1, 0], [0, 1], [1, 1]],
            "skip_reason": [None, None, None],
        }
    )
    dataset, metadata = build_ml_dataset(
        labeled_pairs=pairs,
        splits=splits,
        molecular_features=features,
        output_path=tmp_path / "ml.parquet",
        metadata_path=tmp_path / "meta.json",
    )
    assert metadata["n_rows"] == 2
    assert "drug_a_fingerprint" in dataset.columns
    assert set(dataset["split"]) == {"train", "test"}


def test_prepare_drugs_pending_rxnorm(tmp_path: Path) -> None:
    prescriptions = pd.DataFrame(
        {
            "patient_id": [1, 1],
            "admission_id": [10, 10],
            "drug_name_raw": ["Alpha", "Beta"],
            "drug_name_norm": ["alpha", "beta"],
        }
    )
    prepared, stats = prepare_drugs_pending_rxnorm(
        prescriptions,
        output_path=tmp_path / "prepared.parquet",
        stats_path=tmp_path / "prepared.json",
    )
    assert stats["rxnorm_source_status"]["status"] == "pending"
    assert prepared["rxnorm_status"].iloc[0] == "pending"
    assert prepared["rxnorm_mapped"].eq(False).all()


def test_normalize_without_mapping_marks_pending(tmp_path: Path) -> None:
    prescriptions = pd.DataFrame(
        {
            "patient_id": [1],
            "drug_name_raw": ["Alpha"],
            "drug_name_norm": ["alpha"],
        }
    )
    normalized, stats = normalize_drugs(
        prescriptions,
        mapping_path=None,
        output_path=tmp_path / "norm.parquet",
        stats_path=tmp_path / "norm.json",
    )
    assert stats["rxnorm_source_status"]["status"] == "pending"
    assert pd.isna(normalized["rxnorm_id"].iloc[0])


def test_mimic_iv_prescription_selection(tmp_path: Path) -> None:
    root = tmp_path / "mimic-iv-2.1" / "hosp"
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [10, 20],
            "drug": ["Alpha", "Beta"],
            "starttime": ["2020-01-01", "2020-01-02"],
            "stoptime": ["2020-01-05", "2020-01-06"],
        }
    ).to_csv(root / "prescriptions.csv", index=False)
    table, meta = load_prescriptions(
        mimic_dir=tmp_path / "mimic-iv-2.1",
        output_path=tmp_path / "std.parquet",
        streaming=False,
    )
    assert meta["n_rows"] == 2
    assert "patient_id" in table.columns
    assert meta["source_files"][0]["path"].endswith("prescriptions.csv")


def test_missing_mimic_and_mapping_errors(tmp_path: Path) -> None:
    empty = tmp_path / "mimic"
    empty.mkdir()
    with pytest.raises(MissingInputError):
        load_prescriptions(mimic_dir=empty, output_path=tmp_path / "out.parquet")
    mimic_file = empty / "notes.csv"
    pd.DataFrame({"hadm_id": [1], "text": ["note"]}).to_csv(mimic_file, index=False)
    with pytest.raises(MissingInputError):
        load_prescriptions(mimic_dir=empty, output_path=tmp_path / "out.parquet")


def test_mimic_ingest_clean_twosides_roundtrip(tmp_path: Path) -> None:
    mimic_dir = tmp_path / "mimic"
    mimic_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, None],
            "hadm_id": [10, 10, 10, 20, 30],
            "drug": [" Alpha ", "Beta", "Alpha", "Gamma", "Zeta"],
            "starttime": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-02-01", "not-a-date"],
            "stoptime": ["2020-01-10", "2020-01-08", "2020-01-10", "2020-02-03", "2020-03-01"],
        }
    ).to_csv(mimic_dir / "hosp_prescriptions.csv", index=False)
    table, meta = load_prescriptions(
        mimic_dir=mimic_dir,
        output_path=tmp_path / "std.parquet",
        streaming=False,
    )
    assert "patient_id" in table.columns
    assert meta["n_rows"] == 5
    cleaned, clean_stats = clean_prescriptions(
        tmp_path / "std.parquet",
        output_path=tmp_path / "clean.parquet",
        stats_path=tmp_path / "clean.json",
        streaming=False,
    )
    assert clean_stats["missing_patient_values"] == 1
    assert clean_stats["invalid_dates"] >= 1
    assert "drug_name_norm" in cleaned.columns

    tw_dir = tmp_path / "twosides"
    tw_dir.mkdir()
    pd.DataFrame({"d1": ["x", "b"], "d2": ["a", "b"], "type": [1, 2]}).to_csv(tw_dir / "ddis.csv", index=False)
    pd.DataFrame({"drug_id": ["a", "b", "x"], "smiles": ["C", "CC", "CCC"]}).to_csv(
        tw_dir / "drug_smiles.csv", index=False
    )
    tw, tw_stats = load_twosides(
        twosides_dir=tw_dir,
        output_path=tmp_path / "tw.parquet",
        stats_path=tmp_path / "tw.json",
        streaming=False,
    )
    assert "a||x" in set(tw["pair_key"])
    assert tw_stats["skipped_self_or_empty_pairs"] == 1


def test_morgan_generator_equivalence() -> None:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mol = parse_smiles("CCO")
    assert mol is not None
    fp_list = morgan_fingerprint(mol, radius=2, n_bits=2048)
    assert len(fp_list) == 2048
    assert sum(fp_list) > 0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    gen_fp = [int(b) for b in generator.GetFingerprint(mol)]
    assert fp_list == gen_fp

