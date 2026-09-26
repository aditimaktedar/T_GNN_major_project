"""Command-line entry point for Member B pipeline stages.

Each stage can be run independently. Missing inputs raise a clear error.
Nothing is invented when a source file is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.build_dataset import build_ml_dataset
from src.data.clean_mimic import clean_prescriptions
from src.data.config import (
    DEFAULT_FINGERPRINT_N_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_MIMIC_CHUNKSIZE,
    DEFAULT_NEGATIVE_RATIO,
    DEFAULT_TWOSIDES_CHUNKSIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DRUG_PAIRS_PATH,
    MIMIC_CLEAN_PATH,
    MIMIC_DIR,
    MIMIC_EXPOSURES_PATH,
    MIMIC_NORMALIZED_PATH,
    MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH,
    MIMIC_PUBCHEM_MAPPING_PATH,
    MIMIC_PRESCRIPTIONS_PATH,
    MIMIC_TWOSIDES_LABELED_PATH,
    OFFSIDES_DIR,
    PUBCHEM_MAPPING_PATH,
    TWOSIDES_DIR,
    TWOSIDES_DRUGS_PATH,
    TWOSIDES_INTERACTIONS_PATH,
    TWOSIDES_PAIRS_PATH,
)
from src.data.drug_pairs import construct_drug_pairs
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.inventory import build_inventory, format_inventory
from src.data.inspect_data import format_report, inspect_file
from src.data.labels import integrate_labels
from src.data.mimic import load_prescriptions
from src.data.mimic_twosides_join import join_mimic_twosides_labels
from src.data.mimic_twosides_ml_dataset import prepare_mimic_twosides_ml_dataset
from src.data.normalize_drugs import normalize_drugs, prepare_drugs_pending_rxnorm
from src.data.offsides import load_offsides
from src.data.splits import split_patients
from src.data.twosides import load_twosides
from src.features.pubchem import map_mimic_drugs_to_pubchem, retrieve_smiles
from src.features.pubchem_candidates import prepare_mimic_pubchem_mapping_candidates
from src.features.rdkit_features import featurize_smiles


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_inspect(args: argparse.Namespace) -> int:
    if args.file:
        print(format_report(inspect_file(args.file)), end="")
        return 0
    directories = [MIMIC_DIR, TWOSIDES_DIR, OFFSIDES_DIR]
    records = build_inventory(directories=directories)
    print(format_inventory(records), end="")
    if not records:
        print(
            "Place approved datasets under data/raw/mimic/, data/raw/twosides/, "
            "and data/raw/offsides/ without modifying the originals."
        )
    return 0


def cmd_ingest_mimic(args: argparse.Namespace) -> int:
    _, metadata = load_prescriptions(
        mimic_dir=Path(args.mimic_dir) if args.mimic_dir else None,
        chunksize=args.chunksize,
        streaming=not args.no_streaming,
    )
    _print_json(metadata)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    _, stats = clean_prescriptions(
        Path(args.input),
        chunksize=args.chunksize,
        streaming=not args.no_streaming,
    )
    _print_json(stats)
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    mapping = Path(args.mapping) if args.mapping else None
    _, stats = normalize_drugs(Path(args.input), mapping_path=mapping)
    _print_json(stats)
    return 0


def cmd_prepare_mimic(args: argparse.Namespace) -> int:
    """Run MIMIC ingest, clean, pending-RxNorm prepare, and drug-pair construction."""
    _, ingest_meta = load_prescriptions(
        mimic_dir=Path(args.mimic_dir) if args.mimic_dir else None,
        chunksize=args.chunksize,
        streaming=not args.no_streaming,
    )
    _, clean_stats = clean_prescriptions(
        MIMIC_PRESCRIPTIONS_PATH,
        chunksize=args.chunksize,
        streaming=not args.no_streaming,
    )
    if args.mapping:
        _, norm_stats = normalize_drugs(MIMIC_CLEAN_PATH, mapping_path=Path(args.mapping))
    else:
        _, norm_stats = prepare_drugs_pending_rxnorm(MIMIC_CLEAN_PATH)
    pair_input = MIMIC_NORMALIZED_PATH if args.use_prepared else MIMIC_CLEAN_PATH
    _, pair_stats = construct_drug_pairs(
        pair_input,
        pair_rule=args.rule,
        aggregate_exposures_first=True,
        chunksize=args.chunksize,
    )
    summary = {
        "status": "ok",
        "ingest": ingest_meta,
        "clean": clean_stats,
        "normalize": norm_stats,
        "pairs": pair_stats,
        "rxnorm_status": norm_stats.get("rxnorm_source_status", {}).get("status", "mapped"),
    }
    _print_json(summary)
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    _, stats = construct_drug_pairs(
        Path(args.input),
        pair_rule=args.rule,
        aggregate_exposures_first=not args.no_aggregate,
        chunksize=args.chunksize,
    )
    _print_json(stats)
    return 0


def cmd_twosides(args: argparse.Namespace) -> int:
    _, stats = load_twosides(
        twosides_dir=Path(args.twosides_dir) if args.twosides_dir else None,
        chunksize=args.chunksize,
        streaming=not args.no_streaming,
        load_drugs=not args.skip_drugs,
    )
    _print_json(stats)
    return 0


def cmd_offsides(args: argparse.Namespace) -> int:
    _, stats = load_offsides(offsides_dir=Path(args.offsides_dir) if args.offsides_dir else None)
    _print_json(stats)
    return 0


def cmd_labels_twosides(args: argparse.Namespace) -> int:
    _, stats = join_mimic_twosides_labels(
        mimic_pairs_path=Path(args.pairs) if args.pairs else None,
        mapping_path=Path(args.mapping) if args.mapping else None,
        interactions_path=Path(args.interactions) if args.interactions else None,
        batch_size=args.batch_size,
    )
    _print_json(stats)
    return 0


def cmd_prepare_ml_twosides(args: argparse.Namespace) -> int:
    stats = prepare_mimic_twosides_ml_dataset(
        labeled_path=Path(args.labeled) if args.labeled else None,
        mimic_pairs_path=Path(args.pairs) if args.pairs else None,
        mapping_path=Path(args.mapping) if args.mapping else None,
        unique_pairs_path=Path(args.unique_pairs) if args.unique_pairs else None,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        batch_size=args.batch_size,
        n_buckets=args.n_buckets,
        resume_buckets=args.resume_buckets,
    )
    _print_json(stats)
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    _, stats = integrate_labels(
        Path(args.pairs),
        Path(args.ddi),
        negative_mode=args.negative_mode,
        seed=args.seed,
    )
    _print_json(stats)
    return 0


def cmd_pubchem_candidates(args: argparse.Namespace) -> int:
    _, stats = prepare_mimic_pubchem_mapping_candidates(
        exposures_path=Path(args.exposures) if args.exposures else None,
        twosides_drugs_path=Path(args.twosides_drugs) if args.twosides_drugs else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    _print_json(stats)
    return 0


def cmd_pubchem_map(args: argparse.Namespace) -> int:
    _, stats = map_mimic_drugs_to_pubchem(
        input_path=Path(args.input) if args.input else None,
        name_column=args.name_column,
        pause_seconds=args.pause,
        limit=args.limit,
    )
    _print_json(stats)
    return 0


def cmd_pubchem(args: argparse.Namespace) -> int:
    _, stats = retrieve_smiles(Path(args.input), query_column=args.query_column)
    _print_json(stats)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    _, stats = featurize_smiles(
        Path(args.input),
        radius=args.radius,
        n_bits=args.n_bits,
    )
    _print_json(stats)
    return 0


def cmd_splits(args: argparse.Namespace) -> int:
    _, stats = split_patients(
        Path(args.input),
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    _print_json(stats)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    _, metadata = build_ml_dataset()
    _print_json(metadata)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    print("Running full pipeline. Each missing prerequisite will stop the run.")
    load_prescriptions()
    clean_prescriptions(MIMIC_PRESCRIPTIONS_PATH)
    if not args.mapping:
        raise MissingInputError(
            "Full pipeline requires --mapping path/to/rxnorm_mapping.csv. "
            "RxNorm IDs will not be invented."
        )
    normalize_drugs(MIMIC_CLEAN_PATH, mapping_path=Path(args.mapping))
    construct_drug_pairs(MIMIC_NORMALIZED_PATH, pair_rule=args.rule)
    load_twosides()
    integrate_labels(DRUG_PAIRS_PATH, TWOSIDES_PAIRS_PATH, negative_mode=args.negative_mode)
    retrieve_smiles(MIMIC_NORMALIZED_PATH, query_column="drug_name_norm")
    featurize_smiles(PUBCHEM_MAPPING_PATH, radius=args.radius, n_bits=args.n_bits)
    split_patients(DRUG_PAIRS_PATH, seed=args.seed)
    metadata = build_ml_dataset()[1]
    _print_json(metadata)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TemporalDDI-GNN Member B data pipeline. Stages run independently."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect a file or inventory approved raw folders.")
    inspect.add_argument("file", nargs="?", help="Optional local file path.")
    inspect.set_defaults(func=cmd_inspect)

    ingest = sub.add_parser("ingest-mimic", help="Discover and load MIMIC prescription files.")
    ingest.add_argument("--mimic-dir")
    ingest.add_argument("--chunksize", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    ingest.add_argument("--no-streaming", action="store_true")
    ingest.set_defaults(func=cmd_ingest_mimic)

    clean = sub.add_parser("clean", help="Clean standardized MIMIC prescriptions.")
    clean.add_argument("--input", default=str(MIMIC_PRESCRIPTIONS_PATH))
    clean.add_argument("--chunksize", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    clean.add_argument("--no-streaming", action="store_true")
    clean.set_defaults(func=cmd_clean)

    normalize = sub.add_parser("normalize", help="Apply a supplied RxNorm mapping table or mark mapping pending.")
    normalize.add_argument("--input", default=str(MIMIC_CLEAN_PATH))
    normalize.add_argument("--mapping", help="CSV/TSV with source drug and RxNorm columns. Omit to mark RxNorm pending.")
    normalize.set_defaults(func=cmd_normalize)

    prepare = sub.add_parser(
        "prepare-mimic",
        help="Ingest, clean, prepare drugs (RxNorm pending if no mapping), and build drug pairs.",
    )
    prepare.add_argument("--mimic-dir")
    prepare.add_argument("--mapping", help="Optional RxNorm mapping CSV/TSV.")
    prepare.add_argument("--rule", default="same_admission", choices=["interval_overlap", "same_admission", "same_calendar_day"])
    prepare.add_argument("--use-prepared", action="store_true", help="Build pairs from normalized/prepared file instead of clean file.")
    prepare.add_argument("--chunksize", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    prepare.add_argument("--no-streaming", action="store_true")
    prepare.set_defaults(func=cmd_prepare_mimic)

    pairs = sub.add_parser("pairs", help="Construct concurrent drug pairs.")
    pairs.add_argument("--input", default=str(MIMIC_CLEAN_PATH))
    pairs.add_argument("--rule", default="same_admission", choices=["interval_overlap", "same_admission", "same_calendar_day"])
    pairs.add_argument("--no-aggregate", action="store_true")
    pairs.add_argument("--chunksize", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    pairs.set_defaults(func=cmd_pairs)

    twosides = sub.add_parser("twosides", help="Load local TWOSIDES ddis.csv and drug_smiles.csv.")
    twosides.add_argument("--twosides-dir")
    twosides.add_argument("--chunksize", type=int, default=DEFAULT_TWOSIDES_CHUNKSIZE)
    twosides.add_argument("--no-streaming", action="store_true")
    twosides.add_argument("--skip-drugs", action="store_true", help="Load ddis.csv only.")
    twosides.set_defaults(func=cmd_twosides)

    offsides = sub.add_parser("offsides", help="Load local OFFSIDES files if present.")
    offsides.add_argument("--offsides-dir")
    offsides.set_defaults(func=cmd_offsides)

    labels_twosides = sub.add_parser(
        "labels-twosides",
        help="Join MIMIC pairs to TWOSIDES interaction types via verified local CID mappings.",
    )
    labels_twosides.add_argument("--pairs", default=str(DRUG_PAIRS_PATH))
    labels_twosides.add_argument("--mapping", default=str(MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH))
    labels_twosides.add_argument("--interactions", default=str(TWOSIDES_INTERACTIONS_PATH))
    labels_twosides.add_argument("--batch-size", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    labels_twosides.set_defaults(func=cmd_labels_twosides)

    prepare_ml = sub.add_parser(
        "prepare-ml-twosides",
        help="Stream dedupe positives, sample real negatives, assign patient splits for ML.",
    )
    prepare_ml.add_argument("--labeled", default=str(MIMIC_TWOSIDES_LABELED_PATH))
    prepare_ml.add_argument("--pairs", default=str(DRUG_PAIRS_PATH))
    prepare_ml.add_argument("--mapping", default=str(MIMIC_PUBCHEM_MAPPING_CANDIDATES_PATH))
    prepare_ml.add_argument("--unique-pairs", default=str(TWOSIDES_PAIRS_PATH))
    prepare_ml.add_argument("--negative-ratio", type=float, default=DEFAULT_NEGATIVE_RATIO)
    prepare_ml.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    prepare_ml.add_argument("--train", type=float, default=DEFAULT_TRAIN_RATIO)
    prepare_ml.add_argument("--val", type=float, default=DEFAULT_VAL_RATIO)
    prepare_ml.add_argument("--test", type=float, default=DEFAULT_TEST_RATIO)
    prepare_ml.add_argument("--batch-size", type=int, default=DEFAULT_MIMIC_CHUNKSIZE)
    prepare_ml.add_argument("--n-buckets", type=int, default=512)
    prepare_ml.add_argument(
        "--resume-buckets",
        action="store_true",
        help="Skip pass-1 hash bucketing when temp buckets already exist.",
    )
    prepare_ml.set_defaults(func=cmd_prepare_ml_twosides)

    labels = sub.add_parser("labels", help="Join MIMIC pairs with TWOSIDES. Default unmatched=unknown.")
    labels.add_argument("--pairs", default=str(DRUG_PAIRS_PATH))
    labels.add_argument("--ddi", default=str(TWOSIDES_PAIRS_PATH))
    labels.add_argument("--negative-mode", default="none", choices=["none", "random_unmatched"])
    labels.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    labels.set_defaults(func=cmd_labels)

    pubchem_candidates = sub.add_parser(
        "pubchem-candidates",
        help="Local-only MIMIC→TWOSIDES mapping candidates (no API calls).",
    )
    pubchem_candidates.add_argument("--exposures", default=str(MIMIC_EXPOSURES_PATH))
    pubchem_candidates.add_argument("--twosides-drugs", default=str(TWOSIDES_DRUGS_PATH))
    pubchem_candidates.add_argument("--cache-dir", help="Optional PubChem CID cache directory.")
    pubchem_candidates.set_defaults(func=cmd_pubchem_candidates)

    pubchem_map = sub.add_parser(
        "pubchem-map",
        help="Map unique MIMIC drug names to PubChem CIDs via PUG REST (cached, resumable).",
    )
    pubchem_map.add_argument("--input", default=str(MIMIC_NORMALIZED_PATH))
    pubchem_map.add_argument("--name-column", default="drug_name_norm")
    pubchem_map.add_argument("--pause", type=float, default=0.25, help="Seconds between API requests.")
    pubchem_map.add_argument("--limit", type=int, help="Optional limit for testing/debug only.")
    pubchem_map.set_defaults(func=cmd_pubchem_map)

    pubchem = sub.add_parser("pubchem", help="Retrieve SMILES from PubChem PUG REST with caching.")
    pubchem.add_argument("--input", default=str(MIMIC_NORMALIZED_PATH))
    pubchem.add_argument("--query-column", default="drug_name_norm")
    pubchem.set_defaults(func=cmd_pubchem)

    features = sub.add_parser("features", help="Build Morgan fingerprints from SMILES.")
    features.add_argument("--input", default=str(PUBCHEM_MAPPING_PATH))
    features.add_argument("--radius", type=int, default=DEFAULT_FINGERPRINT_RADIUS)
    features.add_argument("--n-bits", type=int, default=DEFAULT_FINGERPRINT_N_BITS)
    features.set_defaults(func=cmd_features)

    splits = sub.add_parser("splits", help="Create patient-level train/val/test assignments.")
    splits.add_argument("--input", default=str(DRUG_PAIRS_PATH))
    splits.add_argument("--train", type=float, default=DEFAULT_TRAIN_RATIO)
    splits.add_argument("--val", type=float, default=DEFAULT_VAL_RATIO)
    splits.add_argument("--test", type=float, default=DEFAULT_TEST_RATIO)
    splits.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    splits.set_defaults(func=cmd_splits)

    build = sub.add_parser("build", help="Join labeled pairs, splits, and molecular features.")
    build.set_defaults(func=cmd_build)

    run = sub.add_parser("run", help="Run the full pipeline if all required local sources exist.")
    run.add_argument("--mapping", help="RxNorm mapping CSV/TSV (required).")
    run.add_argument("--rule", default="interval_overlap")
    run.add_argument("--negative-mode", default="none")
    run.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    run.add_argument("--radius", type=int, default=DEFAULT_FINGERPRINT_RADIUS)
    run.add_argument("--n-bits", type=int, default=DEFAULT_FINGERPRINT_N_BITS)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MissingInputError, SchemaError, ConfigurationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
