"""
Drug Resolver service.
Maps clinical drug names, DrugBank IDs, PubChem CIDs, and TWOSIDES identifiers
to standardized DrugEntity objects and model vocabulary indices.
"""

import os
import csv
import re
from typing import Dict, Any, List, Optional
from api.schemas import DrugEntity


class DrugResolver:
    def __init__(
        self,
        mapping_csv_path: Optional[str] = None,
        profile_csv_path: Optional[str] = None,
        drug_to_idx: Optional[Dict[str, int]] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mapping_csv = mapping_csv_path or os.path.join(
            base_dir, "RAG_pipeline/data/mappings/twosides_pubchem_drugbank_final.csv"
        )
        self.profile_csv = profile_csv_path or os.path.join(
            base_dir, "RAG_pipeline/data/rag/pair_retrieval_v2/drug_profiles_all_341_multquery_v23.csv"
        )

        self.drug_to_idx = drug_to_idx or {}

        # Lookups
        self.by_drugbank_id: Dict[str, DrugEntity] = {}
        self.by_cid: Dict[str, DrugEntity] = {}
        self.by_name: Dict[str, DrugEntity] = {}
        self.all_drugs: List[DrugEntity] = []

        self._load_mappings()

    def update_model_vocab(self, drug_to_idx: Dict[str, int]):
        """Updates model index mappings once checkpoint metadata is loaded."""
        self.drug_to_idx = drug_to_idx or {}
        for drug in self.all_drugs:
            if drug.pubchem_cid and str(drug.pubchem_cid) in self.drug_to_idx:
                drug.model_index = self.drug_to_idx[str(drug.pubchem_cid)]
                drug.in_model_vocab = True
            elif drug.drugbank_id in self.drug_to_idx:
                drug.model_index = self.drug_to_idx[drug.drugbank_id]
                drug.in_model_vocab = True

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            return ""
        clean = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
        return clean

    def _load_mappings(self):
        # 1. Load TWOSIDES PubChem DrugBank final mapping
        if os.path.exists(self.mapping_csv):
            with open(self.mapping_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    db_id = row.get("drugbank_id", "").strip().upper()
                    cid = str(row.get("pubchem_cid", "")).strip()
                    name = row.get("drugbank_name", "").strip() or row.get("pubchem_title", "").strip()

                    model_idx = self.drug_to_idx.get(cid)
                    in_vocab = model_idx is not None

                    entity = DrugEntity(
                        drugbank_id=db_id,
                        name=name,
                        pubchem_cid=cid if cid else None,
                        model_index=model_idx,
                        in_model_vocab=in_vocab
                    )

                    if db_id:
                        self.by_drugbank_id[db_id] = entity
                    if cid:
                        self.by_cid[cid] = entity
                    if name:
                        norm = self._normalize_name(name)
                        self.by_name[norm] = entity

                    self.all_drugs.append(entity)

        # 2. Augment with RAG profiles if available
        if os.path.exists(self.profile_csv):
            with open(self.profile_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    db_id = row.get("drugbank_id", "").strip().upper()
                    name = row.get("drug_name", "").strip()
                    if db_id and db_id not in self.by_drugbank_id:
                        model_idx = self.drug_to_idx.get(db_id)
                        entity = DrugEntity(
                            drugbank_id=db_id,
                            name=name or db_id,
                            pubchem_cid=None,
                            model_index=model_idx,
                            in_model_vocab=model_idx is not None
                        )
                        self.by_drugbank_id[db_id] = entity
                        if name:
                            self.by_name[self._normalize_name(name)] = entity
                        self.all_drugs.append(entity)
                    elif db_id in self.by_drugbank_id and not self.by_drugbank_id[db_id].name and name:
                        self.by_drugbank_id[db_id].name = name
                        self.by_name[self._normalize_name(name)] = self.by_drugbank_id[db_id]

    def resolve(self, query: str) -> DrugEntity:
        """
        Resolves a search query (name, DrugBank ID, CID) to a DrugEntity.
        Falls back to a gracefully constructed entity if the drug is unrecognized.
        """
        if not query:
            raise ValueError("Query string cannot be empty")

        raw = str(query).strip()
        upper = raw.upper()

        # 1. Match DrugBank ID
        if upper in self.by_drugbank_id:
            return self.by_drugbank_id[upper]

        # 2. Match PubChem CID
        if raw in self.by_cid:
            return self.by_cid[raw]
        cid_match = re.sub(r"^CID0*", "", upper)
        if cid_match in self.by_cid:
            return self.by_cid[cid_match]

        # 3. Match Normalized Name
        norm = self._normalize_name(raw)
        if norm in self.by_name:
            return self.by_name[norm]

        # 4. Partial / Substring Match on Name
        for known_norm, entity in self.by_name.items():
            if norm and (norm in known_norm or known_norm in norm):
                return entity

        # 5. Unknown drug fallback: create external entity
        model_idx = self.drug_to_idx.get(raw)
        return DrugEntity(
            drugbank_id=upper if upper.startswith("DB") else f"EXT_{norm[:10].upper()}",
            name=raw,
            pubchem_cid=None,
            model_index=model_idx,
            in_model_vocab=model_idx is not None
        )

    def list_all_drugs(self, query: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns a list of known drugs with optional prefix filtering."""
        norm_query = self._normalize_name(query) if query else ""
        results = []
        for d in self.all_drugs:
            if norm_query:
                norm_name = self._normalize_name(d.name)
                norm_db = self._normalize_name(d.drugbank_id)
                if norm_query not in norm_name and norm_query not in norm_db:
                    continue
            results.append(d.to_dict())
            if len(results) >= limit:
                break
        return results
