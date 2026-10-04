import os
import math
import struct
import zipfile
from typing import Dict, Any, Optional, List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


SEVERITY_CLASSES = [
    "Minor",
    "Moderate",
    "Major"
]


def load_checkpoint_metadata(checkpoint_path: str) -> Dict[str, Any]:
    """Reads metadata (drug_to_idx, fold, labels, etc.) from a .pt checkpoint without requiring torch."""
    import pickle

    class SafeTorchUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if "torch" in module:
                return lambda *args, **kwargs: (module, name, args)
            return super().find_class(module, name)

        def persistent_load(self, pid):
            return pid

    filename = os.path.basename(checkpoint_path)
    base = os.path.splitext(filename)[0]

    with zipfile.ZipFile(checkpoint_path) as z:
        candidate_pkls = [
            f"{base}/data.pkl",
            "archive/data.pkl",
            "data.pkl"
        ]
        pkl_target = None
        for cand in candidate_pkls:
            if cand in z.namelist():
                pkl_target = cand
                break
        if not pkl_target:
            for item in z.namelist():
                if item.endswith("data.pkl"):
                    pkl_target = item
                    break

        if not pkl_target:
            raise ValueError(f"Could not find data.pkl in {checkpoint_path}")

        with z.open(pkl_target) as f:
            data = SafeTorchUnpickler(f).load()

    meta = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if k != "model_state_dict" and k != "state_dict":
                meta[k] = v
    return meta


class StandaloneInferenceEngine:
    """
    Pure-Python vectorized inference engine for TemporalDDIGNN model_v2 checkpoints.
    Executes exact forward pass directly from checkpoint storages with zero external dependencies.
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.metadata = load_checkpoint_metadata(checkpoint_path)
        self.drug_to_idx = self.metadata.get("drug_to_idx", {})
        self.labels = self.metadata.get("labels", {c: i for i, c in enumerate(SEVERITY_CLASSES)})
        self.fold = self.metadata.get("fold", 1)
        self.num_drugs = len(self.drug_to_idx) or 38
        self.embedding_dim = 32

        self._load_weights()

    def _load_weights(self):
        filename = os.path.basename(self.checkpoint_path)
        base = os.path.splitext(filename)[0]

        with zipfile.ZipFile(self.checkpoint_path) as z:
            storages = {}
            for name in z.namelist():
                if "/data/" in name:
                    storage_idx = int(name.split("/data/")[-1])
                    raw = z.read(name)
                    floats = struct.unpack(f"<{len(raw)//4}f", raw)
                    storages[storage_idx] = floats

        # Expected mappings in Fold 1-5 checkpoints
        # 0: emb.weight (num_drugs, 32)
        # 1: conv1.lin.weight (32, 32)
        # 2: conv1.lin.bias (32,)
        # 3: conv2.lin.weight (32, 32)
        # 4: conv2.lin.bias (32,)
        # 5: shared.0.weight (64, 160)
        # 6: shared.0.bias (64,)
        # 7: presence_head.0.weight (32, 64)
        # 8: presence_head.0.bias (32,)
        # 9: presence_head.2.weight (1, 32)
        # 10: presence_head.2.bias (1,)
        # 11: severity_head.0.weight (32, 64)
        # 12: severity_head.0.bias (32,)
        # 13: severity_head.2.weight (3, 32)
        # 14: severity_head.2.bias (3,)
        self.weights = storages

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))

    @staticmethod
    def _softmax(xs: List[float]) -> List[float]:
        m = max(xs)
        exps = [math.exp(x - m) for x in xs]
        s = sum(exps)
        return [e / s for e in exps]

    @staticmethod
    def _relu(xs: List[float]) -> List[float]:
        return [max(0.0, x) for x in xs]

    @staticmethod
    def _linear(w: Tuple[float, ...], b: Tuple[float, ...], x: List[float], rows: int, cols: int) -> List[float]:
        out = [0.0] * rows
        for r in range(rows):
            val = b[r]
            offset = r * cols
            for c in range(cols):
                val += w[offset + c] * x[c]
            out[r] = val
        return out

    def get_embedding(self, drug_idx: int) -> List[float]:
        emb_data = self.weights[0]
        start = drug_idx * self.embedding_dim
        return list(emb_data[start : start + self.embedding_dim])

    def encode_graph(self, edge_index: Optional[List[Tuple[int, int]]] = None) -> List[List[float]]:
        # x is all embeddings [num_drugs, 32]
        all_embeddings = [self.get_embedding(i) for i in range(self.num_drugs)]

        if not edge_index:
            return all_embeddings

        # SafeGraphConv 1
        adj = {i: [] for i in range(self.num_drugs)}
        for src, dst in edge_index:
            if 0 <= src < self.num_drugs and 0 <= dst < self.num_drugs:
                adj[dst].append(src)

        conv1_out = []
        for dst in range(self.num_drugs):
            nbrs = adj[dst]
            if not nbrs:
                agg = [0.0] * self.embedding_dim
            else:
                deg = len(nbrs)
                agg = [sum(all_embeddings[src][d] for src in nbrs) / deg for d in range(self.embedding_dim)]
            lin_val = self._linear(self.weights[1], self.weights[2], agg, self.embedding_dim, self.embedding_dim)
            conv1_out.append(self._relu(lin_val))

        # SafeGraphConv 2
        conv2_out = []
        for dst in range(self.num_drugs):
            nbrs = adj[dst]
            if not nbrs:
                agg = [0.0] * self.embedding_dim
            else:
                deg = len(nbrs)
                agg = [sum(conv1_out[src][d] for src in nbrs) / deg for d in range(self.embedding_dim)]
            lin_val = self._linear(self.weights[3], self.weights[4], agg, self.embedding_dim, self.embedding_dim)
            conv2_out.append(self._relu(lin_val))

        # Residual: x + x1 + x2
        final_repr = []
        for i in range(self.num_drugs):
            combined = [all_embeddings[i][d] + conv1_out[i][d] + conv2_out[i][d] for d in range(self.embedding_dim)]
            final_repr.append(combined)

        return final_repr

    def predict_pair(
        self,
        drug_a_idx: int,
        drug_b_idx: int,
        edge_index: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        node_repr = self.encode_graph(edge_index)

        h_a = node_repr[drug_a_idx]
        h_b = node_repr[drug_b_idx]

        # Pair representation: [hA, hB, hA*hB, |hA-hB|, hA+hB] -> 160-dim
        pair = []
        pair.extend(h_a)
        pair.extend(h_b)
        pair.extend([a * b for a, b in zip(h_a, h_b)])
        pair.extend([abs(a - b) for a, b in zip(h_a, h_b)])
        pair.extend([a + b for a, b in zip(h_a, h_b)])

        # Shared layer: Linear(160, 64) -> ReLU
        shared_out = self._relu(
            self._linear(self.weights[5], self.weights[6], pair, 64, 160)
        )

        # Presence head: Linear(64, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
        p1 = self._relu(
            self._linear(self.weights[7], self.weights[8], shared_out, 32, 64)
        )
        p_logit = self._linear(self.weights[9], self.weights[10], p1, 1, 32)[0]
        presence_probability = self._sigmoid(p_logit)

        # Severity head: Linear(64, 32) -> ReLU -> Linear(32, 3) -> Softmax
        s1 = self._relu(
            self._linear(self.weights[11], self.weights[12], shared_out, 32, 64)
        )
        s_logits = self._linear(self.weights[13], self.weights[14], s1, 3, 32)
        severity_probabilities = self._softmax(s_logits)

        severity_index = max(range(len(severity_probabilities)), key=lambda i: severity_probabilities[i])

        return {
            "presence_probability": float(presence_probability),
            "presence_label": "Yes" if presence_probability >= 0.5 else "No",
            "severity_label": SEVERITY_CLASSES[severity_index],
            "severity_probabilities": {
                label: float(severity_probabilities[i])
                for i, label in enumerate(SEVERITY_CLASSES)
            },
            "fold": self.fold
        }


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: Optional[str] = None
):
    """Loads weights into a PyTorch TemporalDDIGNN model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for load_checkpoint. Use StandaloneInferenceEngine instead.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remap state dict keys if needed
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith("embedding.") and hasattr(model, "emb"):
            cleaned_dict[k.replace("embedding.", "emb.")] = v
        elif k.startswith("emb.") and hasattr(model, "embedding") and not hasattr(model, "emb"):
            cleaned_dict[k.replace("emb.", "embedding.")] = v
        else:
            cleaned_dict[k] = v

    model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def predict_pair(
    model,
    drug_a_idx: int,
    drug_b_idx: int,
    edge_index=None
) -> Dict[str, Any]:
    """Runs prediction using either a PyTorch model or a StandaloneInferenceEngine."""
    if isinstance(model, StandaloneInferenceEngine):
        return model.predict_pair(drug_a_idx, drug_b_idx, edge_index=edge_index)

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Pass a StandaloneInferenceEngine instance.")

    device = next(model.parameters()).device

    drug_a = torch.tensor([drug_a_idx], dtype=torch.long, device=device)
    drug_b = torch.tensor([drug_b_idx], dtype=torch.long, device=device)

    if edge_index is not None and isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.to(device)

    with torch.no_grad():
        output = model(drug_a, drug_b, edge_index)

    presence_prob = float(torch.sigmoid(output["presence_logits"]).squeeze().item())
    severity_probs = torch.softmax(output["severity_logits"], dim=-1)[0]
    severity_index = int(torch.argmax(severity_probs).item())

    return {
        "presence_probability": presence_prob,
        "presence_label": "Yes" if presence_prob >= 0.5 else "No",
        "severity_label": SEVERITY_CLASSES[severity_index],
        "severity_probabilities": {
            label: float(severity_probs[i].item())
            for i, label in enumerate(SEVERITY_CLASSES)
        }
    }