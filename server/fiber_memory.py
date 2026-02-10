import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class FiberMemory:
    """
    Fiber Memory (FM): A storage layer for Neural Fiber Bundle (NFB) transport matrices.
    Enables one-shot knowledge injection and bias decoupling without re-training.
    """
    def __init__(self, storage_path: str = "tempdata/fiber_memory.json"):
        self.storage_path = Path(storage_path)
        self.memory: Dict[str, Dict] = {}
        self.load()

    def load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Failed to load Fiber Memory: {e}")
                self.memory = {}

    def save_persistent(self):
        os.makedirs(self.storage_path.parent, exist_ok=True)
        # Convert numpy arrays to list for JSON serialization
        serializable_mem = {}
        for key, val in self.memory.items():
            serializable_mem[key] = {
                "source_concept": val["source_concept"],
                "target_concept": val["target_concept"],
                "R": val["R"].tolist() if isinstance(val["R"], np.ndarray) else val["R"],
                "layer_idx": val["layer_idx"],
                "timestamp": val.get("timestamp", "")
            }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(serializable_mem, f, indent=2, ensure_ascii=False)

    def store_transport(self, source: str, target: str, R: np.ndarray, layer_idx: int):
        """Stores a transport matrix for a concept pair at a specific layer."""
        key = f"{source}->{target}_L{layer_idx}"
        self.memory[key] = {
            "source_concept": source,
            "target_concept": target,
            "R": R,
            "layer_idx": layer_idx,
            "timestamp": str(np.datetime64('now'))
        }
        self.save_persistent()

    def get_transport(self, source: str, target: str, layer_idx: int) -> Optional[np.ndarray]:
        key = f"{source}->{target}_L{layer_idx}"
        data = self.memory.get(key)
        if data:
            return np.array(data["R"])
        return None

    def get_all_for_layer(self, layer_idx: int) -> List[np.ndarray]:
        """Returns all transport matrices stored for a specific layer."""
        matrices = []
        for v in self.memory.values():
            if v["layer_idx"] == layer_idx:
                matrices.append(np.array(v["R"]))
        return matrices

    def list_injections(self) -> List[Dict]:
        return [
            {
                "id": k,
                "source": v["source_concept"],
                "target": v["target_concept"],
                "layer": v["layer_idx"]
            }
            for k, v in self.memory.items()
        ]

    def clear(self):
        self.memory = {}
        if self.storage_path.exists():
            os.remove(self.storage_path)
