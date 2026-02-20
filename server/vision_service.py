
import os

import numpy as np
import torch
import torch.nn as nn

from models.vision_projector import VisionProjector


_DIGIT_TO_SEGMENTS = {
    0: ("A", "B", "C", "D", "E", "F"),
    1: ("B", "C"),
    2: ("A", "B", "G", "E", "D"),
    3: ("A", "B", "C", "D", "G"),
    4: ("F", "G", "B", "C"),
    5: ("A", "F", "G", "C", "D"),
    6: ("A", "F", "E", "D", "C", "G"),
    7: ("A", "B", "C"),
    8: ("A", "B", "C", "D", "E", "F", "G"),
    9: ("A", "B", "C", "D", "F", "G"),
}

_BASE_SEGMENTS = {
    "A": (6, 3, 21, 5),
    "B": (20, 6, 22, 13),
    "C": (20, 14, 22, 21),
    "D": (6, 22, 21, 24),
    "E": (5, 14, 7, 21),
    "F": (5, 6, 7, 13),
    "G": (6, 12, 21, 15),
}


def _synthetic_digit_image(digit: int) -> np.ndarray:
    canvas = np.zeros((28, 28), dtype=np.float32)
    for seg in _DIGIT_TO_SEGMENTS.get(int(digit), ()):
        x1, y1, x2, y2 = _BASE_SEGMENTS[seg]
        canvas[y1:y2, x1:x2] = 1.0
    # Keep the same normalization used during training.
    canvas = (canvas - 0.1307) / 0.3081
    return canvas


class VisionService:
    def __init__(self, d_model=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projector = VisionProjector(d_model=d_model).to(self.device)
        self.d_model = d_model
        
        # Load pre-trained weights if available.
        candidates = [
            os.environ.get("VISION_PROJECTOR_PATH", ""),
            "tempdata/vision_projector.pth",
            "models/vision_projector_weights.pt",
        ]
        self.weights_path = None
        for candidate in candidates:
            if not candidate:
                continue
            if os.path.exists(candidate):
                self.projector.load_state_dict(
                    torch.load(candidate, map_location=self.device),
                    strict=False,
                )
                self.weights_path = candidate
                break
        if self.weights_path:
            print(f"VisionProjector weights loaded from {self.weights_path}")
        else:
            print("WARNING: VisionProjector weights not found. Using initialized weights for demo.")

    def project_image(self, image_data):
        """
        Projects a single 28x28 image into the manifold.
        image_data: np.array [1, 28, 28] or [28, 28]
        """
        self.projector.eval()
        with torch.no_grad():
            if len(image_data.shape) == 2:
                image_data = np.expand_dims(image_data, axis=0) # Add channel
            
            x = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 1, 28, 28]
            projection = self.projector(x) # [1, d_model]
            return projection.cpu().numpy().flatten()

    def get_mnist_anchors(self):
        """
        Generates 10 example projections for digits 0-9 for alignment visualization.
        """
        anchors = []
        for i in range(10):
            img = _synthetic_digit_image(i)
            
            proj = self.project_image(img)
            anchors.append({
                "digit": i,
                "projection": proj.tolist(),
                "label": f"MNIST_{i}"
            })
        return anchors

vision_service = VisionService()
