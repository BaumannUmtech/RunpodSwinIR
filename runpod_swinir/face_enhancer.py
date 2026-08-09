from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize

CODEFORMER_ROOT = Path(os.getenv("CODEFORMER_ROOT", "/opt/CodeFormer"))
CODEFORMER_MODEL_PATH = Path(
    os.getenv("CODEFORMER_MODEL_PATH", str(CODEFORMER_ROOT / "weights/CodeFormer/codeformer.pth"))
)


class CodeFormerFaceEnhancer:
    """Restore detected faces and paste them onto an already upscaled RGB image."""

    def __init__(self) -> None:
        if str(CODEFORMER_ROOT) not in sys.path:
            sys.path.insert(0, str(CODEFORMER_ROOT))

        from basicsr.archs import codeformer_arch  # noqa: F401
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.registry import ARCH_REGISTRY
        from facelib.utils.face_restoration_helper import FaceRestoreHelper

        if not CODEFORMER_MODEL_PATH.is_file():
            raise RuntimeError(f"CodeFormer-Modell fehlt: {CODEFORMER_MODEL_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img2tensor = img2tensor
        self.tensor2img = tensor2img
        self.net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        checkpoint = torch.load(CODEFORMER_MODEL_PATH, map_location="cpu", weights_only=True)
        self.net.load_state_dict(checkpoint["params_ema"])
        self.net.eval()
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=os.getenv("CODEFORMER_DETECTION_MODEL", "retinaface_resnet50"),
            save_ext="png",
            use_parse=True,
            device=self.device,
        )
        self.fidelity = float(os.getenv("CODEFORMER_FIDELITY", "0.7"))
        self._lock = threading.Lock()

    def enhance(self, rgb_image: np.ndarray) -> tuple[np.ndarray, int]:
        if rgb_image.dtype != np.uint8:
            raise ValueError("CodeFormer erwartet ein RGB-uint8-Bild.")

        with self._lock:
            helper = self.face_helper
            helper.clean_all()
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            helper.read_image(bgr_image)
            face_count = helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5,
            )
            if face_count == 0:
                return rgb_image, 0

            helper.align_warp_face()
            for cropped_face in helper.cropped_faces:
                face_tensor = self.img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                face_tensor = face_tensor.unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    output = self.net(face_tensor, w=self.fidelity, adain=True)[0]
                restored_face = self.tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
                helper.add_restored_face(restored_face, cropped_face)

            helper.get_inverse_affine(None)
            restored_bgr = helper.paste_faces_to_input_image(upsample_img=bgr_image, draw_box=False)
            if restored_bgr is None:
                raise RuntimeError("CodeFormer hat kein Ergebnisbild erzeugt.")
            return cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB), face_count
