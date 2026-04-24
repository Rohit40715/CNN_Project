import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class KITTIDepthDataset(Dataset):
    def __init__(self, images_root, depths_root, image_size=None, depth_scale=256.0):
        self.images_root = Path(images_root)
        self.depths_root = Path(depths_root)
        self.image_size = image_size
        self.depth_scale = depth_scale

        self.samples = []
        self._build_index()

        if not self.samples:
            raise RuntimeError(
                f"No image-depth pairs found. images_root={self.images_root} depths_root={self.depths_root}"
            )

    def _iter_drive_dirs(self, root_path):
        # Accept either direct drive folders or one extra nesting level (e.g., train/val/date).
        for child in sorted(root_path.iterdir()):
            if not child.is_dir():
                continue

            if (child / "image_02").exists() or (child / "proj_depth").exists():
                yield child
                continue

            for grandchild in sorted(child.iterdir()):
                if not grandchild.is_dir():
                    continue
                if (grandchild / "image_02").exists() or (grandchild / "proj_depth").exists():
                    yield grandchild

    def _build_index(self):
        image_drives = {
            drive_dir.name: drive_dir for drive_dir in self._iter_drive_dirs(self.images_root)
        }
        depth_drives = {
            drive_dir.name: drive_dir for drive_dir in self._iter_drive_dirs(self.depths_root)
        }

        common_drive_names = sorted(set(image_drives.keys()) & set(depth_drives.keys()))

        for drive_name in common_drive_names:
            image_drive_dir = image_drives[drive_name]
            depth_drive_dir = depth_drives[drive_name]

            img_dir = image_drive_dir / "image_02" / "data"
            dep_dir = depth_drive_dir / "proj_depth" / "groundtruth" / "image_02"

            if not img_dir.exists() or not dep_dir.exists():
                continue

            img_files = {p.name: p for p in img_dir.glob("*.png")}
            dep_files = {p.name: p for p in dep_dir.glob("*.png")}

            common_files = sorted(set(img_files.keys()) & set(dep_files.keys()))

            for fname in common_files:
                self.samples.append(
                    {
                        "image_path": img_files[fname],
                        "depth_path": dep_files[fname],
                        "drive": drive_name,
                        "frame": fname,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def _resize(self, image, depth):
        if self.image_size is None:
            return image, depth

        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, depth

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(str(sample["depth_path"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth file not found: {sample['depth_path']}")
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        image, depth = self._resize(image, depth)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std
        depth = depth.astype(np.float32) / self.depth_scale
        mask = (depth > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, depth, mask, sample["drive"], sample["frame"]


# Backward-compatible alias used by train.py.
KITTIDataset = KITTIDepthDataset