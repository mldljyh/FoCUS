import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def scene_object(item):
    tag = item.get("tags")
    if tag == "main":
        tag = "foreground"
    relations = item.get("relation") or []
    if isinstance(relations, str):
        relations = [relations]
    return {
        "category": item["category"],
        "description": item.get("extend_description") or item.get("short_description", ""),
        "relations": relations,
        "tag": tag,
    }


class SceneDataset(Dataset):
    def __init__(self, rows, image_dir):
        self.rows = rows
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_dir / row["image"]) as image:
            return {"image": image.convert("RGB"), "objects": row["objects"]}


def load_datasets(annotations, image_dir, validation_fraction, split_seed):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    with Path(annotations).open(encoding="utf-8") as stream:
        rows = [
            {"image": row["image"], "objects": [scene_object(obj) for obj in row["info"]]}
            for row in json.load(stream)
        ]
    random.Random(split_seed).shuffle(rows)
    boundary = int(len(rows) * (1 - validation_fraction))
    if not boundary:
        raise ValueError("The split leaves no training images")
    train = SceneDataset(rows[:boundary], image_dir)
    valid = SceneDataset(rows[boundary:], image_dir) if boundary < len(rows) else None
    return train, valid
