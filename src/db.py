import os
import faiss
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json


class DBIndexEnum(Enum):
    EUCLIDIAN = faiss.IndexFlatL2


@dataclass
class VectorDB:
    index: DBIndexEnum
    size: int = 2622

    def __post_init__(self):
        self._db = {}
        self._idx = 0
        self._index = self.index.value(self.size)

    def _increase_idx(self) -> int:
        self._idx += 1
        return self._idx

    def get_total(self):
        return self._index.ntotal

    def append(self, key, embedding):
        self._db[str(self._idx)] = key
        self._index.add(embedding)
        self._increase_idx()

    def get_id(self, idx):
        name = self._db.get(str(idx))
        return name

    def save(self, dir_path: str, filename: str = "vecdb") -> None:
        os.makedirs(dir_path, exist_ok=True)
        dir_path = Path(dir_path)
        faiss.write_index(self._index, str(dir_path / f"{filename}.index"))
        with open(dir_path / f"{filename}.json", "w", encoding="utf-8") as outfile:
            json.dump(self._db, outfile, ensure_ascii=False)

    def load(self, dir_path, filename: str = "vecdb"):
        dir_path = Path(dir_path)
        self._index = faiss.read_index(str(dir_path / f"{filename}.index"))
        with open(dir_path / f"{filename}.json", encoding="utf-8") as json_data:
            self._db = json.load(json_data)
        self._idx = len(self._db.keys())


if __name__ == "__main__":
    db = VectorDB(DBIndexEnum.EUCLIDIAN, 1)
    embedding = np.expand_dims(np.array([1], dtype="f"), axis=0)
    db.append("Sample", embedding)
    db.save("db")
