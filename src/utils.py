import numpy as np


def bbox_cut(dets: np.ndarray) -> None:
    """Cut off the face bbox and save"""
    faces = []
    for det in (dets if dets is not None else []):
        x, y, w, h = det[0:4].astype(np.int32)
        face = (x, y, x + w , y + h)
        faces.append(face)
    return faces
