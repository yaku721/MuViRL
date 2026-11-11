import numpy as np


def load_mouth_landmarks(path):
    points = np.load(path)['face_landmark_2d'][:, 48:68]
    coords = points[:, :, :2]
    valid = np.isfinite(coords).all(axis=(1, 2)) & ((coords >= 0) & (coords <= 1)).all(axis=(1, 2))
    if points.shape[2] > 2:
        valid &= (points[:, :, 2] > 0).all(axis=1)
    valid &= (np.ptp(coords, axis=1) > 0).all(axis=1)
    if not valid.any():
        raise ValueError(f'No valid normalized mouth landmarks: {path}')
    return coords, valid


def mouth_polygon_masks(points, width, height, min_pixels=16):
    import cv2

    pixels = points * [width, height]
    outer = np.zeros((height, width), np.uint8)
    inner = np.zeros_like(outer)
    cv2.fillPoly(outer, [np.rint(pixels[:12]).astype(np.int32)], 1)
    cv2.fillPoly(inner, [np.rint(pixels[12:]).astype(np.int32)], 1)
    inner &= outer
    ring = outer.astype(bool) & ~inner.astype(bool)
    if ring.sum() < min_pixels:
        ring[:] = False
    if inner.sum() < min_pixels or abs(cv2.contourArea(pixels[12:].astype(np.float32))) < min_pixels:
        inner[:] = 0
    return np.stack([ring, inner.astype(bool)])
