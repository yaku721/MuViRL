import cv2
import numpy as np
import torch


def load_face_masks(path, width, height):
    landmarks = np.load(path)['face_landmark_2d']
    masks = []
    border = max(1, round(min(width, height)*.01))
    kernel = np.ones((border*2+1, border*2+1), np.uint8)
    for points in landmarks:
        mask = np.zeros((height, width), np.uint8)
        if np.isfinite(points).all() and (points[:27, 2] > 0).all():
            xy = points[:27, :2]*[width, height]
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(xy).astype(np.int32)), 1)
            mask = cv2.erode(mask, kernel)
        masks.append(mask)
    return torch.from_numpy(np.asarray(masks, dtype=bool)).cuda()
