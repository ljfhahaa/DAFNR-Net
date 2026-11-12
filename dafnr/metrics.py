import numpy as np
from scipy import ndimage


def _confusion(pred: np.ndarray, gt: np.ndarray):
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = pred.size - tp - fp - fn
    return tp, fp, fn, tn


def region_metrics(pred: np.ndarray, gt: np.ndarray):
    tp, fp, fn, tn = _confusion(pred, gt)
    dice = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    prec = (1.0 if fn == 0 else 0.0) if tp + fp == 0 else tp / (tp + fp)
    acc = (tp + tn) / pred.size
    return dict(dice=dice, iou=iou, prec=prec, acc=acc)


def _contour(mask: np.ndarray) -> np.ndarray:

    return np.logical_and(mask, ~ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0))


def hd95(pred: np.ndarray, gt: np.ndarray, empty_value: float = None) -> float:






    if empty_value is None:
        empty_value = float(np.hypot(*pred.shape))
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return empty_value
    cp, cg = _contour(pred), _contour(gt)
    dt_g = ndimage.distance_transform_edt(~cg)
    dt_p = ndimage.distance_transform_edt(~cp)
    d_pg, d_gp = dt_g[cp], dt_p[cg]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def _boundary_band(mask: np.ndarray, d: int) -> np.ndarray:



    if not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1, mode="constant")
    eroded = ndimage.binary_erosion(padded, structure=np.ones((3, 3)), iterations=d, border_value=0)[1:-1, 1:-1]
    return np.logical_and(mask, ~eroded)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:

    d = max(1, int(round(dilation_ratio * np.hypot(*pred.shape))))
    bp, bg = _boundary_band(pred, d), _boundary_band(gt, d)
    union = np.logical_or(bp, bg).sum()
    return 1.0 if union == 0 else float(np.logical_and(bp, bg).sum() / union)


def evaluate_pair(prob: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict:

    pred, gt = prob >= threshold, gt.astype(bool)
    m = region_metrics(pred, gt)
    m["biou"] = boundary_iou(pred, gt)
    m["hd95"] = hd95(pred, gt)
    return m


METRIC_NAMES = ["dice", "iou", "prec", "acc", "biou", "hd95"]
