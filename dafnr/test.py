import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dafnr.data import SegDataset
from dafnr.metrics import METRIC_NAMES, evaluate_pair
from dafnr.models import DAFNRNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=None, help="dataset folder (default: the one stored in the checkpoint)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-pred", action="store_true", help="save binary predictions as PNG")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = dict(cfg["model"]); m["pretrained"] = False
    model = DAFNRNet(**m)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    d = cfg["data"]
    root = args.root or d["root"]
    ds = SegDataset(root, args.split, d["size"], mask_suffix=d["mask_suffix"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=d.get("num_workers", 4))

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    pred_dir = os.path.join(out_dir, f"pred_{args.split}")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in loader:
            prob = torch.sigmoid(model(batch["image"].to(device))["seg"]).cpu().numpy()[:, 0]
            gt = batch["mask"].numpy()[:, 0] > 0.5
            for sid, p, g in zip(batch["id"], prob, gt):
                r = evaluate_pair(p, g)
                rows.append({"image": sid, **r})
                if args.save_pred:
                    Image.fromarray(((p >= 0.5) * 255).astype(np.uint8)).save(os.path.join(pred_dir, f"{sid}.png"))

    with open(os.path.join(out_dir, f"{args.split}_per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image"] + METRIC_NAMES)
        w.writeheader()
        for r in rows:
            w.writerow({"image": r["image"], **{k: f"{r[k]:.6f}" for k in METRIC_NAMES}})

    summary = {k: float(np.mean([r[k] for r in rows])) * (1 if k == "hd95" else 100) for k in METRIC_NAMES}
    summary["n_images"] = len(rows)
    with open(os.path.join(out_dir, f"{args.split}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(" | ".join(f"{k} {summary[k]:.2f}" for k in METRIC_NAMES), f"| n={len(rows)}")


if __name__ == "__main__":
    main()
