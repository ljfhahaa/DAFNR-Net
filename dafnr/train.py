import argparse
import csv
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from dafnr.data import SegDataset
from dafnr.losses import DAFNRLoss
from dafnr.models import DAFNRNet
from dafnr.utils import count_parameters, load_config, set_seed, worker_init_fn


def build_model(cfg: dict, pretrained=None) -> DAFNRNet:
    m = dict(cfg["model"])
    if pretrained is not None:
        m["pretrained"] = pretrained
    return DAFNRNet(**m)


@torch.no_grad()
def validate(model, loader, device):

    model.eval()
    scores = []
    for batch in loader:
        prob = torch.sigmoid(model(batch["image"].to(device))["seg"])
        pred, gt = (prob >= 0.5).float(), batch["mask"].to(device)
        inter = (pred * gt).flatten(1).sum(1)
        total = pred.flatten(1).sum(1) + gt.flatten(1).sum(1)
        dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))
        scores.extend(dice.tolist())
    return sum(scores) / len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="override config entries, e.g. train.seed=1")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, t = cfg["data"], cfg["train"]

    gen = set_seed(t["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = os.path.basename(os.path.normpath(d["root"]))
    out_dir = os.path.join(t["output"], name, f"seed{t['seed']}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    train_set = SegDataset(d["root"], "train", d["size"], train=True, mask_suffix=d["mask_suffix"],
                           aug_prob=t["aug_prob"], rotation=t["rotation"], scale=tuple(t["scale"]))
    val_set = SegDataset(d["root"], "val", d["size"], mask_suffix=d["mask_suffix"])
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                              num_workers=d["num_workers"], pin_memory=True, generator=gen,
                              worker_init_fn=worker_init_fn, persistent_workers=d["num_workers"] > 0)
    val_loader = DataLoader(val_set, batch_size=t["batch_size"], shuffle=False, num_workers=d["num_workers"])

    model = build_model(cfg).to(device)
    print(f"DAFNR-Net: {count_parameters(model):.2f}M parameters | train {len(train_set)} | val {len(val_set)}")
    criterion = DAFNRLoss(t["lambda_bnd"])
    optim = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t["epochs"], eta_min=t["min_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=t["amp"] and device.type == "cuda")

    best, log_path = -1.0, os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "loss", "bce", "dice_loss", "bnd_loss", "val_dice", "time_s"])
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        start, total, parts, n = time.time(), 0.0, {"bce": 0.0, "dice": 0.0, "bnd": 0.0}, 0
        for batch in train_loader:
            img, mask = batch["image"].to(device, non_blocking=True), batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=t["amp"] and device.type == "cuda"):
                out = model(img)
            loss, logs = criterion({k: v.float() for k, v in out.items()}, mask)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item(); n += 1
            for k in parts:
                parts[k] += logs[k]
        lr = optim.param_groups[0]["lr"]
        sched.step()
        val_dice = validate(model, val_loader, device)
        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice, "config": cfg},
                       os.path.join(out_dir, "best.pth"))
        elapsed = time.time() - start
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.3e}", f"{total / n:.4f}", f"{parts['bce'] / n:.4f}",
                                    f"{parts['dice'] / n:.4f}", f"{parts['bnd'] / n:.4f}", f"{val_dice:.4f}",
                                    f"{elapsed:.1f}"])
        print(f"epoch {epoch:3d}/{t['epochs']} | loss {total / n:.4f} | val Dice {val_dice:.4f} "
              f"| best {best:.4f} | {elapsed:.0f}s")
    torch.save({"model": model.state_dict(), "epoch": t["epochs"], "config": cfg}, os.path.join(out_dir, "last.pth"))


if __name__ == "__main__":
    main()
