#!/usr/bin/env python3
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="outputs/quantize")
args = parser.parse_args()

path8 = f"{args.dir}/int_export_W8A8.pt"
path4 = f"{args.dir}/int_export_W4A4.pt"

print(f"[INFO] Loading:\n  {path8}\n  {path4}")

w8 = torch.load(path8, map_location="cpu")
w4 = torch.load(path4, map_location="cpu")

print("\n=== CONFIG CHECK ===")
print("W8 config:", w8["config"])
print("W4 config:", w4["config"])

print("\n=== SAMPLE WEIGHT ENTRY CHECK ===")
k8, rec8 = next(iter(w8["weights"].items()))
k4, rec4 = next(iter(w4["weights"].items()))

print("Sample key W8:", k8)
print("Sample key W4:", k4)
print("W8 scale mean:", rec8["scale"].float().mean().item())
print("W4 scale mean:", rec4["scale"].float().mean().item())

print("\n=== L2 DIFF ACROSS ALL WEIGHTS ===")
total = 0
diff_count = 0
for k in w8["weights"]:
    if k not in w4["weights"]:
        continue
    w8_w = w8["weights"][k]["int_weight"].float()
    w4_w = w4["weights"][k]["int_weight"].float()
    l2 = (w8_w - w4_w).pow(2).sum().sqrt().item()
    total += 1
    if l2 > 0:
        diff_count += 1

print(f"Compared {total} entries, diff>0 entries = {diff_count}")
print("[DONE]")

