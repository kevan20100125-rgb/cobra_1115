#!/usr/bin/env python
import os
import torch

from cobra.quantize.runtime.load_quantized_vlm import load_quantized_cobra_vlm


def main():
    COBRA_ROOT = "/work/asdf1234/cobra_1115"
    VLM_EVAL_ROOT = "/work/asdf1234/vlm-evaluation"
    BITS = "W2A2"  # 可改成 W4A4 / W8A8 測試

    # 讓 script 自包含：若外面沒設，就預設用 cobra+3b
    os.environ.setdefault("COBRA_MODEL_ID_OR_PATH", "cobra+3b")

    pct_hi_lo_path = os.path.join(COBRA_ROOT, "outputs/quantize", f"pct_hi_lo_{BITS}.pt")
    int_export_path = os.path.join(COBRA_ROOT, "outputs/quantize", f"int_export_{BITS}.pt")

    # 讀 HF token（和 evaluate.py 一樣）
    hf_token_path = os.path.join(VLM_EVAL_ROOT, ".hf_token")
    with open(hf_token_path, "r") as f:
        hf_token = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[DEBUG] Loading quantized Cobra with bits={BITS} ...")
    model = load_quantized_cobra_vlm(
        bits=BITS,
        pct_hi_lo_path=pct_hi_lo_path,
        # 關鍵：永遠傳字串，讓 _maybe_check_int_export 自己檢查是否存在 / 為空
        int_export_path=int_export_path,
        hf_token=hf_token,
        base_dtype=torch.bfloat16,
        device=device,
    )

    # ====== Sanity Check 1: fake quant flag 總覽 ======
    total_modules = 0
    has_use_act = 0
    has_use_weight = 0
    act_on = 0
    weight_on = 0

    for m in model.modules():
        total_modules += 1
        if hasattr(m, "use_act_quant"):
            has_use_act += 1
            if getattr(m, "use_act_quant"):
                act_on += 1
        if hasattr(m, "use_weight_quant"):
            has_use_weight += 1
            if getattr(m, "use_weight_quant"):
                weight_on += 1

    print("========== Fake-Quant Flag Summary (debug script) ==========")
    print(f"Total modules: {total_modules}")
    print(f"Modules with `use_act_quant` attr: {has_use_act}")
    print(f"Modules with `use_weight_quant` attr: {has_use_weight}")
    print(f"Modules where use_act_quant == True: {act_on}")
    print(f"Modules where use_weight_quant == True: {weight_on}")
    print("============================================================")

    # ====== Sanity Check 2: 有 quantizer 且 scale 不為 None 的覆蓋率 ======
    quant_modules = 0
    act_scale_non_none = 0
    weight_scale_non_none = 0

    for m in model.modules():
        aq = getattr(m, "act_quantizer", None)
        wq = getattr(m, "weight_quantizer", None)

        if aq is not None or wq is not None:
            quant_modules += 1

        if aq is not None and getattr(aq, "scale", None) is not None:
            act_scale_non_none += 1

        if wq is not None and getattr(wq, "scale", None) is not None:
            weight_scale_non_none += 1

    print("\n========== Quantizer Scale Coverage ==========")
    print(f"Modules with any quantizer (act/weight): {quant_modules}")
    print(f"Modules with act_quantizer.scale != None: {act_scale_non_none}")
    print(f"Modules with weight_quantizer.scale != None: {weight_scale_non_none}")
    print("==============================================")

    # ====== Sanity Check 2b: 特別觀察 lm_head / projector 幾顆（calibrator 有 log 的） ======
    print("\n========== Special Modules (lm_head / projector) ==========")
    for name, m in model.named_modules():
        if (
            "lm_head" in name
            or name.startswith("projector.projector.0")
            or name.startswith("projector.projector.1")
            or name.startswith("projector.projector.2")
            or name.startswith("projector.projector.3")
            or name.startswith("projector.projector.4")
        ):
            aq = getattr(m, "act_quantizer", None)
            wq = getattr(m, "weight_quantizer", None)
            print(f"[{name}]")
            if aq is None:
                print("  act_quantizer: None")
            else:
                print(
                    "  act_quantizer: "
                    f"n_bits={getattr(aq, 'n_bits', None)}, "
                    f"scale={getattr(aq, 'scale', None)}, "
                    f"zero_point={getattr(aq, 'zero_point', None)}"
                )
            if wq is None:
                print("  weight_quantizer: None")
            else:
                print(
                    "  weight_quantizer: "
                    f"n_bits={getattr(wq, 'n_bits', None)}, "
                    f"scale={getattr(wq, 'scale', None)}, "
                    f"zero_point={getattr(wq, 'zero_point', None)}"
                )
    print("===========================================================")

    # ====== Sanity Check 3: 抽一顆 Quant* module 看內部 ======
    sample = None
    for m in model.modules():
        if hasattr(m, "act_quantizer") and hasattr(m, "use_act_quant"):
            sample = m
            break

    if sample is None:
        print("\n[WARN] No module with `act_quantizer` + `use_act_quant` found. Something is off.")
        return

    print("\n========== Sample Quant* Module Detail ==========")
    print("Module type:", sample.__class__.__name__)
    print(f"use_act_quant   = {getattr(sample, 'use_act_quant', None)}")
    print(f"use_weight_quant= {getattr(sample, 'use_weight_quant', None)}")

    aq = getattr(sample, "act_quantizer", None)
    wq = getattr(sample, "weight_quantizer", None)

    if aq is None:
        print("act_quantizer   = None")
    else:
        print("act_quantizer:")
        for attr in ["enable", "dynamic", "n_bits", "scale", "zero_point"]:
            val = getattr(aq, attr, "<missing>")
            print(f"  - {attr}: {val}")

    if wq is None:
        print("weight_quantizer= None")
    else:
        print("weight_quantizer:")
        for attr in ["enable", "dynamic", "n_bits", "scale", "zero_point"]:
            val = getattr(wq, attr, "<missing>")
            print(f"  - {attr}: {val}")

    print("=============================================")


if __name__ == "__main__":
    main()

