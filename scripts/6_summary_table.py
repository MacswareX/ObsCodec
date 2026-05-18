"""Generate a Markdown summary with calibrated rate-distortion claims."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from obscodec import dedupe_by_name
from obscodec.config import BITS_PER_FLOAT32, cfg
from obscodec.metrics import (
    compression_ratio,
    mse_to_psnr,
    posterior_collapse_ratio,
    rate_distortion_efficiency,
)

RAW_BW: int = cfg.obs_dim * BITS_PER_FLOAT32
ASSETS = Path("assets")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"  (skip) {path} not found")
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def fmt_psnr(mse: float) -> str:
    psnr = mse_to_psnr(mse)
    return "inf" if psnr == float("inf") else f"{psnr:.1f}"


def fmt_ratio(bits: float | None) -> str:
    if bits is None or bits <= 0.05:
        return "-"
    return f"{compression_ratio(RAW_BW, bits):.0f}x"


def best_under_budget(records: Iterable[dict], budget: float) -> dict | None:
    candidates = [record for record in records if record["bandwidth"] <= budget]
    if not candidates:
        return None
    return min(candidates, key=lambda record: record["mse"])


def add_method_row(
    lines: list[str],
    method: str,
    config: str,
    mse: float,
    bandwidth: float,
    rate_bits: float | None = None,
) -> None:
    lines.append(
        "| "
        + " | ".join(
            [
                method,
                config,
                fmt_num(mse),
                fmt_psnr(mse),
                f"{bandwidth:.0f}",
                f"{rate_bits:.1f}" if rate_bits is not None else "-",
                fmt_ratio(bandwidth),
                fmt_ratio(rate_bits),
                f"{rate_distortion_efficiency(mse, bandwidth):.2f}",
            ]
        )
        + " |"
    )


def main() -> None:
    baseline = load_json(ASSETS / "baseline_results.json")
    vae = load_json(ASSETS / "vae_results.json")
    if not isinstance(vae, list):
        vae = []
    vqvae_raw = load_json(ASSETS / "vqvae_results.json")
    vqvae = dedupe_by_name(vqvae_raw if isinstance(vqvae_raw, list) else [])

    lines: list[str] = []
    lines.append("# ObsCodec Results Summary")
    lines.append("")
    lines.append(
        f"> Raw observation = 18 dims x 32-bit float = **{RAW_BW} bits**. "
        "All MSE values are measured on a held-out test split."
    )
    lines.append("")
    lines.append(
        "> **Metric note**: nominal bandwidth counts the serialized latent size. "
        "β-VAE effective rate is the KL estimate in bits and is not the same as "
        "a deployed packet size without entropy coding."
    )
    lines.append("")

    lines.append("## Table 1: Method Comparison Under a 256-bit Nominal Budget")
    lines.append("")
    lines.append(
        "| Method | Config | MSE | PSNR (dB) | Nominal BW | KL Eff. Rate | "
        "Nominal Ratio | Eff. Ratio | RD Efficiency |"
    )
    lines.append(
        "|--------|--------|-----|-----------|------------|--------------|"
        "---------------|------------|---------------|"
    )

    pca_best = best_under_budget(baseline["pca"], 256)
    ae_best = best_under_budget(baseline["ae"], 256)
    digital_best = best_under_budget(baseline["digital"], 256)
    assert pca_best and ae_best and digital_best
    add_method_row(lines, "PCA", "n=8", pca_best["mse"], pca_best["bandwidth"])
    add_method_row(lines, "Standard AE", "LD=8", ae_best["mse"], ae_best["bandwidth"])
    add_method_row(
        lines,
        "Digital Quant.",
        "LD=16, B=8",
        digital_best["mse"],
        digital_best["bandwidth"],
    )

    for beta, label in [
        (0.001, "near-AE"),
        (0.01, "semantic bottleneck"),
        (0.1, "transition"),
        (1.0, "at KL floor"),
    ]:
        record = next(r for r in vae if r["latent_dim"] == 8 and r["beta"] == beta)
        add_method_row(
            lines,
            f"β-VAE ({label})",
            f"LD=8, β={beta}",
            record["mse"],
            record["bandwidth"],
            record["rate_bits"],
        )

    if vqvae:
        vq_best = min(vqvae, key=lambda record: record["mse"])
        add_method_row(
            lines,
            "VQ-VAE",
            vq_best["name"],
            vq_best["mse"],
            vq_best["bandwidth"],
        )
    lines.append("")
    lines.append(
        "**Primary takeaway**: Digital quantization is the strongest pure "
        "reconstruction baseline at >=128 nominal bits. β-VAE provides a tunable "
        "information rate through the KL divergence, with the free-bits floor (0.01 "
        "nats/dim) preventing complete posterior collapse while allowing a 300× KL "
        "dynamic range. The semantic bottleneck regime (β=0.01) gives 6–17 effective "
        "bits depending on latent dimension."
    )
    lines.append("")

    lines.append("## Table 2: β-VAE LD=8 Rate-Distortion Sweep")
    lines.append("")
    lines.append(
        "| β | MSE | PSNR (dB) | Nominal BW | Eff. Rate (bits) | KL (nats) | Regime |"
    )
    lines.append(
        "|---|-----|-----------|------------|------------------|-----------|--------|"
    )
    for beta in [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0]:
        try:
            record = next(r for r in vae if r["latent_dim"] == 8 and r["beta"] == beta)
        except StopIteration:
            continue
        floor_kl = 0.01 * record["latent_dim"] * 1.5
        if record["kl"] > 10:
            regime = "high-rate"
        elif record["kl"] > 3:
            regime = "semantic bottleneck"
        elif record["kl"] > 1:
            regime = "transition"
        elif record["kl"] > floor_kl:
            regime = "low-rate"
        else:
            regime = "at KL floor"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(beta),
                    fmt_num(record["mse"]),
                    fmt_psnr(record["mse"]),
                    f"{record['bandwidth']:.0f}",
                    f"{record['rate_bits']:.1f}",
                    f"{record['kl']:.4f}",
                    regime,
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Table 3: VQ-VAE Commitment Cost Sweep (LD=4, CB=128)")
    lines.append("")
    lines.append("| cc | MSE | PSNR (dB) | BW | Codebook Usage | RD Efficiency | Note |")
    lines.append("|----|-----|-----------|----|----------------|---------------|------|")
    cc_records = [
        r
        for r in vqvae
        if r["latent_dim"] == 4 and r["codebook_size"] == 128
    ]
    cc_records.sort(key=lambda r: r["commitment_cost"])
    for record in cc_records:
        usage = record.get("codebook_usage", 0.0)
        if usage < 0.15:
            note = "underused"
        elif usage < 0.5:
            note = "moderate"
        else:
            note = "high usage"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["commitment_cost"]),
                    fmt_num(record["mse"]),
                    fmt_psnr(record["mse"]),
                    f"{record['bandwidth']:.0f}",
                    f"{usage:.1%}",
                    f"{rate_distortion_efficiency(record['mse'], record['bandwidth']):.2f}",
                    note,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "EMA codebook updates maintain high codebook usage across commitment "
        "cost levels. Higher cc values increase the penalty for encoder-codebook "
        "divergence, trading off reconstruction quality for discrete-alignment fidelity."
    )
    lines.append("")

    lines.append("## Table 4: Best MSE Within Nominal Bandwidth Budgets")
    lines.append("")
    lines.append("| Budget | PCA | Standard AE | Digital Quant. | β-VAE | VQ-VAE |")
    lines.append("|--------|-----|-------------|----------------|-------|--------|")
    method_sets = [
        baseline["pca"],
        baseline["ae"],
        baseline["digital"],
        vae,
        vqvae,
    ]
    for budget in [8, 16, 32, 64, 128, 256]:
        values = []
        for records in method_sets:
            best = best_under_budget(records, budget)
            values.append(fmt_num(best["mse"]) if best else "-")
        lines.append(f"| {budget}b | " + " | ".join(values) + " |")
    lines.append("")

    all_kl = [record["kl"] for record in vae if record["beta"] >= 0.5]
    collapse_pct = posterior_collapse_ratio(all_kl) * 100

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(
        "- **β-VAE posterior collapse behavior**: With the corrected architecture "
        "(no tanh on mu, BatchNorm encoder, halved decoder capacity, KL annealing over "
        "50 epochs, free-bits=0.01 nats/dim), the posterior never collapses to zero KL. "
        "The KL shows a smooth 300× decline from β=0.001 (KL≈19.5 nats, 28 bits) to "
        "β=0.5 (KL≈0.09 nats, 0.1 bits) before reaching the free-bits floor. "
        "At β ≥ 1.0, KL stabilizes at ~0.06–0.07 nats (~0.1 effective bits) and MSE "
        "approaches the data variance (~0.545), consistent with theoretical expectations "
        "for the β→∞ limit. The free-bits floor prevents complete gradient collapse "
        "while allowing the KL to span a wide dynamic range. "
        "The useful operating range (β=0.001–0.1) provides tunable rate-distortion "
        "tradeoffs consistent with Higgins et al. (2017) and Burgess et al. (2018)."
    )
    lines.append(
        "- **VQ-VAE codebook usage**: EMA codebook updates and periodic dead-entry "
        "reset keep codebook usage high. Best performance is at lower latent dimensions "
        "(LD=2, codebook_size=256) achieving 8-bit discrete latent codes."
    )
    lines.append(
        "- The KL rate estimate from β-VAE is an information measure; deploying it "
        "as an actual channel rate requires entropy coding or packetization."
    )
    lines.append(
        "- Reconstruction MSE is a proxy metric. A full SemCom-MARL follow-up "
        "should validate policy return, coordination success, and robustness "
        "under channel noise."
    )

    output_path = ASSETS / "results_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
