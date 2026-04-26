#!/usr/bin/env python3
"""
Çok Kaynaklı Akustik Pipeline Değerlendirme

İki veya daha fazla eş zamanlı ses kaynağı olan kayıtlar için değerlendirme aracı.
Her algoritmada top-N yerel tepe noktası bulunur, opsiyonel ground-truth ile eşleştirilir,
precision/recall/F1 hesaplanır ve B&W uyumlu figürler üretilir.

MUSIC algoritması n_sources parametresiyle çalışır (doğru subspace boyutu).

Kullanım örnekleri:
    # Ground truth olmadan (sadece peak tespiti):
    python scripts/evaluate_pipeline_multisource.py \\
        --recording data/recordings/recording_20260415_111444/recording_20260415_111444.wav \\
        --n-sources 2 \\
        --algorithms DAS MVDR MUSIC Hybrid

    # Ground truth ile (iki kaynak):
    python scripts/evaluate_pipeline_multisource.py \\
        --recording data/recordings/recording_20260415_111444/recording_20260415_111444.wav \\
        --ground-truths -0.71 0.0 0.71  0.71 0.0 0.71 \\
        --n-sources 2 \\
        --algorithms DAS MVDR MUSIC Hybrid \\
        --output results/multisource/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.beamforming import (
    BeamformingConfig,
    load_mic_geometry,
    create_focus_grid,
    das_beamformer_realtime,
    mvdr_beamformer_realtime,
    music_beamformer_realtime,
    power_to_db,
    _precompute_distances,
)
from src.evaluation.metrics import (
    angular_localization_error,
    measure_pipeline_latency,
    detection_precision_recall,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_top_n_peaks(
    power_map: np.ndarray,
    grid_points: np.ndarray,
    n: int,
    min_sep_deg: float = 10.0,
) -> List[Tuple[np.ndarray, float]]:
    """
    Güç haritasında minimum açısal ayrımı olan top-N yerel tepe noktası bul.

    Args:
        power_map:    (n_points,) lineer güç değerleri
        grid_points:  (n_points, 3) grid koordinatları
        n:            Kaç tepe noktası aranacak
        min_sep_deg:  Tepeler arası minimum açısal ayrım (derece)

    Returns:
        [(xyz, power_linear), ...] en güçlüden başlayarak sıralı
    """
    remaining = power_map.copy().astype(float)
    peaks: List[Tuple[np.ndarray, float]] = []

    for _ in range(n):
        idx = int(np.argmax(remaining))
        if remaining[idx] <= 0:
            break
        peak_xyz = grid_points[idx].copy()
        peak_power = float(remaining[idx])
        peaks.append((peak_xyz, peak_power))

        # Bu peak'in çevresindeki noktaları maskelele (açısal bölge)
        for j in range(len(grid_points)):
            angle = angular_localization_error(grid_points[j], peak_xyz)
            if angle < min_sep_deg:
                remaining[j] = 0.0

    return peaks


def match_peaks_to_gt(
    detected: List[Tuple[np.ndarray, float]],
    ground_truths: List[np.ndarray],
) -> List[Tuple[int, int, float]]:
    """
    Greedy matching: her tespit edilen peak → en yakın eşleşmemiş GT.

    Returns:
        [(det_idx, gt_idx, angular_error_deg), ...]
    """
    matched: List[Tuple[int, int, float]] = []
    unmatched_gt = set(range(len(ground_truths)))

    for di, (det_xyz, _) in enumerate(detected):
        if not unmatched_gt:
            break
        best_angle = np.inf
        best_gi = -1
        for gi in unmatched_gt:
            angle = angular_localization_error(det_xyz, ground_truths[gi])
            if angle < best_angle:
                best_angle = angle
                best_gi = gi
        if best_gi >= 0:
            matched.append((di, best_gi, float(best_angle)))
            unmatched_gt.discard(best_gi)

    return matched


def project_gt_to_focus_plane(gt_xyz: np.ndarray, z_focus: float) -> Tuple[float, float]:
    """
    GT yön vektörünü odak düzlemine projeksiyon ile (x, y) koordinatı hesapla.

    Odak düzlemi Z=z_focus'ta. GT noktası (gx, gy, gz)'den başlayan ray:
        x_proj = gx * z_focus / gz
        y_proj = gy * z_focus / gz
    gz ≈ 0 ise (90° kaynak) düzlem içinde değil → sınıra klipler.
    """
    gx, gy, gz = float(gt_xyz[0]), float(gt_xyz[1]), float(gt_xyz[2])
    if abs(gz) < 1e-6:
        # 90°+ kaynak — odak düzlemini kesmez; grid X sınırına klipler
        sign_x = np.sign(gx) if gx != 0 else 1.0
        return float(sign_x * z_focus * 10), float(gy)
    return float(gx * z_focus / gz), float(gy * z_focus / gz)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_beamforming_config(cfg: dict) -> BeamformingConfig:
    bf = cfg.get("beamforming", {})
    grid = bf.get("grid", {})
    freq = bf.get("frequency", {})
    calib = cfg.get("calibration", {}).get("environment", {})
    temp = calib.get("temperature", 20)
    sound_speed = 331.3 + 0.606 * temp
    x_range = grid.get("x_range", [-1.5, 1.5])
    y_range = grid.get("y_range", [-1.5, 1.5])
    return BeamformingConfig(
        grid_size_x=x_range[1] - x_range[0],
        grid_size_y=y_range[1] - y_range[0],
        grid_resolution=grid.get("resolution", 0.05),
        focus_distance=grid.get("z", 1.0),
        freq_min=freq.get("min", 500),
        freq_max=freq.get("max", 8000),
        freq_bins=freq.get("bands", 3) * 3,
        sound_speed=sound_speed,
    )


def run_hybrid(chunk, mic_positions, grid_points, sample_rate, config,
               distances, max_freq_bins):
    try:
        from src.algorithms.beamforming import hybrid_beamformer_realtime
        result = hybrid_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances,
        )
        return result[0] if isinstance(result, tuple) else result
    except (ImportError, TypeError):
        return das_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances,
        )


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _render_multisource_ax(
    ax,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    power_db: np.ndarray,
    title: str,
    spl_offset: float,
    detected_peaks: Optional[List[Tuple[np.ndarray, float]]] = None,
    ground_truths: Optional[List[np.ndarray]] = None,
    z_focus: float = 1.0,
    dynamic_range: float = 15.0,
    cmap: str = "hot_r",
    contour_levels_db: tuple = (-12, -9, -6, -3),
    fontsize: int = 10,
):
    """
    Çok kaynaklı B&W uyumlu güç haritası.

    Tespit edilen peak'ler: numaralı beyaz daire
    Ground truth konumları: sarı X işareti (varsa)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    v_max = float(power_db.max())
    v_min = v_max - dynamic_range

    im = ax.pcolormesh(
        x_vals, y_vals, power_db,
        cmap=cmap, vmin=v_min, vmax=v_max, shading="auto",
    )

    # Kontur çizgileri
    abs_levels = sorted([v_max + d for d in contour_levels_db if v_max + d > v_min])
    if abs_levels:
        cs = ax.contour(
            x_vals, y_vals, power_db,
            levels=abs_levels,
            colors="black",
            linewidths=0.8,
            linestyles=["--", "-.", "-", "-"],
        )
        fmt = {lv: f"{lv - v_max:+.0f} dB" for lv in abs_levels}
        ax.clabel(cs, fmt=fmt, fontsize=fontsize - 2, inline=True, inline_spacing=2)

    legend_elements = []

    # Tespit edilen peak'ler (beyaz dolgulu daire, numaralı)
    if detected_peaks:
        for i, (xyz, _) in enumerate(detected_peaks):
            ax.plot(
                xyz[0], xyz[1],
                marker="o", color="white", markersize=9,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5,
            )
            ax.annotate(
                f"S{i+1}",
                xy=(xyz[0], xyz[1]), xytext=(5, 4),
                textcoords="offset points",
                fontsize=fontsize - 2, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
            )
        legend_elements.append(
            mpatches.Patch(facecolor="white", edgecolor="black", label="Tespit edilen kaynak")
        )

    # Ground truth işaretleri (sarı X)
    if ground_truths:
        for i, gt in enumerate(ground_truths):
            xp, yp = project_gt_to_focus_plane(gt, z_focus)
            ax.plot(
                xp, yp,
                marker="x", color="yellow", markersize=11,
                markeredgewidth=2.2, zorder=6,
            )
            ax.annotate(
                f"GT{i+1}",
                xy=(xp, yp), xytext=(-18, 5),
                textcoords="offset points",
                fontsize=fontsize - 2, color="yellow", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.65),
            )
        legend_elements.append(
            mpatches.Patch(facecolor="yellow", edgecolor="black", label="Ground Truth")
        )

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("dBSPL" if spl_offset > 0 else "dBFS", fontsize=fontsize - 1)
    cb.ax.tick_params(labelsize=fontsize - 2)

    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=fontsize)
    ax.set_ylabel("Y (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4, color="gray")

    if legend_elements:
        ax.legend(
            handles=legend_elements, fontsize=fontsize - 2,
            loc="upper right", framealpha=0.75,
        )


def save_multisource_figures(
    wav_stem: str,
    power_maps_db: dict,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    detected_peaks_per_alg: dict,
    ground_truths: Optional[List[np.ndarray]],
    z_focus: float,
    spl_offset: float,
    output_dir: Path,
):
    """
    Karşılaştırma figürü (2×2) + her algoritma için ayrı figür üret.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib yok — figürler üretilmedi.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    valid_algs = list(power_maps_db.keys())
    n = len(valid_algs)
    if n == 0:
        return

    # --- Karşılaştırma figürü ---
    nrows, ncols = (1, n) if n <= 2 else (2, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, alg in enumerate(valid_algs):
        _render_multisource_ax(
            axes_flat[i], x_vals, y_vals, power_maps_db[alg],
            title=alg, spl_offset=spl_offset,
            detected_peaks=detected_peaks_per_alg.get(alg),
            ground_truths=ground_truths,
            z_focus=z_focus, fontsize=10,
        )
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Çok Kaynaklı Beamforming — {wav_stem}", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    comp_path = output_dir / f"multisource_maps_{wav_stem}.png"
    fig.savefig(comp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Karşılaştırma figürü: {comp_path}")

    # --- Bireysel figürler ---
    for alg in valid_algs:
        fig_s, ax_s = plt.subplots(1, 1, figsize=(4.5, 4.0))
        _render_multisource_ax(
            ax_s, x_vals, y_vals, power_maps_db[alg],
            title=alg, spl_offset=spl_offset,
            detected_peaks=detected_peaks_per_alg.get(alg),
            ground_truths=ground_truths,
            z_focus=z_focus, fontsize=11,
        )
        plt.tight_layout()
        alg_slug = alg.lower().replace(" ", "_")
        sp = output_dir / f"multisource_{alg_slug}_{wav_stem}.png"
        fig_s.savefig(sp, dpi=300, bbox_inches="tight")
        plt.close(fig_s)
        logger.info(f"  {alg} figürü: {sp}")


# ---------------------------------------------------------------------------
# Output printing
# ---------------------------------------------------------------------------

def print_multisource_table(
    results: dict,
    ground_truths: Optional[List[np.ndarray]],
    spl_offset: float,
):
    mode = "dBSPL" if spl_offset > 0 else "dBFS"
    print("\n" + "=" * 80)
    print(f"  ÇOK KAYNAKLI DEĞERLENDİRME  [{mode}, offset={spl_offset:.1f} dB]")
    print("=" * 80)

    for alg, m in results.items():
        print(f"\n[ {alg} ]")
        lat = m["latency"]
        print(f"  Gecikme:  {lat['mean_ms']:.1f} ± {lat['std_ms']:.1f} ms")
        peaks_info = m.get("first_chunk_peaks", [])
        for pi, p in enumerate(peaks_info):
            az = p.get("azimuth_deg", "-")
            el = p.get("elevation_deg", "-")
            pdb = p.get("peak_db", "-")
            print(f"  S{pi+1}: ({p['x']:.2f}, {p['y']:.2f}, {p['z']:.2f})  "
                  f"az={az:.1f}°  el={el:.1f}°  {pdb:.1f} {mode}")
        if ground_truths and "precision_recall" in m:
            pr = m["precision_recall"]
            print(f"  Precision: {pr['precision']:.2f}  Recall: {pr['recall']:.2f}  "
                  f"F1: {pr['f1']:.2f}  (TP={pr['tp']} FP={pr['fp']} FN={pr['fn']})")
            for k, v in m.get("per_source_errors_deg", {}).items():
                print(f"  {k}:  {v:.1f}°")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_multisource(
    wav_path: Path,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    grid_shape: Tuple[int, int],
    config: BeamformingConfig,
    algorithms: List[str],
    ground_truths: Optional[List[np.ndarray]],
    n_sources: int,
    spl_offset: float,
    chunk_duration: float,
    sample_rate: int,
    max_freq_bins: int,
    min_peak_sep_deg: float,
    tolerance_deg: float,
    output_dir: Path,
    no_figures: bool = False,
) -> dict:
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), always_2d=True)
    audio = audio[:, : mic_positions.shape[0]]
    chunk_samples = int(chunk_duration * sr)
    distances = _precompute_distances(mic_positions, grid_points)

    fn_map = {
        "DAS": lambda c: das_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances,
        ),
        "MVDR": lambda c: mvdr_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances,
        ),
        "MUSIC": lambda c: music_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances,
            n_sources=n_sources,
        ),
        "Hybrid": lambda c: run_hybrid(
            c, mic_positions, grid_points, sr, config, distances, max_freq_bins
        ),
    }

    results = {}
    n_chunks = max(1, len(audio) // chunk_samples)

    # Per-chunk accumulators
    chunk_latencies: dict = {a: [] for a in algorithms if a in fn_map}
    chunk_peaks_per_alg: dict = {a: [] for a in algorithms if a in fn_map}
    chunk_pr_per_alg: dict = {a: [] for a in algorithms if a in fn_map}
    first_chunk_peaks: dict = {}
    first_power_maps_db: dict = {}

    for i in range(n_chunks):
        chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]
        if len(chunk) < chunk_samples:
            break

        for alg in algorithms:
            if alg not in fn_map:
                continue
            beamform = fn_map[alg]

            lat = measure_pipeline_latency(chunk, beamform, n_trials=3)
            chunk_latencies[alg].append(lat["mean_ms"])

            pm = beamform(chunk)
            peaks = find_top_n_peaks(pm, grid_points, n_sources, min_peak_sep_deg)

            if i == 0:
                first_chunk_peaks[alg] = peaks
                first_power_maps_db[alg] = power_to_db(pm, spl_offset=spl_offset).reshape(grid_shape)

            chunk_peaks_per_alg[alg].append(peaks)

            if ground_truths:
                det_xyzs = [p[0] for p in peaks]
                pr = detection_precision_recall(
                    det_xyzs, ground_truths, tolerance_deg=tolerance_deg
                )
                chunk_pr_per_alg[alg].append(pr)

    # Summarize per algorithm
    for alg in algorithms:
        if alg not in fn_map or not chunk_latencies[alg]:
            continue

        lats = chunk_latencies[alg]
        peak_info_list = []
        for xyz, pwr in first_chunk_peaks.get(alg, []):
            az = float(np.degrees(np.arctan2(xyz[0], xyz[2]))) if xyz[2] > 1e-6 else (
                90.0 if xyz[0] > 0 else -90.0
            )
            dist_xz = np.hypot(xyz[0], xyz[2])
            el = float(np.degrees(np.arctan2(xyz[1], dist_xz)))
            peak_info_list.append({
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "azimuth_deg": round(az, 2),
                "elevation_deg": round(el, 2),
                "peak_db": round(float(power_to_db(pwr, spl_offset=spl_offset)), 1),
            })

        m: dict = {
            "latency": {
                "mean_ms": float(np.mean(lats)),
                "std_ms": float(np.std(lats)),
                "min_ms": float(np.min(lats)),
                "max_ms": float(np.max(lats)),
            },
            "first_chunk_peaks": peak_info_list,
        }

        if ground_truths and chunk_pr_per_alg[alg]:
            pr_list = chunk_pr_per_alg[alg]
            m["precision_recall"] = {
                "precision": float(np.mean([p["precision"] for p in pr_list])),
                "recall": float(np.mean([p["recall"] for p in pr_list])),
                "f1": float(np.mean([p["f1"] for p in pr_list])),
                "tp": int(round(np.mean([p["tp"] for p in pr_list]))),
                "fp": int(round(np.mean([p["fp"] for p in pr_list]))),
                "fn": int(round(np.mean([p["fn"] for p in pr_list]))),
            }
            # Per-source angular error from first chunk matching
            first_peaks = first_chunk_peaks.get(alg, [])
            matches = match_peaks_to_gt(first_peaks, ground_truths)
            per_src = {}
            for di, gi, err in matches:
                per_src[f"S{di+1}→GT{gi+1}_deg"] = round(err, 2)
            m["per_source_errors_deg"] = per_src

        results[alg] = m
        logger.info(
            f"{alg}: lat={np.mean(lats):.1f}ms  "
            f"peaks={len(peak_info_list)}"
            + (f"  F1={m.get('precision_recall', {}).get('f1', '-'):.2f}"
               if ground_truths else "")
        )

    # Figures
    if not no_figures and first_power_maps_db:
        x_vals = grid_points[:, 0].reshape(grid_shape)
        y_vals = grid_points[:, 1].reshape(grid_shape)
        save_multisource_figures(
            wav_stem=wav_path.stem,
            power_maps_db=first_power_maps_db,
            x_vals=x_vals,
            y_vals=y_vals,
            detected_peaks_per_alg=first_chunk_peaks,
            ground_truths=ground_truths,
            z_focus=config.focus_distance,
            spl_offset=spl_offset,
            output_dir=output_dir,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Çok kaynaklı akustik pipeline değerlendirmesi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--recording", required=True, help="WAV dosyası yolu")
    parser.add_argument(
        "--ground-truths", nargs="+", type=float, metavar="V",
        default=None,
        help="Ground truth koordinatları (X1 Y1 Z1  X2 Y2 Z2 ...). "
             "Sıralı üçlü olarak girilir.",
    )
    parser.add_argument(
        "--n-sources", type=int, default=2,
        help="Tespit edilecek/beklenen kaynak sayısı",
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["DAS", "MVDR", "MUSIC", "Hybrid"],
        choices=["DAS", "MVDR", "MUSIC", "Hybrid"],
    )
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--max-freq-bins", type=int, default=10)
    parser.add_argument(
        "--min-peak-sep-deg", type=float, default=10.0,
        help="Peaks arası minimum açısal ayrım (derece)",
    )
    parser.add_argument(
        "--tolerance-deg", type=float, default=15.0,
        help="Precision/recall için eşik açısı (derece)",
    )
    parser.add_argument("--output", default="results/multisource")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    wav_path = Path(args.recording).resolve()
    if not wav_path.exists():
        logger.error(f"WAV bulunamadı: {wav_path}")
        sys.exit(1)

    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        logger.error(f"Config bulunamadı: {config_path}")
        sys.exit(1)

    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    spl_offset = float(cfg.get("calibration", {}).get("spl_offset_db", 0.0))
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 48000))
    logger.info(f"SPL offset: {spl_offset} dB")

    geom_file = project_root / cfg.get("microphone", {}).get(
        "geometry_file", "config/micgeom.xml"
    )
    mic_positions = load_mic_geometry(str(geom_file))
    bf_config = build_beamforming_config(cfg)
    grid_points, grid_shape = create_focus_grid(bf_config)
    logger.info(f"Grid: {grid_shape[0]}×{grid_shape[1]} = {len(grid_points)} nokta")

    # Ground truths: düz liste → üçlülere böl
    ground_truths: Optional[List[np.ndarray]] = None
    if args.ground_truths:
        vals = args.ground_truths
        if len(vals) % 3 != 0:
            logger.error("--ground-truths: sayılar 3'ün katı olmalı (X Y Z üçlüleri)")
            sys.exit(1)
        ground_truths = [
            np.array(vals[i: i + 3], dtype=float) for i in range(0, len(vals), 3)
        ]
        logger.info(f"{len(ground_truths)} ground-truth kaynağı yüklendi.")
        for gi, gt in enumerate(ground_truths):
            logger.info(f"  GT{gi+1}: {gt}")

    results = evaluate_multisource(
        wav_path=wav_path,
        mic_positions=mic_positions,
        grid_points=grid_points,
        grid_shape=grid_shape,
        config=bf_config,
        algorithms=args.algorithms,
        ground_truths=ground_truths,
        n_sources=args.n_sources,
        spl_offset=spl_offset,
        chunk_duration=args.chunk_duration,
        sample_rate=sample_rate,
        max_freq_bins=args.max_freq_bins,
        min_peak_sep_deg=args.min_peak_sep_deg,
        tolerance_deg=args.tolerance_deg,
        output_dir=output_dir,
        no_figures=args.no_figures,
    )

    print_multisource_table(results, ground_truths, spl_offset)

    json_path = output_dir / f"multisource_{wav_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sonuçlar: {json_path}")

    return results


if __name__ == "__main__":
    main()
