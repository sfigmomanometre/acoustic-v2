#!/usr/bin/env python3
"""
Akustik Pipeline Değerlendirme Scripti

Kayıtlı WAV dosyası üzerinde seçilen algoritmaları çalıştırır,
lokalizasyon hatası ve gecikme metriklerini hesaplar,
karşılaştırma tablosu ve güç haritası figürlerini kaydeder.

Kullanım örnekleri:
    # Tek WAV, bilinen kaynak konumuyla:
    python scripts/evaluate_pipeline.py \\
        --recording data/recordings/lab_test_1m_0deg/audio.wav \\
        --ground-truth 1.0 0.0 1.0 \\
        --algorithms DAS MVDR MUSIC

    # Sadece hız ve güç ölçümü (ground-truth yok):
    python scripts/evaluate_pipeline.py \\
        --recording data/recordings/outdoor_session/audio.wav \\
        --algorithms DAS Hybrid

    # Tüm parametreler:
    python scripts/evaluate_pipeline.py --help
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Proje kök dizinini path'e ekle
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
    normalize_power_map,
    _precompute_distances,
)
from src.evaluation.metrics import (
    angular_localization_error,
    peak_to_xyz,
    snr_improvement,
    measure_pipeline_latency,
    batch_evaluate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
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
    grid_size_x = x_range[1] - x_range[0]
    grid_size_y = y_range[1] - y_range[0]

    return BeamformingConfig(
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        grid_resolution=grid.get("resolution", 0.05),
        focus_distance=grid.get("z", 1.0),
        freq_min=freq.get("min", 500),
        freq_max=freq.get("max", 8000),
        freq_bins=freq.get("bands", 3) * 3,
        sound_speed=sound_speed,
    )


def run_hybrid(chunk, mic_positions, grid_points, sample_rate, config,
               distances, max_freq_bins, roi_threshold_db=6.0):
    """
    İki aşamalı DAS→MUSIC Hybrid beamformer.
    GUI'deki hybrid_beamformer_realtime olmadan, direkt çağrı.
    """
    try:
        from src.algorithms.beamforming import hybrid_beamformer_realtime
        return hybrid_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances,
            roi_threshold_db=roi_threshold_db,
        )
    except (ImportError, TypeError):
        # Hybrid yoksa DAS ile devam et
        return das_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances,
        )


def print_results_table(results: dict, spl_offset: float):
    """Sonuçları ASCII tablo olarak yazdır."""
    mode = "dBSPL" if spl_offset > 0 else "dBFS"
    print("\n" + "=" * 70)
    print(f"  DEĞERLENDIRME SONUÇLARI  [{mode}, offset={spl_offset:.1f} dB]")
    print("=" * 70)
    header = f"{'Algoritma':<10} {'Gecikme (ms)':>14} {'Peak (dB)':>12} {'Açısal Hata (°)':>18}"
    print(header)
    print("-" * 70)
    for alg, m in results.items():
        lat = m["latency"]["mean_ms"]
        lat_std = m["latency"]["std_ms"]
        peak = m["peak_db"]["mean"]
        err_str = (f"{m['mean_angular_error_deg']:>8.1f} ± {m['std_angular_error_deg']:.1f}"
                   if "mean_angular_error_deg" in m else "       —")
        print(f"{alg:<10} {lat:>8.1f} ±{lat_std:>4.1f}  {peak:>10.1f}   {err_str}")
    print("=" * 70)


def save_power_maps(
    wav_path: Path,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    grid_shape: tuple,
    config: BeamformingConfig,
    algorithms: list,
    sample_rate: int,
    spl_offset: float,
    max_freq_bins: int,
    output_dir: Path,
    chunk_duration: float = 0.5,
):
    """Her algoritma için ilk chunk'ın güç haritasını PNG olarak kaydet."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import soundfile as sf
    except ImportError:
        logger.warning("matplotlib veya soundfile yok — figürler kaydedilmedi.")
        return

    audio, sr = sf.read(str(wav_path), always_2d=True)
    audio = audio[:, :mic_positions.shape[0]]
    chunk_samples = int(chunk_duration * sr)
    chunk = audio[:chunk_samples]
    if len(chunk) < chunk_samples:
        logger.warning("Chunk boyutu yetersiz, figür üretilemiyor.")
        return

    distances = _precompute_distances(mic_positions, grid_points)

    fn_map = {
        "DAS":    lambda c: das_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances),
        "MVDR":   lambda c: mvdr_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances),
        "MUSIC":  lambda c: music_beamformer_realtime(
            c, mic_positions, grid_points, sr, config,
            max_freq_bins=max_freq_bins, distances=distances),
        "Hybrid": lambda c: run_hybrid(
            c, mic_positions, grid_points, sr, config, distances, max_freq_bins),
    }

    n_algs = len(algorithms)
    fig, axes = plt.subplots(1, n_algs, figsize=(5 * n_algs, 5))
    if n_algs == 1:
        axes = [axes]

    x_vals = grid_points[:, 0].reshape(grid_shape)
    y_vals = grid_points[:, 1].reshape(grid_shape)

    for ax, alg in zip(axes, algorithms):
        if alg not in fn_map:
            continue
        power_map = fn_map[alg](chunk)
        power_db = power_to_db(power_map, spl_offset=spl_offset).reshape(grid_shape)

        im = ax.pcolormesh(
            x_vals, y_vals, power_db,
            cmap="jet",
            vmin=power_db.max() - 15,
            vmax=power_db.max(),
        )
        plt.colorbar(im, ax=ax, label="dBSPL" if spl_offset > 0 else "dBFS")
        ax.set_title(alg)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")

    plt.suptitle(f"Güç Haritaları — {wav_path.name}", y=1.02)
    plt.tight_layout()

    fig_path = output_dir / f"power_maps_{wav_path.stem}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Figür kaydedildi: {fig_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Akustik pipeline değerlendirme aracı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--recording", required=True,
        help="Değerlendirilecek WAV dosyası yolu",
    )
    parser.add_argument(
        "--ground-truth", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=None,
        help="Gerçek kaynak konumu (metre). Örn: 1.0 0.0 1.0",
    )
    parser.add_argument(
        "--algorithms", nargs="+",
        default=["DAS"],
        choices=["DAS", "MVDR", "MUSIC", "Hybrid"],
        help="Değerlendirilecek algoritmalar",
    )
    parser.add_argument(
        "--tolerance", type=float, default=5.0,
        help="Lokalizasyon tolerans açısı (derece)",
    )
    parser.add_argument(
        "--chunk-duration", type=float, default=0.5,
        help="Analiz penceresi (saniye)",
    )
    parser.add_argument(
        "--max-freq-bins", type=int, default=10,
        help="Frekans bin sınırı (hız/doğruluk dengesi)",
    )
    parser.add_argument(
        "--output", default="results",
        help="Çıktı dizini (figürler ve JSON)",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="config.yaml yolu",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Figür üretimini atla",
    )
    args = parser.parse_args()

    # --- Yolları çöz ---
    wav_path = Path(args.recording).resolve()
    if not wav_path.exists():
        logger.error(f"WAV dosyası bulunamadı: {wav_path}")
        sys.exit(1)

    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        logger.error(f"Config bulunamadı: {config_path}")
        sys.exit(1)

    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Config yükle ---
    cfg = load_config(config_path)
    spl_offset = float(cfg.get("calibration", {}).get("spl_offset_db", 0.0))
    calibrated = cfg.get("calibration", {}).get("calibrated", False)
    sample_rate = int(cfg.get("audio", {}).get("sample_rate", 48000))

    logger.info(f"SPL offset: {spl_offset} dB (calibrated={calibrated})")
    logger.info(f"Sample rate: {sample_rate} Hz")

    # --- Geometri ---
    geom_file = project_root / cfg.get("microphone", {}).get(
        "geometry_file", "config/micgeom.xml")
    mic_positions = load_mic_geometry(str(geom_file))

    bf_config = build_beamforming_config(cfg)
    grid_points, grid_shape = create_focus_grid(bf_config)
    logger.info(f"Grid: {grid_shape[0]}×{grid_shape[1]} = {len(grid_points)} nokta")

    # --- Ground truth ---
    ground_truth = None
    if args.ground_truth:
        ground_truth = np.array(args.ground_truth, dtype=float)
        logger.info(f"Ground truth: {ground_truth}")

    # --- Batch değerlendirme (Hybrid özel işlem gerektirir) ---
    algs_for_batch = [a for a in args.algorithms if a != "Hybrid"]
    results = {}

    if algs_for_batch:
        results.update(batch_evaluate(
            wav_path=str(wav_path),
            mic_positions=mic_positions,
            grid_points=grid_points,
            config=bf_config,
            ground_truth_xyz=ground_truth,
            algorithms=algs_for_batch,
            spl_offset=spl_offset,
            chunk_duration=args.chunk_duration,
            sample_rate=sample_rate,
            max_freq_bins=args.max_freq_bins,
        ))

    # Hybrid ayrı (batch_evaluate desteklemiyor olabilir)
    if "Hybrid" in args.algorithms:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), always_2d=True)
        audio = audio[:, :mic_positions.shape[0]]
        chunk_samples = int(args.chunk_duration * sr)
        distances = _precompute_distances(mic_positions, grid_points)

        hybrid_latencies, hybrid_peaks, hybrid_errors = [], [], []
        n_chunks = max(1, len(audio) // chunk_samples)
        for i in range(n_chunks):
            chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]
            if len(chunk) < chunk_samples:
                break
            lat = measure_pipeline_latency(
                chunk,
                lambda c: run_hybrid(c, mic_positions, grid_points, sr, bf_config,
                                     distances, args.max_freq_bins),
                n_trials=3,
            )
            hybrid_latencies.append(lat["mean_ms"])
            pm = run_hybrid(chunk, mic_positions, grid_points, sr, bf_config,
                            distances, args.max_freq_bins)
            peak_db = float(power_to_db(float(np.max(pm)), spl_offset=spl_offset))
            hybrid_peaks.append(peak_db)
            if ground_truth is not None:
                est = peak_to_xyz(pm, grid_points)
                hybrid_errors.append(angular_localization_error(est, ground_truth))

        hybrid_result = {
            "latency": {
                "mean_ms": float(np.mean(hybrid_latencies)),
                "std_ms":  float(np.std(hybrid_latencies)),
                "min_ms":  float(np.min(hybrid_latencies)),
                "max_ms":  float(np.max(hybrid_latencies)),
            },
            "peak_db": {
                "mean": float(np.mean(hybrid_peaks)),
                "std":  float(np.std(hybrid_peaks)),
            },
        }
        if hybrid_errors:
            hybrid_result["mean_angular_error_deg"] = float(np.mean(hybrid_errors))
            hybrid_result["std_angular_error_deg"]  = float(np.std(hybrid_errors))
        results["Hybrid"] = hybrid_result

    # --- Tablo ---
    print_results_table(results, spl_offset)

    # --- JSON kaydet ---
    json_path = output_dir / f"results_{wav_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Sonuçlar kaydedildi: {json_path}")

    # --- Figürler ---
    if not args.no_figures:
        save_power_maps(
            wav_path=wav_path,
            mic_positions=mic_positions,
            grid_points=grid_points,
            grid_shape=grid_shape,
            config=bf_config,
            algorithms=args.algorithms,
            sample_rate=sample_rate,
            spl_offset=spl_offset,
            max_freq_bins=args.max_freq_bins,
            output_dir=output_dir,
            chunk_duration=args.chunk_duration,
        )

    return results


if __name__ == "__main__":
    main()
