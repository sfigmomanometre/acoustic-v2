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
    measure_birdnet_latency,
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
        result = hybrid_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances,
            roi_threshold_db=roi_threshold_db,
        )
        # hybrid_beamformer_realtime returns (full_power, n_sources_used, roi_mask)
        return result[0] if isinstance(result, tuple) else result
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


def print_birdnet_table(birdnet_result: dict):
    """BirdNET gecikme sonuçlarını ASCII tablo olarak yazdır."""
    m = birdnet_result
    print("\n" + "=" * 70)
    print("  BirdNET İŞLEM HIZI")
    print("=" * 70)
    print(f"  {'Gecikme/chunk':<20} {m['mean_ms']:>7.1f} ± {m['std_ms']:.1f} ms")
    print(f"  {'Min / Maks':<20} {m['min_ms']:>7.1f} / {m['max_ms']:.1f} ms")
    print(f"  {'RTF':<20} {m['rtf']:>7.4f}  (1.0 = gerçek-zamanlı)")
    print(f"  {'Throughput':<20} {m['throughput_x']:>7.1f}× gerçek zamanlıdan hızlı")
    print(f"  {'Analiz edilen':<20} {m['chunks_analyzed']:>7} chunk "
          f"({m['audio_duration_s']:.1f}s ses)")
    print(f"  {'Toplam işlem':<20} {m['total_processing_s']:>7.2f} s")
    print("=" * 70)


def _render_power_map_ax(
    ax,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    power_db: np.ndarray,
    title: str,
    spl_offset: float,
    dynamic_range: float = 15.0,
    cmap: str = "hot_r",
    contour_levels_db: tuple = (-12, -9, -6, -3),
    fontsize: int = 10,
    normalize_to_peak: bool = False,
):
    """
    Tek eksene konturlu güç haritası çiz.

    normalize_to_peak=True ise colorbar'ı her panel kendi peak'ine göre
    gösterir (0 dB = peak, −N dB = background). Bu seçenek MVDR/MUSIC gibi
    dar demet algoritmalarının zıtlığını ortaya çıkarır.

    Siyah-beyaz baskıya uyum için:
    - 'hot_r' colormap (karanlıktan açığa — B&W'da parlaklık degradesi)
    - Siyah kontur çizgileri (belirli dB seviyeleri)
    - Kontur etiketleri
    """
    import matplotlib.pyplot as plt

    v_max = float(power_db.max())
    v_min = v_max - dynamic_range

    # normalize_to_peak=True → haritayı [−dynamic_range, 0] aralığına kaydır
    if normalize_to_peak:
        plot_data = power_db - v_max          # peak = 0 dB, arka plan < 0
        pv_max = 0.0
        pv_min = -dynamic_range
        abs_levels = sorted([d for d in contour_levels_db if d > pv_min])
        cb_label = "dB (peak-normalized)"
    else:
        plot_data = power_db
        pv_max = v_max
        pv_min = v_min
        abs_levels = sorted([v_max + d for d in contour_levels_db if v_max + d > v_min])
        cb_label = "dBSPL" if spl_offset > 0 else "dBFS"

    # Arka plan: doldurulmuş renk haritası
    im = ax.pcolormesh(
        x_vals, y_vals, plot_data,
        cmap=cmap,
        vmin=pv_min,
        vmax=pv_max,
        shading="auto",
    )

    # Kontur seviyeleri (her iki mod için aynı göreli dB mantığı)
    if abs_levels:
        cs = ax.contour(
            x_vals, y_vals, plot_data,
            levels=abs_levels,
            colors="black",
            linewidths=0.8,
            linestyles=["--", "-.", "-", "-"],
        )
        if normalize_to_peak:
            fmt = {lv: f"{lv:+.0f} dB" for lv in abs_levels}
        else:
            fmt = {lv: f"{lv - v_max:+.0f} dB" for lv in abs_levels}
        ax.clabel(cs, fmt=fmt, fontsize=fontsize - 2, inline=True, inline_spacing=2)

    # Tepe noktasını işaretle
    peak_idx = np.unravel_index(np.argmax(plot_data), plot_data.shape)
    ax.plot(
        x_vals[peak_idx], y_vals[peak_idx],
        marker="+", color="black", markersize=10, markeredgewidth=1.5,
    )

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cb_label, fontsize=fontsize - 1)
    cb.ax.tick_params(labelsize=fontsize - 2)

    ax.set_title(title, fontsize=fontsize + 1, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=fontsize)
    ax.set_ylabel("Y (m)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4, color="gray")


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
    """
    Her algoritma için ilk chunk'ın güç haritasını kaydet.

    Üretilen dosyalar:
      power_maps_<stem>.png          — 2×2 (veya 1×N) karşılaştırma figürü
      power_map_<stem>_<ALG>.png     — Her algoritma için ayrı yüksek çözünürlüklü figür

    Her figür B&W baskıya uyumlu:
      - hot_r colormap (parlaklık degradesi B&W'da korunur)
      - Siyah kontur çizgileri (−3/−6/−9/−12 dB)
      - Kontur etiketleri
      - Peak konumu işaretli (+)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import soundfile as sf
    except ImportError:
        logger.warning("matplotlib veya soundfile yok — figürler kaydedilmedi.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

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

    x_vals = grid_points[:, 0].reshape(grid_shape)
    y_vals = grid_points[:, 1].reshape(grid_shape)

    # Tüm algoritmaların güç haritalarını hesapla (bir kez)
    power_maps = {}
    for alg in algorithms:
        if alg in fn_map:
            pm = fn_map[alg](chunk)
            power_maps[alg] = power_to_db(pm, spl_offset=spl_offset).reshape(grid_shape)

    valid_algs = [a for a in algorithms if a in power_maps]
    if not valid_algs:
        logger.warning("Hiçbir algoritma hesaplanamadı.")
        return

    # --- 1. Karşılaştırma figürü (2×2 veya 1×N) ---
    n = len(valid_algs)
    if n <= 2:
        nrows, ncols = 1, n
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    # DAS: geniş demet → 15 dB dinamik aralık, mutlak dBSPL göster
    # MVDR/MUSIC/Hybrid: dar demet → 6 dB dinamik aralık, peak'e normalize et
    _NARROW_BEAM_ALGS = {"MVDR", "MUSIC", "Hybrid"}

    for i, alg in enumerate(valid_algs):
        is_narrow = alg in _NARROW_BEAM_ALGS
        _render_power_map_ax(
            axes_flat[i], x_vals, y_vals, power_maps[alg],
            title=alg, spl_offset=spl_offset, fontsize=10,
            dynamic_range=6.0 if is_narrow else 15.0,
            normalize_to_peak=is_narrow,
        )
    # Boş panelleri gizle (örn. 3 algoritma → 4. panel boş)
    for j in range(len(valid_algs), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Beamforming Güç Haritaları — {wav_path.stem}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()

    comparison_path = output_dir / f"power_maps_{wav_path.stem}.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Karşılaştırma figürü kaydedildi: {comparison_path}")

    # --- 2. Her algoritma için ayrı yüksek çözünürlüklü figür ---
    for alg in valid_algs:
        is_narrow = alg in _NARROW_BEAM_ALGS
        fig_s, ax_s = plt.subplots(1, 1, figsize=(4.5, 4.0))
        _render_power_map_ax(
            ax_s, x_vals, y_vals, power_maps[alg],
            title=alg, spl_offset=spl_offset, fontsize=11,
            dynamic_range=6.0 if is_narrow else 15.0,
            normalize_to_peak=is_narrow,
        )
        plt.tight_layout()
        alg_slug = alg.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("→", "")
        single_path = output_dir / f"power_map_{wav_path.stem}_{alg_slug}.png"
        fig_s.savefig(single_path, dpi=300, bbox_inches="tight")
        plt.close(fig_s)
        logger.info(f"  {alg} figürü kaydedildi: {single_path}")


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
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Sadece figür üret: metrik hesaplama ve JSON kayıt atlanır",
    )
    parser.add_argument(
        "--birdnet", action="store_true",
        help="BirdNET işlem hızını da ölç ve rapora ekle",
    )
    parser.add_argument(
        "--birdnet-channel", type=int, default=0,
        help="BirdNET analizi için kullanılacak WAV kanalı (varsayılan: 0)",
    )
    parser.add_argument(
        "--birdnet-max-chunks", type=int, default=None,
        help="BirdNET için maksimum chunk sayısı (varsayılan: tüm kayıt)",
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

    # --- Sadece figür modu ---
    if args.figures_only:
        if args.no_figures:
            logger.warning("--figures-only ve --no-figures aynı anda kullanıldı; figürler üretilmeyecek.")
            return {}
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
        return {}

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

    # --- BirdNET hız ölçümü ---
    if args.birdnet:
        logger.info("BirdNET işlem hızı ölçülüyor...")
        try:
            birdnet_result = measure_birdnet_latency(
                wav_path=str(wav_path),
                sample_rate=sample_rate,
                chunk_duration=3.0,
                channel=args.birdnet_channel,
                n_warmup=1,
                max_chunks=args.birdnet_max_chunks,
            )
            print_birdnet_table(birdnet_result)
            results["BirdNET"] = birdnet_result
        except Exception as exc:
            logger.warning(f"BirdNET ölçümü başarısız: {exc}")

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
