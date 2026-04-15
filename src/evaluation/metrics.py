"""
Akustik Kamera Değerlendirme Metrikleri

Lokalizasyon doğruluğu, SNR kazancı ve pipeline gecikmesi ölçümleri.
AES paper'ı için kullanılacak sayısal sonuçlar üretir.
"""

import time
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def angular_localization_error(
    estimated_xyz: np.ndarray,
    true_xyz: np.ndarray,
    array_center: np.ndarray = None
) -> float:
    """
    Tahmini ve gerçek kaynak konumu arasındaki açısal hata (derece).

    Mikrofon dizisinin merkezi referans alınarak her iki vektör arasındaki
    açı hesaplanır. Uzaklık bilgisini dışarıda bırakır, sadece yön hatası.

    Args:
        estimated_xyz: Tahmini kaynak konumu (3,) array [x, y, z]
        true_xyz:      Gerçek kaynak konumu (3,) array [x, y, z]
        array_center:  Dizi merkezi (3,); None ise [0, 0, 0] kabul edilir.

    Returns:
        Açısal hata (derece, 0–180 arasında)
    """
    if array_center is None:
        array_center = np.zeros(3)

    v_est  = np.asarray(estimated_xyz, dtype=float) - array_center
    v_true = np.asarray(true_xyz,      dtype=float) - array_center

    norm_est  = np.linalg.norm(v_est)
    norm_true = np.linalg.norm(v_true)

    if norm_est < 1e-9 or norm_true < 1e-9:
        logger.warning("Sıfır uzunluklu vektör — hata 0 döndürülüyor.")
        return 0.0

    cos_theta = np.clip(np.dot(v_est, v_true) / (norm_est * norm_true), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def peak_to_xyz(
    power_map: np.ndarray,
    grid_points: np.ndarray
) -> np.ndarray:
    """
    Güç haritasındaki en yüksek noktanın koordinatını döndürür.

    Args:
        power_map:   (n_points,) lineer güç değerleri
        grid_points: (n_points, 3) grid koordinatları [x, y, z]

    Returns:
        (3,) array — peak koordinatı [x, y, z]
    """
    peak_idx = int(np.argmax(power_map))
    return grid_points[peak_idx].copy()


def snr_improvement(
    beamformed_signal: np.ndarray,
    reference_signal: np.ndarray,
    noise_percentile: float = 20.0
) -> float:
    """
    Beamforming SNR kazancı (dB).

    Güç haritasının tepe değerini gürültü tabanına (alt yüzdelik) böler.
    Referans sinyal (tek kanal) ile karşılaştırır.

    Args:
        beamformed_signal: (n_points,) beamformer güç haritası (lineer)
        reference_signal:  (n_samples,) tek kanallı zaman-domain ses
        noise_percentile:  Gürültü tabanı için yüzdelik dilim (varsayılan %20)

    Returns:
        SNR kazancı dB cinsinden (beamformed – referans SNR)
    """
    # Beamformer SNR: peak / gürültü tabanı
    noise_floor_bf = float(np.percentile(beamformed_signal, noise_percentile))
    peak_bf = float(np.max(beamformed_signal))
    if noise_floor_bf < 1e-30:
        noise_floor_bf = 1e-30
    snr_bf = 10 * np.log10(peak_bf / noise_floor_bf)

    # Referans SNR: en güçlü kanal RMS / gürültü tabanı
    ref_power = float(np.mean(reference_signal ** 2))
    noise_ref  = float(np.percentile(np.abs(reference_signal), noise_percentile) ** 2)
    if noise_ref < 1e-30:
        noise_ref = 1e-30
    snr_ref = 10 * np.log10(ref_power / noise_ref)

    return snr_bf - snr_ref


def measure_pipeline_latency(
    audio_chunk: np.ndarray,
    beamform_fn: Callable,
    n_trials: int = 10,
    **beamform_kwargs
) -> Dict[str, float]:
    """
    Bir chunk'ın uçtan uca beamforming işlem süresi.

    Args:
        audio_chunk:     (n_samples, n_mics) ses verisi
        beamform_fn:     Çağrılacak beamforming fonksiyonu
        n_trials:        Ölçüm tekrar sayısı (ortalama için)
        **beamform_kwargs: beamform_fn'e iletilecek ek argümanlar

    Returns:
        dict:
          - mean_ms:   Ortalama gecikme (ms)
          - std_ms:    Standart sapma (ms)
          - min_ms:    Minimum gecikme (ms)
          - max_ms:    Maksimum gecikme (ms)
    """
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        beamform_fn(audio_chunk, **beamform_kwargs)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
    }


def detection_precision_recall(
    detections: List[np.ndarray],
    ground_truth: List[np.ndarray],
    tolerance_deg: float = 5.0,
    array_center: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Kaynak tespiti doğruluk metrikleri: precision, recall, F1.

    Her tespit için en yakın ground-truth noktasına açısal mesafe hesaplanır.
    Mesafe tolerance_deg altındaysa True Positive sayılır.

    Args:
        detections:    Tahmin edilen kaynak konumları [(3,), ...]
        ground_truth:  Gerçek kaynak konumları [(3,), ...]
        tolerance_deg: Eşik açı (derece); bu açı altındaki tespitler TP
        array_center:  Dizi merkezi (None → [0,0,0])

    Returns:
        dict: precision, recall, f1, tp, fp, fn
    """
    if array_center is None:
        array_center = np.zeros(3)

    tp = 0
    matched_gt = set()

    for det in detections:
        best_angle = np.inf
        best_idx = -1
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            angle = angular_localization_error(det, gt, array_center)
            if angle < best_angle:
                best_angle = angle
                best_idx = i
        if best_angle <= tolerance_deg:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(detections) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
    }


def batch_evaluate(
    wav_path: str,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    config,
    ground_truth_xyz: Optional[np.ndarray] = None,
    algorithms: Optional[List[str]] = None,
    spl_offset: float = 0.0,
    chunk_duration: float = 0.5,
    sample_rate: int = 48000,
    max_freq_bins: int = 10,
) -> Dict[str, dict]:
    """
    Kayıtlı WAV dosyası üzerinde tüm algoritmaları değerlendirir.

    Args:
        wav_path:         WAV dosyası yolu
        mic_positions:    (n_mics, 3) mikrofon konumları
        grid_points:      (n_points, 3) grid noktaları
        config:           BeamformingConfig nesnesi
        ground_truth_xyz: Gerçek kaynak konumu (None ise localization metrikleri atlanır)
        algorithms:       ['DAS', 'MVDR', 'MUSIC']; None ise ['DAS']
        spl_offset:       dBFS → dBSPL offset (config'den alınmalı)
        chunk_duration:   Analiz penceresi (saniye)
        sample_rate:      Örnekleme hızı
        max_freq_bins:    Hız için frekans bin sınırı

    Returns:
        Algoritma adı → metrik dict eşlemesi:
          - mean_angular_error_deg  (sadece ground_truth verilmişse)
          - latency (mean_ms, std_ms, min_ms, max_ms)
          - peak_db
    """
    import soundfile as sf
    from src.algorithms.beamforming import (
        das_beamformer_realtime,
        mvdr_beamformer_realtime,
        music_beamformer_realtime,
        power_to_db,
        _precompute_distances,
    )

    if algorithms is None:
        algorithms = ["DAS"]

    # WAV yükle
    audio, sr = sf.read(wav_path, always_2d=True)
    if sr != sample_rate:
        logger.warning(f"WAV sample rate {sr} != expected {sample_rate}")
    n_channels = audio.shape[1]
    if n_channels < mic_positions.shape[0]:
        raise ValueError(f"WAV has {n_channels} channels, expected {mic_positions.shape[0]}")
    # Sadece mic_positions kadar kanal al
    audio = audio[:, :mic_positions.shape[0]]

    chunk_samples = int(chunk_duration * sample_rate)

    # Ön hesaplama
    distances = _precompute_distances(mic_positions, grid_points)

    fn_map = {
        "DAS":   lambda chunk: das_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances),
        "MVDR":  lambda chunk: mvdr_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances),
        "MUSIC": lambda chunk: music_beamformer_realtime(
            chunk, mic_positions, grid_points, sample_rate, config,
            max_freq_bins=max_freq_bins, distances=distances),
    }

    results = {}
    for alg in algorithms:
        if alg not in fn_map:
            logger.warning(f"Bilinmeyen algoritma: {alg}, atlanıyor.")
            continue

        beamform = fn_map[alg]
        angular_errors = []
        all_latencies  = []
        peak_dbs       = []

        n_chunks = max(1, len(audio) // chunk_samples)
        for i in range(n_chunks):
            chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]
            if len(chunk) < chunk_samples:
                break

            # Gecikme ölçümü (3 deneme ile, hız için)
            lat = measure_pipeline_latency(chunk, beamform, n_trials=3)
            all_latencies.append(lat["mean_ms"])

            # Güç haritası
            power_map = beamform(chunk)
            peak_db = float(power_to_db(np.max(power_map), spl_offset=spl_offset))
            peak_dbs.append(peak_db)

            # Lokalizasyon hatası
            if ground_truth_xyz is not None:
                est_xyz = peak_to_xyz(power_map, grid_points)
                err = angular_localization_error(est_xyz, ground_truth_xyz)
                angular_errors.append(err)

        metrics = {
            "latency": {
                "mean_ms": float(np.mean(all_latencies)),
                "std_ms":  float(np.std(all_latencies)),
                "min_ms":  float(np.min(all_latencies)),
                "max_ms":  float(np.max(all_latencies)),
            },
            "peak_db": {
                "mean": float(np.mean(peak_dbs)),
                "std":  float(np.std(peak_dbs)),
            },
        }
        if angular_errors:
            metrics["mean_angular_error_deg"] = float(np.mean(angular_errors))
            metrics["std_angular_error_deg"]  = float(np.std(angular_errors))

        results[alg] = metrics
        logger.info(f"{alg}: latency={metrics['latency']['mean_ms']:.1f}ms, "
                    f"peak={metrics['peak_db']['mean']:.1f}dB"
                    + (f", err={metrics['mean_angular_error_deg']:.1f}°"
                       if angular_errors else ""))

    return results
