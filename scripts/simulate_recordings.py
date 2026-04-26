#!/usr/bin/env python3
"""
Gerçekçi Simüle Mikrofon Dizi Kayıtları Üretici
================================================

Fiziksel model: düzlem dalga propagasyonu + 1/r genlik düşüşü + eklenmiş gürültü.
Gerçek mikrofon geometrisi (config/micgeom.xml) kullanılır.

Üretilen senaryolar
-------------------
P1 — Lokalizasyon doğruluğu (mesafe × azimut):
    P1-A: 1.0 m,  0°    GT: (0.000, 1.000, 1.000)
    P1-B: 1.0 m, 45°    GT: (0.707, 0.707, 1.000)
    P1-C: 1.0 m, 90°    GT: (1.000, 0.000, 1.000)
    P1-D: 2.0 m,  0°    GT: (0.000, 2.000, 1.000)
    P1-E: 2.0 m, 45°    GT: (1.414, 1.414, 1.000)
    P1-F: 3.0 m,  0°    GT: (0.000, 3.000, 1.000)

Cocktail — Birden fazla eş zamanlı kaynak:
    N2: 2 kaynak ±45° — GT: (±0.707, 0.707, 1.000)
    N3: 3 kaynak 0°/60°/-60° — GT: (0, 1, 1), (0.866, 0.5, 1), (-0.866, 0.5, 1)

Kullanım
--------
    python scripts/simulate_recordings.py
    python scripts/simulate_recordings.py --snr 20 --duration 15
    python scripts/simulate_recordings.py --scenarios P1 cocktail
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import xml.etree.ElementTree as ET

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fizik sabitleri
# ---------------------------------------------------------------------------
SPEED_OF_SOUND = 343.0   # m/s, 20°C
SAMPLE_RATE    = 48000   # Hz


# ---------------------------------------------------------------------------
# Mikrofon geometrisi
# ---------------------------------------------------------------------------

def load_mic_positions(xml_path: Path) -> np.ndarray:
    """micgeom.xml'den mikrofon konumlarını yükle → (N, 3) array."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    positions = []
    for pos in root.findall(".//pos"):
        x = float(pos.get("x", 0))
        y = float(pos.get("y", 0))
        z = float(pos.get("z", 0))
        positions.append([x, y, z])
    return np.array(positions, dtype=float)


# ---------------------------------------------------------------------------
# Kaynak sinyali
# ---------------------------------------------------------------------------

def pink_noise(n_samples: int, rng: np.random.Generator,
               f_low: float = 2000, f_high: float = 8000,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Bant sınırlı pembe gürültü — kuş sesi spektral profilini simüle eder.
    Güç ~1/f, bant dışı sıfırlanır.
    """
    # Beyaz gürültü → FFT
    white = rng.standard_normal(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

    # 1/f pembeleştirme
    f_safe = np.where(freqs > 0, freqs, 1.0)
    pink_filter = 1.0 / np.sqrt(f_safe)
    pink_filter[0] = 0.0  # DC'yi kaldır

    # Bant sınırlandırma
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    pink_filter[~band_mask] = 0.0

    signal = np.fft.irfft(X * pink_filter, n=n_samples).astype(np.float32)

    # Birim RMS'e normalleştir
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-10:
        signal /= rms
    return signal


# ---------------------------------------------------------------------------
# Çok kanallı sinyal simülasyonu
# ---------------------------------------------------------------------------

def simulate_array_recording(
    mic_positions: np.ndarray,
    sources: list[dict],          # [{'pos': [x,y,z], 'signal': array, 'amplitude': float}]
    snr_db: float,
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator = None,
    z_source: float = 1.0,
) -> np.ndarray:
    """
    Mikrofon dizi kaydını simüle et.

    Args:
        mic_positions: (M, 3) mikrofon konumları [m]
        sources:       Kaynak listesi, her biri 'pos', 'signal', 'amplitude' içerir
        snr_db:        Sinyal-gürültü oranı (dB) — en güçlü kaynağa göre
        duration_s:    Kayıt süresi (saniye)
        sample_rate:   Örnekleme hızı (Hz)
        rng:           Numpy rastgele sayı üreteci

    Returns:
        (n_samples, M) float32 çok kanallı ses dizisi
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(duration_s * sample_rate)
    n_mics = len(mic_positions)

    # Çıktı tamponu
    output = np.zeros((n_samples, n_mics), dtype=np.float64)

    for src in sources:
        src_pos = np.asarray(src["pos"], dtype=float)
        # Kaynak sinyalini gerekli uzunluğa kırp/döngüle
        raw = src["signal"]
        if len(raw) >= n_samples:
            signal = raw[:n_samples]
        else:
            repeats = int(np.ceil(n_samples / len(raw)))
            signal = np.tile(raw, repeats)[:n_samples]

        amp = float(src.get("amplitude", 1.0))

        for m, mic_pos in enumerate(mic_positions):
            dist = float(np.linalg.norm(src_pos - mic_pos))
            delay_samples = dist / SPEED_OF_SOUND * sample_rate

            # Tam örnek gecikme + lineer interpolasyon
            delay_int  = int(delay_samples)
            delay_frac = delay_samples - delay_int

            # Sinyali kaydır (tam örnek)
            delayed = np.zeros(n_samples, dtype=np.float64)
            if delay_int < n_samples:
                delayed[delay_int:] = signal[: n_samples - delay_int]

            # Kesirli gecikme (lineer interpolasyon)
            if delay_frac > 0 and delay_int + 1 < n_samples:
                delayed_p1 = np.zeros(n_samples, dtype=np.float64)
                d1 = delay_int + 1
                delayed_p1[d1:] = signal[: n_samples - d1]
                delayed = (1 - delay_frac) * delayed + delay_frac * delayed_p1

            # 1/r genlik düşüşü
            output[:, m] += amp * delayed / max(dist, 0.01)

    # Gürültü ekle: SNR = 10·log10(P_signal / P_noise) → P_noise = P_signal / 10^(SNR/10)
    # En güçlü kaynağın ilk mikrofondaki gücünü referans al
    signal_power = float(np.mean(output[:, 0] ** 2)) if np.any(output != 0) else 1.0
    if signal_power < 1e-30:
        signal_power = 1e-30
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal((n_samples, n_mics)) * np.sqrt(noise_power)

    result = (output + noise).astype(np.float32)

    # Kırpma önleme
    max_val = np.max(np.abs(result))
    if max_val > 0.95:
        result *= 0.95 / max_val

    return result


# ---------------------------------------------------------------------------
# Senaryo tanımları
# ---------------------------------------------------------------------------

def azimuth_to_xyz(depth_z: float, azimuth_deg: float) -> list:
    """
    Azimut açısını XYZ konumuna çevir — kaynak FOCAL PLANE'e (z=depth_z) yerleştirilir.

    Koordinat sistemi (micgeom.xml ile aynı):
        +Z = array önü — derinlik, 0° yönü
        +X = sağ
        +Y = yukarı (yükseklik; y=0 → kuş array ile aynı yükseklikte)

    Focal plane yerleşimi (z = depth_z sabit):
        x = depth_z × tan(azimuth)
        y = 0

    Neden focal plane?
        Beamforming grid z=1.0 sabit olduğundan, kaynağı grid'den uzak (z≠1.0)
        yerleştirmek near-field hatası yaratır. Kaynağı focal plane'de tutmak
        yönsel tutarlılığı garanti eder; "mesafe" z parametresi ile kontrol edilir.

    Grid kısıtı: x∈[-1.5, 1.5] → azimuth ≤ arctan(1.5/1.0) ≈ 56°.
    """
    az_rad = np.radians(azimuth_deg)
    x = depth_z * np.tan(az_rad)
    y = 0.0
    z = float(depth_z)
    return [x, y, z]


# P1 pozisyonları
#   depth_z: kaynağın array'den derinliği (m)  — "mesafe" etkisini SNR ile simüle eder
#   azimuth: açısal konum (focal plane'de temsil edilir)
P1_SCENARIOS = [
    {"id": "P1-A", "depth_z": 1.0, "azimuth":  0,  "label": "1m_0deg"},
    {"id": "P1-B", "depth_z": 1.0, "azimuth": 30,  "label": "1m_30deg"},
    {"id": "P1-C", "depth_z": 1.0, "azimuth": 45,  "label": "1m_45deg"},
    {"id": "P1-D", "depth_z": 2.0, "azimuth":  0,  "label": "2m_0deg"},
    {"id": "P1-E", "depth_z": 2.0, "azimuth": 30,  "label": "2m_30deg"},
    {"id": "P1-F", "depth_z": 3.0, "azimuth":  0,  "label": "3m_0deg"},
]

COCKTAIL_SCENARIOS = [
    {
        "id": "cocktail_N2",
        "label": "cocktail_N2",
        "sources": [
            {"depth_z": 1.0, "azimuth":  45},   # sağ 45°
            {"depth_z": 1.0, "azimuth": -45},   # sol 45°
        ],
    },
    {
        "id": "cocktail_N3",
        "label": "cocktail_N3",
        "sources": [
            {"depth_z": 1.0, "azimuth":   0},   # düz önde
            {"depth_z": 1.0, "azimuth":  45},   # sağ 45°
            {"depth_z": 1.0, "azimuth": -45},   # sol 45°
        ],
    },
]


# ---------------------------------------------------------------------------
# Kayıt üretimi ve kaydetme
# ---------------------------------------------------------------------------

def generate_and_save(
    scenario_id: str,
    label: str,
    sources_spec: list[dict],    # [{'distance': float, 'azimuth': float}]
    mic_positions: np.ndarray,
    output_root: Path,
    snr_db: float,
    duration_s: float,
    rng: np.random.Generator,
    gt_path: Path | None = None,   # Ground truth XYZ'yi kaydet
) -> dict:
    """
    Tek bir senaryo için WAV + ground truth bilgisi üret ve kaydet.

    Returns:
        {'wav': Path, 'ground_truths': [[x,y,z], ...]}
    """
    n_samples = int(duration_s * SAMPLE_RATE)
    ground_truths = []
    sources_for_sim = []

    for i, spec in enumerate(sources_spec):
        pos = azimuth_to_xyz(spec["depth_z"], spec["azimuth"])
        ground_truths.append(pos)
        # 1000–8000 Hz: gerçek kuş sesi bandı.
        # MVDR/MUSIC, grating lobe'ları yüksek frekanslarda baskılar.
        signal = pink_noise(n_samples, rng,
                            f_low=spec.get("f_low", 1000),
                            f_high=spec.get("f_high", 8000))
        sources_for_sim.append({
            "pos": pos,
            "signal": signal,
            "amplitude": spec.get("amplitude", 1.0),
        })

    audio = simulate_array_recording(
        mic_positions=mic_positions,
        sources=sources_for_sim,
        snr_db=snr_db,
        duration_s=duration_s,
        rng=rng,
    )

    out_dir = output_root / f"sim_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE, subtype="FLOAT")

    # Ground truth dosyası
    import json
    gt_data = {
        "scenario_id": scenario_id,
        "label": label,
        "ground_truths": ground_truths,
        "snr_db": snr_db,
        "duration_s": duration_s,
        "n_sources": len(ground_truths),
    }
    gt_file = out_dir / "ground_truth.json"
    with open(gt_file, "w") as f:
        json.dump(gt_data, f, indent=2)

    logger.info(
        f"[{scenario_id}] → {wav_path.relative_to(project_root)}  "
        f"({audio.shape[0]} samples, {len(ground_truths)} kaynak, SNR={snr_db}dB)"
    )
    return {"wav": wav_path, "ground_truths": ground_truths}


# ---------------------------------------------------------------------------
# Ana program
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P1 / Cocktail için simüle çok kanallı WAV dosyaları üretir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenarios", nargs="+", default=["P1", "cocktail"],
                        choices=["P1", "cocktail"],
                        help="Üretilecek senaryolar")
    parser.add_argument("--snr", type=float, default=15.0,
                        help="Sinyal-gürültü oranı (dB)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Kayıt süresi (saniye)")
    parser.add_argument("--output", default="data/simulated",
                        help="Çıktı kök dizini")
    parser.add_argument("--seed", type=int, default=42,
                        help="Rastgelelik tohumu (tekrarlanabilirlik için)")
    parser.add_argument("--geom", default="config/micgeom.xml",
                        help="Mikrofon geometrisi XML dosyası")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_root = (project_root / args.output).resolve()
    geom_path = (project_root / args.geom).resolve()

    if not geom_path.exists():
        logger.error(f"Geometri dosyası bulunamadı: {geom_path}")
        sys.exit(1)

    mic_positions = load_mic_positions(geom_path)
    logger.info(f"{len(mic_positions)} mikrofon konumu yüklendi.")

    all_metadata = {}

    # ------- P1 senaryoları -------
    if "P1" in args.scenarios:
        logger.info("── P1 Lokalizasyon senaryoları ──────────────────────────────")
        for sc in P1_SCENARIOS:
            result = generate_and_save(
                scenario_id=sc["id"],
                label=sc["label"],
                sources_spec=[{"depth_z": sc["depth_z"], "azimuth": sc["azimuth"]}],
                mic_positions=mic_positions,
                output_root=output_root,
                snr_db=args.snr,
                duration_s=args.duration,
                rng=rng,
            )
            all_metadata[sc["id"]] = result

    # ------- Cocktail senaryoları -------
    if "cocktail" in args.scenarios:
        logger.info("── Cocktail senaryoları ─────────────────────────────────────")
        for sc in COCKTAIL_SCENARIOS:
            result = generate_and_save(
                scenario_id=sc["id"],
                label=sc["label"],
                sources_spec=sc["sources"],
                mic_positions=mic_positions,
                output_root=output_root,
                snr_db=args.snr,
                duration_s=args.duration,
                rng=rng,
            )
            all_metadata[sc["id"]] = result

    # ------- Özet ve değerlendirme komutları -------
    print("\n" + "=" * 72)
    print("  ÜRETİLEN DOSYALAR VE DEĞERLENDİRME KOMUTLARI")
    print("=" * 72)

    for sc_id, meta in all_metadata.items():
        wav_rel = meta["wav"].relative_to(project_root)
        gts = meta["ground_truths"]

        if len(gts) == 1:
            x, y, z = gts[0]
            print(f"\n# {sc_id}")
            print(f"python scripts/evaluate_pipeline.py \\")
            print(f'  --recording "{wav_rel}" \\')
            print(f"  --ground-truth {x:.4f} {y:.4f} {z:.4f} \\")
            print(f"  --algorithms DAS MVDR MUSIC Hybrid \\")
            print(f"  --output results/simulated/{sc_id}/")
        else:
            gt_args = " ".join(
                f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in gts
            )
            print(f"\n# {sc_id} — {len(gts)} kaynak")
            print(f"python scripts/evaluate_pipeline_multisource.py \\")
            print(f'  --recording "{wav_rel}" \\')
            print(f"  --ground-truths {gt_args} \\")
            print(f"  --output results/simulated/{sc_id}/")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
