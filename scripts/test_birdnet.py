"""
BirdNET Test Scripti
Kullanım: python scripts/test_birdnet.py [--recordings-dir data/recordings] [--lat LAT] [--lon LON]

data/recordings/ altındaki tüm WAV dosyalarını tarar ve BirdNET sonuçlarını raporlar.
Her kayıt için:
  - Ham kanal 0 (tek mikrofon baseline) analizi
  - Tespit edilen türler ve güven skorları
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Proje kökünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_birdnet")

# TensorFlow loglarını sustur
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def find_wav_files(recordings_dir: Path) -> list[Path]:
    wavs = []
    if recordings_dir.is_dir():
        for item in sorted(recordings_dir.iterdir()):
            if item.is_file() and item.suffix.lower() == ".wav":
                wavs.append(item)
            elif item.is_dir():
                for f in sorted(item.glob("*.wav")):
                    wavs.append(f)
    return wavs


def analyze_file(clf, wav_path: Path, lat, lon) -> dict:
    logger.info(f"Analiz ediliyor: {wav_path.name}")
    try:
        detections = clf.classify_file(
            str(wav_path),
            lat=lat,
            lon=lon,
            channel=0,  # Ham kanal 0 (tek mikrofon baseline)
        )
        return {"path": wav_path, "detections": detections, "error": None}
    except Exception as e:
        logger.error(f"Hata ({wav_path.name}): {e}")
        return {"path": wav_path, "detections": [], "error": str(e)}


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 65)
    print("  BirdNET ANALİZ RAPORU")
    print("=" * 65)

    total_detections = 0
    all_species: dict[str, float] = {}

    for r in results:
        dets = r["detections"]
        total_detections += len(dets)
        status = f"HATA: {r['error']}" if r["error"] else f"{len(dets)} tespit"
        print(f"\n{r['path'].name}  →  {status}")
        if dets:
            for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
                print(
                    f"  {d['common_name']:<35} "
                    f"{d['confidence']:.3f}  "
                    f"[{d['start_time']:.1f}s–{d['end_time']:.1f}s]"
                )
                # En yüksek güven skorunu tür bazında topla
                name = d["common_name"]
                all_species[name] = max(all_species.get(name, 0.0), d["confidence"])

    print("\n" + "=" * 65)
    print(f"TOPLAM: {len(results)} kayıt, {total_detections} tespit")
    if all_species:
        print(f"\nTespit edilen türler ({len(all_species)} tür):")
        for sp, conf in sorted(all_species.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(conf * 20)
            print(f"  {sp:<35} {conf:.3f}  {bar}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BirdNET ses dosyası test scripti")
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("data/recordings"),
        help="Kayıt dizini (varsayılan: data/recordings)",
    )
    parser.add_argument("--lat", type=float, default=None, help="Enlem (isteğe bağlı)")
    parser.add_argument("--lon", type=float, default=None, help="Boylam (isteğe bağlı)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.1,
        help="Minimum güven eşiği (varsayılan: 0.1)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Tek bir dosyayı analiz et (recordings-dir yerine)",
    )
    args = parser.parse_args()

    from src.classification.birdnet import BirdNETClassifier

    clf = BirdNETClassifier(
        min_confidence=args.min_confidence,
        lat=args.lat,
        lon=args.lon,
    )

    if args.file:
        wav_files = [args.file]
    else:
        base = Path(args.recordings_dir)
        if not base.exists():
            logger.error(f"Dizin bulunamadı: {base}")
            sys.exit(1)
        wav_files = find_wav_files(base)
        if not wav_files:
            logger.warning(f"{base} altında WAV dosyası bulunamadı.")
            sys.exit(0)

    logger.info(f"{len(wav_files)} WAV dosyası bulundu.")

    results = [analyze_file(clf, f, args.lat, args.lon) for f in wav_files]
    print_report(results)


if __name__ == "__main__":
    main()
