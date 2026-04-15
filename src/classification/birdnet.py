"""
BirdNET Entegrasyon Modülü
Kullanım: Tek kanal (ham) ve beamformed ses üzerinde tür tespiti ve karşılaştırma.

Paper referansı: "Beyond Species Identification: Real-Time Spatial Interaction
Analysis in Avian Bioacoustics Using Microphone Arrays and Hybrid Beamforming
on Edge Architectures"
"""

import io
import os
import sys
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BIRDNET_SAMPLE_RATE = 48000

# Location filter açıkken bile her zaman dahil edilecek etiketler.
# Bu etiketler BirdNET model label listesindeki tam string'lerdir.
ALWAYS_INCLUDE_LABELS = [
    "Human non-vocal_Human non-vocal",
    "Human vocal_Human vocal",
    "Human whistle_Human whistle",
    "Noise_Noise",
]


def _ensure_venv_on_path():
    """Projenin .venv/site-packages dizinini sys.path'e ekle (idempotent)."""
    project_root = Path(__file__).resolve().parent.parent.parent
    venv_lib = project_root / ".venv" / "lib"
    if venv_lib.exists():
        for site_pkg in venv_lib.glob("python*/site-packages"):
            if str(site_pkg) not in sys.path:
                sys.path.insert(0, str(site_pkg))


def _install_tflite_shim():
    """
    ai_edge_litert'i tflite_runtime olarak kaydet.

    Python 3.12 + macOS'te tensorflow 2.21 içindeki _pywrap_cpu_feature_guard.so
    '_PyEval_SetProfileAllThreads' sembolünü bulamıyor (Py3.12'de kaldırıldı).
    ai_edge_litert bu sorunu yaşamıyor; birdnetlib'in `import tflite_runtime`
    çağrısını ai_edge_litert'e yönlendiriyoruz.
    """
    if "tflite_runtime" not in sys.modules:
        try:
            import ai_edge_litert.interpreter as _lrt_interp
            import types

            shim = types.ModuleType("tflite_runtime")
            shim.interpreter = _lrt_interp
            sys.modules["tflite_runtime"] = shim
            sys.modules["tflite_runtime.interpreter"] = _lrt_interp
            logger.debug("tflite_runtime → ai_edge_litert shim kuruldu.")
        except ImportError:
            pass  # ai_edge_litert yoksa birdnetlib kendi tensorflow fallback'ini dener


_ensure_venv_on_path()
_install_tflite_shim()


@contextlib.contextmanager
def _suppress_stdout():
    """birdnetlib'in aşırı print çıktılarını sustur."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class BirdNETClassifier:
    """
    birdnetlib tabanlı kuş türü tespit sınıfı.

    Hem dosya hem de numpy dizisi üzerinden çalışabilir.
    Ham tek kanal vs beamformed çıkışı karşılaştırması sunar.

    Örnek kullanım:
        clf = BirdNETClassifier(min_confidence=0.1)
        sonuclar = clf.classify_file("kayit.wav")
        for d in sonuclar:
            print(d['common_name'], d['confidence'])
    """

    def __init__(self, min_confidence: float = 0.1, sensitivity: float = 1.0):
        self.min_confidence = min_confidence
        self.sensitivity = sensitivity
        self._analyzer = None

    def _get_analyzer(self):
        if self._analyzer is None:
            from birdnetlib.analyzer import Analyzer
            logger.info("BirdNET modeli yükleniyor (ilk kullanım)...")
            with _suppress_stdout():
                self._analyzer = Analyzer()
            logger.info("BirdNET modeli hazır.")
        return self._analyzer

    # ------------------------------------------------------------------
    # Dosya tabanlı analiz
    # ------------------------------------------------------------------

    def classify_file(
        self,
        wav_path: str,
        date: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        overlap: float = 0.0,
        channel: Optional[int] = None,
    ) -> list[dict]:
        """
        WAV dosyasını analiz et, tespit edilen türleri döndür.

        Args:
            wav_path: WAV dosyası yolu (16 kanal desteklenir).
            date: Tarih (sezon bilgisi için; None ise bugün).
            min_confidence: Minimum güven eşiği (None ise instance default).
            overlap: 3 saniyelik pencereler arası örtüşme (0.0–2.9 saniye).
            channel: Çok kanallı WAV için kanal indeksi (0-indexed).
                     None ise librosa mono mix kullanılır.

        Returns:
            [{'common_name', 'scientific_name', 'confidence',
              'start_time', 'end_time'}, ...]
        """
        wav_path = str(wav_path)
        min_conf = min_confidence if min_confidence is not None else self.min_confidence

        if channel is not None:
            audio_np = self._load_channel(wav_path, channel)
            return self._classify_array(audio_np, date=date,
                                        min_confidence=min_conf, overlap=overlap)

        from birdnetlib.main import Recording
        analyzer = self._get_analyzer()
        with _suppress_stdout():
            rec = Recording(
                analyzer,
                wav_path,
                date=date or datetime.now(),
                sensitivity=self.sensitivity,
                min_conf=min_conf,
                overlap=overlap,
            )
            rec.analyze()
        return rec.detections

    # ------------------------------------------------------------------
    # Numpy dizisi tabanlı analiz (gerçek zamanlı kullanım için)
    # ------------------------------------------------------------------

    def classify_audio(
        self,
        audio_np: np.ndarray,
        sample_rate: int = BIRDNET_SAMPLE_RATE,
        date: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        overlap: float = 0.0,
        channel: int = 0,
        lat: float = -1,
        lon: float = -1,
        include_human: bool = True,
    ) -> list[dict]:
        """
        Numpy ses dizisini analiz et.

        Args:
            audio_np: (samples,) veya (samples, channels) şeklinde numpy dizisi.
            sample_rate: Ses örnekleme hızı (Hz). 48000 değilse yeniden örneklenir.
            channel: Çok kanallı giriş için kanal indeksi.
            lat: Enlem (lokasyon filtresi için). -1 = filtre kapalı.
            lon: Boylam (lokasyon filtresi için). -1 = filtre kapalı.

        Returns:
            classify_file ile aynı format.
        """
        mono = self._to_mono(audio_np, channel)
        if sample_rate != BIRDNET_SAMPLE_RATE:
            import librosa
            mono = librosa.resample(
                mono.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=BIRDNET_SAMPLE_RATE,
            )

        return self._classify_array(
            mono,
            date=date,
            min_confidence=min_confidence if min_confidence is not None else self.min_confidence,
            overlap=overlap,
            lat=lat,
            lon=lon,
            include_human=include_human,
        )

    # ------------------------------------------------------------------
    # Paper karşılaştırması: ham kanal 0 vs beamformed
    # ------------------------------------------------------------------

    def compare_raw_vs_beamformed(
        self,
        audio_16ch_np: np.ndarray,
        beamformed_mono_np: np.ndarray,
        sample_rate: int = BIRDNET_SAMPLE_RATE,
        date: Optional[datetime] = None,
        raw_channel: int = 0,
    ) -> dict:
        """
        Aynı kayıt için ham tek kanal vs beamformed BirdNET karşılaştırması.

        Returns:
            {
              'raw': [...detections...],
              'beamformed': [...detections...],
              'summary': {species → {raw_conf, beam_conf, improvement_db}},
              'new_detections': [...],
              'lost_detections': [...],
            }
        """
        raw_mono = self._to_mono(audio_16ch_np, raw_channel)
        if sample_rate != BIRDNET_SAMPLE_RATE:
            import librosa
            raw_mono = librosa.resample(
                raw_mono.astype(np.float32), orig_sr=sample_rate, target_sr=BIRDNET_SAMPLE_RATE)
            beamformed_mono_np = librosa.resample(
                beamformed_mono_np.astype(np.float32), orig_sr=sample_rate, target_sr=BIRDNET_SAMPLE_RATE)

        raw_dets = self._classify_array(raw_mono, date=date,
                                        min_confidence=self.min_confidence)
        beam_dets = self._classify_array(beamformed_mono_np, date=date,
                                         min_confidence=self.min_confidence)

        summary = self._build_comparison_summary(raw_dets, beam_dets)
        raw_species = {d["common_name"] for d in raw_dets}
        beam_species = {d["common_name"] for d in beam_dets}

        return {
            "raw": raw_dets,
            "beamformed": beam_dets,
            "summary": summary,
            "new_detections": sorted(beam_species - raw_species),
            "lost_detections": sorted(raw_species - beam_species),
        }

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _classify_array(
        self,
        audio_mono: np.ndarray,
        date: Optional[datetime],
        min_confidence: float,
        overlap: float = 0.0,
        lat: float = -1,
        lon: float = -1,
        include_human: bool = True,
    ) -> list[dict]:
        """Mono numpy dizisini BytesIO WAV olarak birdnetlib'e gönder.

        lat/lon=-1 → lokasyon filtresi kapalı (birdnetlib varsayılan davranışı).
        lat/lon verilirse yalnızca o bölgede görülebilen türler döner.
        include_human=True → lokasyon filtresi açık olsa bile ALWAYS_INCLUDE_LABELS
        (Human non-vocal, Human vocal, Human whistle, Noise) listeye eklenir.
        """
        import soundfile as sf
        from birdnetlib.main import RecordingFileObject

        buf = io.BytesIO()
        sf.write(buf, audio_mono.astype(np.float32), BIRDNET_SAMPLE_RATE,
                 format="WAV", subtype="FLOAT")
        buf.seek(0)

        analyzer = self._get_analyzer()
        with _suppress_stdout():
            rec = RecordingFileObject(
                analyzer,
                buf,
                date=date or datetime.now(),
                sensitivity=self.sensitivity,
                min_conf=min_confidence,
                overlap=overlap,
                lat=lat,
                lon=lon,
            )
            # Lokasyon filtresi aktifken insan/gürültü etiketleri silinir.
            # include_human=True ise bunları predicted_species_list'e geri ekle.
            if include_human and lat != -1:
                sp_list = analyzer.predicted_species_list
                for label in ALWAYS_INCLUDE_LABELS:
                    if label not in sp_list:
                        sp_list.append(label)
            rec.analyze()
        return rec.detections

    @staticmethod
    def _to_mono(audio_np: np.ndarray, channel: int = 0) -> np.ndarray:
        if audio_np.ndim == 1:
            return audio_np.astype(np.float32)
        if audio_np.shape[1] <= channel:
            raise ValueError(f"Kanal {channel} mevcut değil (toplam: {audio_np.shape[1]})")
        return audio_np[:, channel].astype(np.float32)

    @staticmethod
    def _load_channel(wav_path: str, channel: int) -> np.ndarray:
        import soundfile as sf
        data, _ = sf.read(wav_path, dtype="float32", always_2d=True)
        if data.shape[1] <= channel:
            raise ValueError(f"Kanal {channel} mevcut değil (toplam: {data.shape[1]})")
        return data[:, channel]

    @staticmethod
    def _build_comparison_summary(raw_dets: list, beam_dets: list) -> dict:
        raw_by_species: dict[str, float] = {}
        for d in raw_dets:
            name = d["common_name"]
            raw_by_species[name] = max(raw_by_species.get(name, 0.0), d["confidence"])

        beam_by_species: dict[str, float] = {}
        for d in beam_dets:
            name = d["common_name"]
            beam_by_species[name] = max(beam_by_species.get(name, 0.0), d["confidence"])

        all_species = set(list(raw_by_species.keys()) + list(beam_by_species.keys()))
        summary = {}
        for species in sorted(all_species):
            raw_conf = raw_by_species.get(species, 0.0)
            beam_conf = beam_by_species.get(species, 0.0)
            improvement_db = 20.0 * np.log10(max(beam_conf, 1e-9) / max(raw_conf, 1e-9))
            summary[species] = {
                "raw_confidence": round(raw_conf, 4),
                "beamformed_confidence": round(beam_conf, 4),
                "improvement_db": round(improvement_db, 2),
            }
        return summary

    def print_detections(self, detections: list[dict], label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        if not detections:
            print(f"{prefix}Tespit yok.")
            return
        print(f"{prefix}{len(detections)} tespit:")
        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
            print(
                f"  {d['common_name']:<35} "
                f"güven: {d['confidence']:.3f}  "
                f"[{d['start_time']:.1f}s – {d['end_time']:.1f}s]"
            )
