"""
UMA-16 Akustik Kamera Sistemi - Ana Uygulama
Real-time akustik kaynak lokalizasyonu

Kullanım:
    python main.py --mode test          # Cihaz testleri
    python main.py --mode geometry      # Geometri görselleştirme
    python main.py --mode record        # Test kaydı
    python main.py --mode beamforming   # Beamforming (yakında)
    python main.py --mode gui           # GUI (yakında)
"""

import argparse
import sys
import logging
from pathlib import Path

# Logging ayarla
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Proje modüllerini import et
from src.geometry.parser import MicGeometryParser
from src.audio.device_test import list_audio_devices, test_uma16_connection


def mode_test():
    """Cihaz testleri"""
    print("\n" + "="*70)
    print("🔧 CİHAZ TEST MODU")
    print("="*70 + "\n")
    
    # Ses cihazlarını listele
    list_audio_devices()
    
    # UMA-16 testi
    print("\n🎤 UMA-16 Bağlantı Testi...\n")
    success = test_uma16_connection(duration=2.0)
    
    if success:
        print("\n✅ Tüm testler başarılı!")
    else:
        print("\n❌ Test başarısız oldu.")
        sys.exit(1)


def mode_geometry(config_path: str = "config/config.yaml"):
    """Mikrofon geometrisini yükle ve görselleştir"""
    print("\n" + "="*70)
    print("📐 GEOMETRİ GÖRSELLEŞTİRME MODU")
    print("="*70 + "\n")
    
    # XML dosya yolu
    xml_path = "config/micgeom.xml"
    
    if not Path(xml_path).exists():
        logger.error(f"❌ Geometri dosyası bulunamadı: {xml_path}")
        sys.exit(1)
    
    # Parser oluştur ve parse et
    parser = MicGeometryParser(xml_path)
    positions = parser.parse()
    
    # Bilgileri göster
    info = parser.get_array_info()
    
    print("\n📊 MİKROFON DİZİSİ BİLGİLERİ:")
    print("-" * 70)
    for key, value in info.items():
        if key != 'bounding_box':
            print(f"  {key}: {value}")
    
    # Görselleştir
    print("\n🎨 Görselleştirme açılıyor...")
    parser.visualize()
    
    print("\n✅ Geometri başarıyla yüklendi!")


def mode_record(duration: float = 5.0, output_dir: str = "data/recordings"):
    """Test kaydı yap"""
    print("\n" + "="*70)
    print("🎙️ KAYIT MODU")
    print("="*70 + "\n")
    
    import sounddevice as sd
    import soundfile as sf
    from datetime import datetime
    import numpy as np
    
    # Output klasörünü oluştur
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # UMA-16 cihazını bul
    from src.audio.device_test import find_uma16_device
    device_id = find_uma16_device()
    
    if device_id is None:
        print("❌ UMA-16 cihazı bulunamadı!")
        sys.exit(1)
    
    # Kayıt ayarları
    sample_rate = 48000
    channels = 16
    
    print(f"⚙️  Ayarlar:")
    print(f"  Süre: {duration} saniye")
    print(f"  Örnekleme Hızı: {sample_rate} Hz")
    print(f"  Kanal: {channels}")
    print(f"\n🎤 Kayıt başlıyor...")
    print("  (Mikrofon yakınına ses çıkarın!)\n")
    
    # Kayıt
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        device=device_id,
        dtype='float32'
    )
    
    sd.wait()
    
    # Dosya adı (timestamp ile)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"test_recording_{timestamp}.wav"
    
    # Kaydet
    sf.write(filename, recording, sample_rate)
    
    print(f"✅ Kayıt tamamlandı!")
    print(f"  Dosya: {filename}")
    print(f"  Boyut: {recording.shape}")
    print(f"  RMS (ortalama): {np.sqrt(np.mean(recording**2)):.6f}")
    
    # Kanal istatistikleri
    print(f"\n📊 Kanal İstatistikleri:")
    active_channels = 0
    for ch in range(channels):
        rms = np.sqrt(np.mean(recording[:, ch]**2))
        if rms > 1e-5:
            active_channels += 1
    
    print(f"  Aktif kanal: {active_channels}/{channels}")


def mode_info():
    """Sistem bilgilerini göster"""
    print("\n" + "="*70)
    print("ℹ️  SİSTEM BİLGİLERİ")
    print("="*70 + "\n")
    
    import platform
    import numpy as np
    import scipy
    import sounddevice as sd
    
    try:
        import acoular
        acoular_version = acoular.__version__
    except:
        acoular_version = "Yüklü değil"
    
    print(f"🖥️  Platform:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    
    print(f"\n📦 Kütüphaneler:")
    print(f"  NumPy: {np.__version__}")
    print(f"  SciPy: {scipy.__version__}")
    print(f"  SoundDevice: {sd.__version__}")
    print(f"  Acoular: {acoular_version}")
    
    print(f"\n🎤 Ses Cihazları:")
    from src.audio.device_test import find_uma16_device
    device_id = find_uma16_device()
    
    if device_id is not None:
        print(f"  ✅ UMA-16 bulundu (Device {device_id})")
    else:
        print(f"  ❌ UMA-16 bulunamadı")
    
    print(f"\n📁 Proje Yapısı:")
    print(f"  Config: {'✅' if Path('config/config.yaml').exists() else '❌'}")
    print(f"  Geometri XML: {'✅' if Path('config/micgeom.xml').exists() else '❌'}")
    print(f"  Data klasörü: {'✅' if Path('data').exists() else '❌'}")


def main():
    parser = argparse.ArgumentParser(
        description="UMA-16 Akustik Kamera Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --mode test              # Cihaz testleri
  python main.py --mode geometry          # Geometri görselleştirme
  python main.py --mode record --duration 10  # 10 saniye kayıt
  python main.py --mode info              # Sistem bilgileri
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'geometry', 'record', 'info'],
        default='info',
        help='Çalışma modu'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Kayıt süresi (saniye) [mode=record için]'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Konfigürasyon dosyası yolu'
    )
    
    args = parser.parse_args()
    
    # Mod seçimine göre çalıştır
    try:
        if args.mode == 'test':
            mode_test()
        elif args.mode == 'geometry':
            mode_geometry(args.config)
        elif args.mode == 'record':
            mode_record(duration=args.duration)
        elif args.mode == 'info':
            mode_info()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Hata: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # ASCII Art Banner
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        UMA-16 Akustik Kamera Sistemi                         ║
    ║        Real-time Akustik Kaynak Lokalizasyonu                ║
    ║                                                               ║
    ║        Yüksek Lisans Tezi - Emre Göktuğ AKTAŞ                ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    main()
