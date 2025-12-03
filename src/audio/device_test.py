"""
Audio Cihaz Test Modülü
miniDSP UMA-16 bağlantısını test etmek için yardımcı fonksiyonlar
"""

import sounddevice as sd
import numpy as np
import time
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_audio_devices() -> List[Dict]:
    """
    Sistemdeki tüm ses cihazlarını listeler.
    
    Returns:
        List[Dict]: Cihaz bilgileri listesi
    """
    devices = sd.query_devices()
    
    print("=" * 70)
    print("SİSTEMDEKİ SES CİHAZLARI")
    print("=" * 70)
    
    for i, device in enumerate(devices):
        is_default_input = (i == sd.default.device[0])
        is_default_output = (i == sd.default.device[1])
        
        marker = ""
        if is_default_input:
            marker = "< INPUT"
        if is_default_output:
            marker = "< OUTPUT"
        
        print(f"\n{i}: {device['name']} {marker}")
        print(f"   Max Input Channels: {device['max_input_channels']}")
        print(f"   Max Output Channels: {device['max_output_channels']}")
        print(f"   Default Sample Rate: {device['default_samplerate']} Hz")
    
    print("\n" + "=" * 70)
    
    return devices


def find_uma16_device() -> Optional[int]:
    """
    miniDSP UMA-16 cihazını bulur.
    
    Returns:
        Optional[int]: Cihaz ID'si, bulunamazsa None
    """
    devices = sd.query_devices()
    
    for i, device in enumerate(devices):
        # UMA-16 ismini ara (case insensitive)
        if 'uma' in device['name'].lower() and device['max_input_channels'] >= 16:
            logger.info(f"✓ UMA-16 bulundu: Device {i} - {device['name']}")
            return i
    
    logger.warning("⚠ UMA-16 cihazı bulunamadı!")
    return None


def test_uma16_connection(duration: float = 2.0, sample_rate: int = 48000) -> bool:
    """
    miniDSP UMA-16'dan kısa bir test kaydı yapar.
    
    Args:
        duration: Kayıt süresi (saniye)
        sample_rate: Örnekleme hızı (Hz)
    
    Returns:
        bool: Test başarılı ise True
    """
    device_id = find_uma16_device()
    
    if device_id is None:
        print("❌ UMA-16 cihazı bulunamadı!")
        return False
    
    try:
        print(f"\n{'=' * 70}")
        print(f"UMA-16 TEST KAYDI")
        print(f"{'=' * 70}")
        print(f"Cihaz ID: {device_id}")
        print(f"Süre: {duration} saniye")
        print(f"Örnekleme Hızı: {sample_rate} Hz")
        print(f"Kanal Sayısı: 16")
        print(f"\n🎤 Kayıt başlıyor...")
        
        # Kayıt yap
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=16,
            device=device_id,
            dtype='float32'
        )
        
        sd.wait()  # Kaydın bitmesini bekle
        
        print(f"✓ Kayıt tamamlandı!")
        print(f"\n{'=' * 70}")
        print(f"KAYIT İSTATİSTİKLERİ")
        print(f"{'=' * 70}")
        print(f"Shape: {recording.shape}")
        print(f"Duration: {recording.shape[0] / sample_rate:.2f} saniye")
        print(f"Channels: {recording.shape[1]}")
        
        # Her kanal için istatistikler
        print(f"\nKANAL İSTATİSTİKLERİ:")
        print(f"{'Kanal':<8} {'RMS':<12} {'Peak':<12} {'Aktif?'}")
        print(f"-" * 50)
        
        for ch in range(16):
            rms = np.sqrt(np.mean(recording[:, ch]**2))
            peak = np.max(np.abs(recording[:, ch]))
            is_active = "✓" if rms > 1e-6 else "✗ (sessiz)"
            
            print(f"{ch:<8} {rms:<12.6f} {peak:<12.6f} {is_active}")
        
        # Genel değerlendirme
        active_channels = np.sum([np.sqrt(np.mean(recording[:, ch]**2)) > 1e-6 for ch in range(16)])
        
        print(f"\n{'=' * 70}")
        print(f"SONUÇ: {active_channels}/16 kanal aktif")
        
        if active_channels == 0:
            print("⚠ UYARI: Hiçbir kanalda sinyal algılanmadı!")
            print("   - Mikrofonlar bağlı mı kontrol edin")
            print("   - Sessiz bir ortamda mı test ediyorsunuz?")
            print("   - Mikrofon gain ayarlarını kontrol edin")
        elif active_channels < 16:
            print(f"⚠ UYARI: Sadece {active_channels} kanal aktif")
            print("   - Tüm mikrofonların bağlı olduğundan emin olun")
        else:
            print("✓ Tüm kanallar çalışıyor!")
        
        print(f"{'=' * 70}\n")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test başarısız: {e}")
        return False


def test_single_channel(channel: int = 0, duration: float = 5.0, 
                       sample_rate: int = 48000, plot: bool = True) -> Optional[np.ndarray]:
    """
    Tek bir kanaldan test kaydı yapar ve görselleştirir.
    
    Args:
        channel: Test edilecek kanal numarası (0-15)
        duration: Kayıt süresi (saniye)
        sample_rate: Örnekleme hızı
        plot: Sonuçları görselleştir mi?
    
    Returns:
        Optional[np.ndarray]: Kayıt verisi, hata durumunda None
    """
    device_id = find_uma16_device()
    
    if device_id is None:
        return None
    
    if not (0 <= channel < 16):
        logger.error(f"Geçersiz kanal: {channel}. 0-15 arası olmalı.")
        return None
    
    try:
        print(f"\n🎤 Kanal {channel} - {duration} saniye kayıt yapılıyor...")
        
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=16,
            device=device_id,
            dtype='float32'
        )
        
        sd.wait()
        
        # Sadece istenen kanalı al
        channel_data = recording[:, channel]
        
        print(f"✓ Kayıt tamamlandı!")
        print(f"  RMS: {np.sqrt(np.mean(channel_data**2)):.6f}")
        print(f"  Peak: {np.max(np.abs(channel_data)):.6f}")
        
        # Görselleştirme
        if plot:
            import matplotlib.pyplot as plt
            
            time_axis = np.arange(len(channel_data)) / sample_rate
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Waveform
            ax1.plot(time_axis, channel_data, linewidth=0.5)
            ax1.set_xlabel('Zaman (s)')
            ax1.set_ylabel('Genlik')
            ax1.set_title(f'Kanal {channel} - Dalga Formu')
            ax1.grid(True, alpha=0.3)
            
            # Spektrogram
            ax2.specgram(channel_data, Fs=sample_rate, cmap='viridis')
            ax2.set_xlabel('Zaman (s)')
            ax2.set_ylabel('Frekans (Hz)')
            ax2.set_title(f'Kanal {channel} - Spektrogram')
            
            plt.tight_layout()
            plt.show()
        
        return channel_data
        
    except Exception as e:
        logger.error(f"Kanal test hatası: {e}")
        return None


# CLI için main fonksiyon
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UMA-16 Audio Cihaz Test Aracı")
    parser.add_argument('--list', action='store_true', help='Tüm cihazları listele')
    parser.add_argument('--test', action='store_true', help='UMA-16 bağlantı testi yap')
    parser.add_argument('--channel', type=int, help='Belirli bir kanalı test et (0-15)')
    parser.add_argument('--duration', type=float, default=2.0, help='Test süresi (saniye)')
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
    
    if args.test:
        test_uma16_connection(duration=args.duration)
    
    if args.channel is not None:
        test_single_channel(channel=args.channel, duration=args.duration, plot=True)
    
    if not (args.list or args.test or args.channel is not None):
        # Argüman verilmemişse default test
        print("UMA-16 Cihaz Test Aracı\n")
        list_audio_devices()
        print("\nBağlantı testi yapılıyor...\n")
        test_uma16_connection()
