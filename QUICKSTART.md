# 🚀 Hızlı Başlangıç Kılavuzu

## ✅ Kurulum Tamamlandı!

Tebrikler! Temel altyapı başarıyla kuruldu. Şu an yapabilecekleriniz:

## 🎯 Mevcut Özellikler (Faz 1 - TAMAMLANDI)

### 1️⃣ Sistem Bilgileri
```bash
python main.py --mode info
```
- Python ve kütüphane versiyonları
- UMA-16 bağlantı durumu
- Proje dosyaları kontrolü

### 2️⃣ Cihaz Testleri
```bash
python main.py --mode test
```
- Tüm ses cihazlarını listeler
- UMA-16'dan 2 saniyelik test kaydı alır
- 16 kanalın hepsini kontrol eder
- ✅ **BAŞARILI**: 16/16 kanal çalışıyor!

### 3️⃣ Mikrofon Geometrisi
```bash
python main.py --mode geometry
```
- XML dosyasını parse eder
- Dizi bilgilerini gösterir
- 3D + 2D görselleştirme yapar

### 4️⃣ Test Kaydı
```bash
python main.py --mode record --duration 10
```
- 10 saniyelik WAV kaydı alır
- `data/recordings/` klasörüne kaydeder
- Timestamp ile otomatik isimlendirme

## 📊 Durum Raporu

```
✅ Virtual environment (.venv/)
✅ Kütüphane kurulumları (numpy, scipy, acoular, sounddevice, opencv...)
✅ UMA-16 bağlantısı (16/16 kanal aktif)
✅ Mikrofon geometrisi (config/micgeom.xml)
✅ Config dosyaları (config/config.yaml)
✅ Klasör yapısı
✅ Test modülleri
✅ Ana uygulama (main.py)

🔲 Audio stream modülü (Sonraki adım)
🔲 Beamforming algoritmaları
🔲 Real-time işleme
🔲 Video entegrasyonu
🔲 GUI arayüzü
```

## 📝 Sonraki Adımlar (Faz 2)

### Backend #2: Audio Stream Modülü
**Amaç**: Real-time ses akışı için buffer yönetimi

**Yapılacaklar:**
1. `src/audio/stream.py` - Streaming class
2. Circular buffer implementasyonu
3. Callback mekanizması
4. Notebook test: `02_audio_stream_test.ipynb`

### Backend #3: Beamforming Algoritmaları
**Amaç**: Offline veri ile beamforming test

**Yapılacaklar:**
1. `src/beamforming/algorithms.py` - DAS implementasyonu
2. `src/beamforming/grid.py` - Grid hesaplama
3. Test kaydı üzerinde deneme
4. Notebook: `03_beamforming_offline.ipynb`

## 🧪 Test Senaryosu Önerisi

1. **Basit Test (Şimdi yapabilirsiniz!)**
   ```bash
   # 5 saniye kayıt al (konuşun veya müzik çalın)
   python main.py --mode record --duration 5
   
   # Kaydedilen dosyayı kontrol edin
   ls -lh data/recordings/
   ```

2. **Geometri Kontrolü**
   ```bash
   # Jupyter notebook başlat
   jupyter notebook
   
   # notebooks/01_mic_geometry_check.ipynb açın ve çalıştırın
   ```

3. **Manuel Cihaz Testi**
   ```bash
   # Belirli bir kanalı test et (örnek: Kanal 5)
   python -m src.audio.device_test --channel 5 --duration 3
   ```

## 🎓 Öğrenim Kaynakları

### Beamforming Temelleri
- Acoular Tutorial: http://acoular.org/get_started/index.html
- Delay-and-Sum kavramı
- Steering vector hesaplama
- Acoustic maps

### Kod Örnekleri
```python
# Geometri kullanımı
from src.geometry.parser import MicGeometryParser
parser = MicGeometryParser('config/micgeom.xml')
mic_geom = parser.to_acoular()
print(f"Mikrofon sayısı: {mic_geom.num_mics}")

# Ses cihazı kontrolü
from src.audio.device_test import find_uma16_device, test_uma16_connection
device_id = find_uma16_device()
test_uma16_connection(duration=2.0)
```

## 🐛 Sorun Giderme

### Problem: "Module not found" hatası
```bash
# Virtual environment'ın aktif olduğundan emin olun
which python
# Çıktı: .../acoustic-v2/.venv/bin/python olmalı

# Eğer değilse:
source .venv/bin/activate  # macOS/Linux
```

### Problem: UMA-16 tanınmıyor
```bash
# Cihazları listele
python -c "import sounddevice as sd; print(sd.query_devices())"

# macOS ses izinlerini kontrol et
# System Preferences → Security & Privacy → Microphone
```

### Problem: Acoular uyarısı (OpenBLAS)
Bu uyarı normaldir ve performansı çok etkilemez. Hızlandırmak için:
```bash
export OPENBLAS_NUM_THREADS=1
python main.py ...
```

## 📞 Yardım

Bir sorun mu var? Şunları kontrol edin:

1. **Kütüphane versiyonları**: `python main.py --mode info`
2. **Cihaz bağlantısı**: `python main.py --mode test`
3. **Log dosyaları**: `acoustic_camera.log` (oluşturuluyorsa)

## 🎯 Hedef: Real-Time Akustik Kamera

**Vizyonumuz:**
```
[UMA-16 Mikrofonlar] → [Real-time Audio Stream] → [Beamforming] 
                                                         ↓
[USB Kamera] → [Video Capture] → [Overlay] → [GUI Display]
```

**Şu an buradayız:** ✅ Cihazlar hazır, geometri tanımlı, test başarılı!

**Sonraki durak:** 🚀 Real-time stream ve beamforming!

---

**Güncelleme**: 3 Aralık 2024  
**Durum**: Faz 1 Tamamlandı ✅  
**Sonraki**: Faz 2 - Audio Stream başlasın mı? 🤔
