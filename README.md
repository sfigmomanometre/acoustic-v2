# UMA-16 Akustik Kamera Sistemi

> Real-time akustik kaynak lokalizasyonu ve 3D ses haritalama sistemi

## 📋 Proje Özeti

miniDSP UMA-16 mikrofon dizisi ve USB kamera kullanarak gerçek zamanlı akustik kaynak lokalizasyonu yapan sistem. Beamforming algoritmaları ile ses kaynaklarının konumlarını tespit edip video üzerinde görselleştirir.

**Tez**: Yüksek Lisans - Emre Göktuğ AKTAŞ  
**Tarih**: Aralık 2024

## 🎯 Proje Hedefleri

- ✅ Mikrofon dizisi geometrisi tanımlama (XML parser)
- ✅ Real-time ses verisi toplama (miniDSP UMA-16)
- ✅ Beamforming algoritmaları (DAS implementasyonu)
- ✅ Akustik harita oluşturma ve görselleştirme
- ✅ USB kamera entegrasyonu
- ✅ Video-akustik overlay (Full-screen mapping)
- ✅ GUI arayüzü (PySide6/Qt6)
- ⚠️ Performans optimizasyonu (devam ediyor)
- [ ] Kamera kalibrasyonu (perspektif düzeltme)
- [ ] MVDR, MUSIC algoritmaları

## 🔧 Donanım

- **Mikrofon Dizisi**: miniDSP UMA-16 (16 kanal)
- **Kamera**: USB webcam
- **Platform**: macOS

## 🚀 Hızlı Başlangıç

### 1. Virtual Environment Kurulumu

```bash
# Proje dizinine git
cd acoustic-v2

# Virtual environment oluştur
python3 -m venv venv

# Aktif et (macOS/Linux)
source venv/bin/activate

# Kütüphaneleri kur
pip install -r requirements.txt
```

### 2. Mikrofon Geometrisini Kontrol Et

```bash
# Jupyter notebook başlat
jupyter notebook

# notebooks/01_mic_geometry_check.ipynb açın ve çalıştırın
```

### 3. Ses Cihazı Testi

```bash
python -m src.audio.device_test
```

## 📁 Proje Yapısı

```
acoustic-v2/
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
├── .gitignore                  # Git ignore dosyası
├── config/
│   ├── micgeom.xml             # Mikrofon geometrisi (UMA-16)
│   └── config.yaml             # Sistem konfigürasyonu
├── src/
│   ├── __init__.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   └── parser.py           # XML geometri parser
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── stream.py           # Real-time ses akışı
│   │   ├── device_test.py      # Cihaz test aracı
│   │   └── preprocessing.py    # Ses ön işleme
│   ├── beamforming/
│   │   ├── __init__.py
│   │   ├── algorithms.py       # Beamforming algoritmaları
│   │   └── grid.py             # Grid hesaplama
│   ├── video/
│   │   ├── __init__.py
│   │   └── capture.py          # Kamera yakalama
│   └── visualization/
│       ├── __init__.py
│       └── plotter.py          # Gerçek zamanlı çizim
├── tests/
│   ├── __init__.py
│   ├── test_geometry.py
│   ├── test_audio.py
│   └── test_beamforming.py
├── notebooks/
│   ├── 01_mic_geometry_check.ipynb
│   ├── 02_audio_stream_test.ipynb
│   └── 03_beamforming_offline.ipynb
├── data/
│   ├── recordings/             # Test kayıtları
│   └── calibration/            # Kalibrasyon verileri
└── main.py                     # Ana uygulama

```

## 📚 Teknoloji Stack

### Backend
- **Acoular**: Akustik beamforming kütüphanesi
- **NumPy/SciPy**: Sinyal işleme
- **SoundDevice**: Real-time ses I/O
- **OpenCV**: Video işleme

### Frontend (Gelecek)
- **PyQt5** veya **Tkinter**: GUI framework
- **Matplotlib**: Akustik harita görselleştirme

## 🔄 Geliştirme Aşamaları

### ✅ Faz 1: Temel Altyapı (Şu an buradayız)
- [x] Proje yapısı oluşturma
- [ ] Virtual environment kurulumu
- [ ] Mikrofon geometrisi parser
- [ ] Cihaz bağlantı testleri

### 🔲 Faz 2: Offline Beamforming
- [ ] Test verisi toplama
- [ ] DAS algoritması implementasyonu
- [ ] Akustik harita üretimi
- [ ] Görselleştirme

### 🔲 Faz 3: Real-Time Sistem
- [ ] Streaming ses işleme
- [ ] Real-time beamforming
- [ ] Kamera entegrasyonu
- [ ] Video overlay

### 🔲 Faz 4: GUI ve İyileştirmeler
- [ ] Arayüz tasarımı
- [ ] Parametre kontrolleri
- [ ] Performans optimizasyonu
- [ ] Kalibrasyon araçları

## ⚙️ Konfigürasyon

`config/config.yaml` dosyasında tüm sistem parametrelerini düzenleyebilirsiniz:

```yaml
audio:
  sample_rate: 48000
  channels: 16
  chunk_size: 4096

beamforming:
  algorithm: "DAS"
  frequency_range: [500, 8000]
  
video:
  resolution: [1920, 1080]
  fps: 30
```

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest tests/

# Belirli bir test
pytest tests/test_geometry.py -v
```

## 📖 Referanslar

- [Acoular Documentation](http://acoular.org/)
- [miniDSP UMA-16 Specs](https://www.minidsp.com/products/usb-audio-interface/uma-16)
- Beamforming Theory: Johnson & Dudgeon

## 🐛 Sorun Giderme

### miniDSP tanınmıyor?
```bash
# Cihazları listele
python -c "import sounddevice as sd; print(sd.query_devices())"

# macOS ses izinlerini kontrol edin
```

### Import hataları?
```bash
# Virtual environment'ın aktif olduğundan emin olun
which python  # venv/bin/python görmeli

# Kütüphaneleri tekrar kurun
pip install -r requirements.txt --force-reinstall
```

## 📝 Changelog

- **2024-12-03**: 
  - Proje başlangıcı
  - Temel klasör yapısı oluşturuldu
  - README ve requirements hazırlandı

---

**Geliştirici**: Emre Göktuğ AKTAŞ  
**Lisans**: MIT (veya akademik kullanım)
