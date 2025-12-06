# 🗺️ Akustik Heatmap Overlay - Yol Haritası ve Çözüm

## 📅 Tarih: 7 Aralık 2025

## 🎯 Problem Tanımı

GUI çalışıyor ve beamforming algoritması ses kaynaklarını tespit ediyor, ancak **akustik heatmap video üzerinde sadece küçük bir nokta olarak görünüyor**. Sesin geldiği konumu kamera görüntüsü üzerinde doğru şekilde haritalamak gerekiyor.

---

## ✅ Uygulanan Çözümler (Faz 1 - TAMAMLANDI)

### **1. Full-Screen Overlay Mapping**

**Değişiklik:** `_update_video_overlay()` fonksiyonu tamamen yeniden yazıldı.

**Öncesi:**
- Heatmap sadece video boyutunun %40'ı kadar resize ediliyordu
- Peak noktasına göre küçük bir alan gösteriliyordu
- Grid koordinatları doğru dönüştürülmüyordu

**Sonrası:**
- ✅ Heatmap **tüm video frame'ine** yayıldı
- ✅ Aspect ratio korunarak letterbox mantığı eklendi
- ✅ Grid koordinatları (X, Y metrik) → Piksel koordinatlarına düzgün mapping

```python
# Grid fiziksel boyutu: 1.2m x 1.2m @ 1.0m mesafe
# Video çözünürlüğü: 1920x1080 (örnek)
# Mapping: (-0.6, -0.6) → (0, 1080), (+0.6, +0.6) → (1920, 0)
```

### **2. Grid Boyutu Artırıldı**

**Değişiklik:** `BeamformingConfig` parametreleri güncellendi.

```python
# ÖNCE:
grid_size_x=0.6  # 60 cm - çok dar
grid_size_y=0.6

# SONRA:
grid_size_x=1.2  # 120 cm - kamera FOV'una uygun
grid_size_y=1.2
```

**Açıklama:**
- 1 metre mesafede tipik bir webcam ~60-80° FOV görür
- Bu, yaklaşık 1.0-1.5m fiziksel alan demektir
- Grid'i 1.2m yaparak kamera görüş alanının çoğunu kapsıyoruz

### **3. Koordinat Dönüşümü Düzeltildi**

**Önce:**
```python
# Basit projeksiyon - yanlış
video_x = int((norm_x + 1.0) / 2.0 * video_w)
```

**Sonra:**
```python
# Doğru: Grid metrik → Normalize → Piksel
norm_x = (peak_x_m + grid_size_x / 2.0) / grid_size_x  # 0 to 1
peak_pixel_x = int(norm_x * overlay_w)
peak_video_x = x_offset + peak_pixel_x  # Letterbox offset ekle
```

### **4. Geliştirilmiş Crosshair & Info Display**

```python
# Peak noktasında:
- ✅ Yeşil + işareti (30 piksel, kalın)
- ✅ Dolgulu daire (merkez)
- ✅ dB seviyesi (örnek: "-15.3 dB")
- ✅ Pozisyon bilgisi (örnek: "(25.3, -12.7) cm")
- ✅ Siyah gölge + yeşil yazı (okunabilirlik)
```

---

## 🔧 Parametreler (Optimize Edildi)

### **Beamforming Config**
```yaml
Grid Size: 1.2m x 1.2m        # Geniş alan kapsaması
Grid Resolution: 5 cm         # Dengeli performans
Focus Distance: 1.0m          # Tipik kullanım mesafesi
Frequency Range: 500-8000 Hz  # İnsan sesi + ambient
Field Type: near-field        # Doğru lokalizasyon
```

### **Görselleştirme**
```yaml
Colormap: jet (default)       # Klasik sıcak-soğuk
Alpha: 60%                    # Video görünür kalıyor
dB Range: -40 to -10 dB      # Noise floor filtreleme
Threshold: 10% above min      # Düşük sinyaller maskeleniyor
```

---

## 🚀 Nasıl Test Edilir?

### **Adım 1: GUI'yi Başlat**
```bash
cd /Users/emregoktugaktas/Desktop/Yüksek\ Lisans\ TEZ/codes/acoustic-v2
source venv/bin/activate  # veya: source .venv/bin/activate
python run_gui.py
```

### **Adım 2: Sistem Başlat**
1. **Audio Cihazı:** `UMA16v2 (Auto)` seçili olmalı
2. **Video Cihazı:** `Webcam 0` (veya USB kamera)
3. **Beamforming:** Checkbox'ı **aktif** et
4. **Video Overlay:** Checkbox'ı **aktif** et
5. **DURDUR/BAŞLAT** butonuna bas → Yeşil görünmeli

### **Adım 3: Ses Üret ve Gözlemle**
- Konuş veya müzik çal
- **Mavi/kırmızı heatmap** tüm ekrana yayılmalı
- **Yeşil crosshair** sesin geldiği yerde olmalı
- **dB ve pozisyon bilgisi** crosshair yanında görünmeli

### **Adım 4: Parametreleri Ayarla**
- **Alpha slider:** Heatmap'i daha belirgin/şeffaf yap
- **dB Range slider:** Hassasiyeti ayarla
- **Frekans slider:** Odaklanmak istediğin ses aralığını seç

---

## 🐛 Hâlâ Sorun Varsa

### **Problem 1: Heatmap Görünmüyor**
**Çözüm:**
- `Alpha slider`'ı 80-100%'e çek
- `dB Range` slider'ı ayarla (örnek: -50 to -5)
- VU meter'ları kontrol et - ses geliyor mu?

### **Problem 2: Crosshair Yanlış Yerde**
**Olası Neden:** Kamera ve mikrofon dizisi fiziksel konumu uyuşmuyor
**Geçici Çözüm:** Kamerayı mikrofon dizisinin merkezine yerleştir
**Kalıcı Çözüm:** Kalibrasyon gerekir (Faz 2)

### **Problem 3: Heatmap Çok Yavaş**
**Çözüm:**
- `Grid Çözünürlüğü` değerini artır (5cm → 8cm)
- `Beamforming interval` kodda artırılabilir
- Frekans aralığını daralt (örnek: 1000-4000 Hz)

---

## 📋 Sonraki Adımlar (Faz 2 - TODO)

### **1. Kamera Kalibrasyonu** 🔴 YÜKSEK ÖNCELİK

**Amaç:** 3D akustik grid → 2D kamera pikselleri dönüşümünü doğru yapmak

**Gerekli İşler:**
```python
# Kamera intrinsic parametreleri
- Focal length (fx, fy)
- Principal point (cx, cy)
- Lens distortion coefficients (k1, k2, p1, p2)

# Extrinsic parametreleri (mikrofon dizisi → kamera)
- Rotation matrix (R)
- Translation vector (T)

# Kalibrasyon toolları:
- OpenCV calibration (checkerboard pattern)
- Manual alignment GUI
```

**Dosya:** `src/calibration/camera_calibration.py` (yeni)

**Kullanım:**
```python
from calibration.camera_calibration import CameraCalibration

# Calibration yükle
calib = CameraCalibration.load("config/camera_params.yaml")

# 3D → 2D projection
pixel_x, pixel_y = calib.project_3d_to_2d(grid_point_3d)
```

### **2. Perspective Projection** 🟡 ORTA ÖNCELİK

**Problem:** Şu anki mapping basit linear interpolation kullanıyor.

**Çözüm:** Gerçek perspektif projeksiyon kullan:
```python
# Pinhole camera model
[u]   [fx  0  cx]   [X]
[v] = [ 0 fy cy] * [Y]
[1]   [ 0  0  1]   [Z]
```

### **3. Multi-Source Detection** 🟢 DÜŞÜK ÖNCELİK

**Amaç:** Birden fazla ses kaynağını aynı anda göster

**Gerekli:**
- Peak detection algoritması (local maxima)
- N en yüksek peak'i bul
- Her biri için crosshair çiz

### **4. Temporal Smoothing** 🟢 DÜŞÜK ÖNCELİK

**Amaç:** Heatmap'teki titremeleri azalt

**Gerekli:**
- Moving average (son N frame)
- Kalman filter (peak tracking için)

---

## 📊 Performans Metrikleri

### **Hedef:**
- Real-time: **25-30 FPS** (beamforming + overlay)
- Latency: **< 100 ms** (ses → görüntü)
- Grid boyutu: **30x30 = 900 nokta** (optimal)

### **Şu Anki Durum:**
- FPS: ~15-20 (beamforming her 2 callback'te bir)
- Grid: Varsayılan 24x24 = 576 nokta (5cm resolution)
- CPU: ~20-30% (single thread)

### **Optimizasyon Fırsatları:**
- Numba JIT compilation → **2-3x hızlanma**
- GPU acceleration (CuPy) → **5-10x hızlanma**
- Multi-threading → **1.5-2x hızlanma**

---

## 🎓 Tez İçin Notlar

### **Bölüm: Görselleştirme ve Overlay**

**Algoritma:**
1. Beamforming → Power map (N×N grid, dB cinsinden)
2. Normalizasyon & Thresholding → Noise floor kaldır
3. Gaussian smoothing → Keskin kenarları yumuşat
4. Colormap uygula (Jet, Hot, Viridis, etc.)
5. Alpha blending → Video frame ile karıştır
6. Aspect ratio düzeltmesi → Letterbox/pillarbox
7. Peak detection → En yüksek güç noktası bul
8. Annotation → Crosshair, dB, pozisyon

**Diyagram için:**
```
┌─────────────────┐
│ Audio (16 ch)   │
└────────┬────────┘
         │ FFT
         ▼
┌─────────────────┐        ┌─────────────────┐
│ Cross-Spectral  │        │ Video Frame     │
│ Matrix (CSM)    │        │ (1920x1080)     │
└────────┬────────┘        └────────┬────────┘
         │ Beamforming            │
         ▼                        │
┌─────────────────┐              │
│ Power Map       │              │
│ (24x24 grid)    │              │
└────────┬────────┘              │
         │ Visualization         │
         ▼                        │
┌─────────────────┐              │
│ Heatmap (RGBA)  │──── Overlay ─┤
│ (1920x1080)     │              │
└─────────────────┘              ▼
                        ┌─────────────────┐
                        │ Final Display   │
                        └─────────────────┘
```

---

## 📝 Değişiklik Logu

### **v0.2 (7 Aralık 2025)**
- ✅ Full-screen overlay mapping
- ✅ Grid boyutu 60cm → 120cm
- ✅ Koordinat dönüşümü düzeltildi
- ✅ Crosshair & info display iyileştirildi
- ✅ Aspect ratio koruması eklendi

### **v0.1 (Önceki)**
- Basic beamforming (DAS)
- Küçük overlay (40% video boyutu)
- Peak-centered görüntüleme

---

## 🔗 İlgili Dosyalar

```
src/gui/main_window.py          # Ana GUI - overlay logic
src/algorithms/beamforming.py   # DAS beamformer
config/config.yaml              # Parametreler
docs/OVERLAY_FIX_GUIDE.md       # Bu dosya
```

---

## ✉️ İletişim

Sorular için: Repository issue açın veya tez danışmanınıza sorun.

**Son Güncelleme:** 7 Aralık 2025
