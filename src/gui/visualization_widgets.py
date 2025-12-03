"""
Görselleştirme Widget'ları - Spektrogram & Waveform
Real-time audio visualization için pyqtgraph tabanlı widget'lar
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class SpectrogramWidget(QWidget):
    """
    Real-time Spektrogram Widget
    - 5 saniye sliding window
    - 16 kanal ortalaması veya tek kanal seçimi
    - pyqtgraph ImageItem kullanarak hızlı güncelleme
    """
    
    def __init__(self, sample_rate=48000, window_duration=5.0):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.window_samples = int(sample_rate * window_duration)
        
        # STFT parametreleri
        self.nperseg = 2048  # FFT window size
        self.noverlap = self.nperseg // 2
        self.nfft = 2048
        
        # Spektrogram data buffer
        self.spectrogram_data = None
        self.time_axis = None
        self.freq_axis = None
        
        # Selected channel (0 = all channels average, 1-16 = specific channel)
        self.selected_channel = 0
        
        self._init_ui()
        
        logger.info("SpectrogramWidget initialized")
    
    def _init_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Kontrol paneli
        control_layout = QHBoxLayout()
        
        # Kanal seçici
        control_layout.addWidget(QLabel("Kanal:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Tüm Kanallar (Ortalama)", 0)
        for i in range(1, 17):
            self.channel_combo.addItem(f"Ch{i}", i)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        control_layout.addWidget(self.channel_combo)
        control_layout.addStretch()
        
        # Info label
        self.info_label = QLabel("Spektrogram hazır")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
        
        # PyQtGraph plot widget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.setLabel('left', 'Frekans', units='Hz')
        self.plot_widget.setLabel('bottom', 'Zaman', units='s')
        self.plot_widget.setTitle('Spektrogram', color='w', size='12pt')
        
        # ImageItem for spectrogram
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        
        # Colormap (iyileştirilmiş jet - daha kontrastlı)
        colors = [
            (0, 0, 20),      # Çok koyu mavi (neredeyse siyah)
            (0, 0, 140),     # Koyu mavi
            (0, 100, 255),   # Mavi
            (0, 200, 255),   # Açık mavi/Cyan
            (0, 255, 150),   # Yeşil-cyan
            (100, 255, 0),   # Yeşil
            (255, 255, 0),   # Sarı
            (255, 150, 0),   # Turuncu
            (255, 50, 0),    # Kırmızı-turuncu
            (255, 0, 0),     # Parlak kırmızı
            (200, 0, 0),     # Koyu kırmızı
        ]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
        self.image_item.setColorMap(cmap)
        
        # Colorbar
        self.colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap=cmap,
            label='Güç (dB)'
        )
        self.colorbar.setImageItem(self.image_item)
        self.plot_widget.addItem(self.colorbar)
        
        layout.addWidget(self.plot_widget)
    
    def _on_channel_changed(self, index):
        """Kanal seçimi değiştiğinde"""
        self.selected_channel = self.channel_combo.currentData()
        logger.debug(f"Spektrogram channel changed to: {self.selected_channel}")
        
        # Info güncelle
        if self.selected_channel == 0:
            self.info_label.setText("Tüm kanalların ortalaması gösteriliyor")
        else:
            self.info_label.setText(f"Ch{self.selected_channel} gösteriliyor")
    
    def update_data(self, audio_data: np.ndarray):
        """
        Audio verisini güncelle ve spektrogram hesapla
        
        Args:
            audio_data: (num_samples, num_channels) veya (num_samples,) şeklinde numpy array
        """
        try:
            # Veri boyutunu kontrol et
            if audio_data.ndim == 1:
                # Tek kanal
                signal_data = audio_data
            elif audio_data.ndim == 2:
                # Çok kanallı
                if self.selected_channel == 0:
                    # Tüm kanalların ortalaması
                    signal_data = np.mean(audio_data, axis=1)
                else:
                    # Seçili kanal
                    ch_idx = self.selected_channel - 1
                    if ch_idx < audio_data.shape[1]:
                        signal_data = audio_data[:, ch_idx]
                    else:
                        logger.warning(f"Channel {self.selected_channel} not available")
                        return
            else:
                logger.error(f"Unexpected audio data shape: {audio_data.shape}")
                return
            
            # Son N saniye al (sliding window)
            if len(signal_data) > self.window_samples:
                signal_data = signal_data[-self.window_samples:]
            
            # STFT hesapla
            f, t, Sxx = signal.spectrogram(
                signal_data,
                fs=self.sample_rate,
                window='hann',
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                scaling='density'
            )
            
            # dB'ye çevir
            Sxx_db = 10 * np.log10(Sxx + 1e-10)  # Avoid log(0)
            
            # Adaptive normalization - dinamik range
            # Her frame için min/max al, daha iyi görselleştirme
            percentile_low = np.percentile(Sxx_db, 10)  # Alt %10
            percentile_high = np.percentile(Sxx_db, 95)  # Üst %95
            
            vmin = max(percentile_low, -100)  # Çok düşük değerleri kes
            vmax = percentile_high
            
            # Normalize (0-1 arası)
            Sxx_normalized = np.clip((Sxx_db - vmin) / (vmax - vmin + 1e-6), 0, 1)
            
            # Transpose (pyqtgraph için: [x, y] = [time, freq])
            self.spectrogram_data = Sxx_normalized.T
            self.freq_axis = f
            self.time_axis = t
            
            # ImageItem güncelle
            self.image_item.setImage(
                self.spectrogram_data,
                autoLevels=False,
                levels=(0, 1)
            )
            
            # Eksenleri ayarla
            # Scale and position the image
            tr = pg.QtGui.QTransform()
            tr.scale(t[-1] / self.spectrogram_data.shape[0], 
                    f[-1] / self.spectrogram_data.shape[1])
            self.image_item.setTransform(tr)
            
            # View aralığını ayarla
            self.plot_widget.setXRange(0, t[-1], padding=0)
            self.plot_widget.setYRange(0, f[-1], padding=0)
            
        except Exception as e:
            logger.error(f"Spektrogram update error: {e}")
    
    def clear(self):
        """Spektrogramı temizle"""
        if self.image_item is not None:
            self.image_item.clear()


class WaveformWidget(QWidget):
    """
    Real-time FFT Spectrum Widget (Frequency Domain)
    - Anlık FFT spektrum gösterimi (hangi frekansta ne kadar ses var)
    - Dikey çubuklar ile frekans-genlik grafiği
    - Geçmişe kayma YOK - sadece o anki frekans dağılımı
    """
    
    def __init__(self, sample_rate=48000, fft_size=2048):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        # Peak hold for FFT bins
        self.peak_data = None
        self.peak_decay = 0.90  # Hızlı decay
        
        # Selected channel
        self.selected_channel = 0  # 0 = average, 1-16 = specific
        
        self._init_ui()
        
        logger.info("FFT Spectrum Widget initialized")
    
    def _init_ui(self):
        """UI bileşenlerini oluştur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Kontrol paneli
        control_layout = QHBoxLayout()
        
        # Kanal seçici
        control_layout.addWidget(QLabel("Kanal:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Tüm Kanallar (Ortalama)", 0)
        for i in range(1, 17):
            self.channel_combo.addItem(f"Ch{i}", i)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        control_layout.addWidget(self.channel_combo)
        control_layout.addStretch()
        
        # Info label
        self.info_label = QLabel("Waveform hazır")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
        
        # PyQtGraph plot widget
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')  # Beyaz background (referans görseldeki gibi)
        self.plot_widget.setLabel('left', 'Güç (dB)', color='k')
        self.plot_widget.setLabel('bottom', 'Frekans', units='Hz', color='k')
        self.plot_widget.setTitle('FFT Spektrum', color='k', size='11pt')
        self.plot_widget.setYRange(-80, 0)
        self.plot_widget.setLogMode(x=False, y=False)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # Ana spektrum çizgisi (mavi line + fill)
        self.spectrum_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(70, 130, 220), width=2),  # Mavi çizgi
            fillLevel=-80,  # Bottom'dan doldur
            brush=(70, 130, 220, 80)  # Soluk mavi fill
        )
        
        # Peak hold çizgisi (soluk mavi line)
        self.peak_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(150, 180, 230), width=1.5, style=pg.QtCore.Qt.DashLine)
        )
        
        layout.addWidget(self.plot_widget)
    
    def _on_channel_changed(self, index):
        """Kanal seçimi değiştiğinde"""
        self.selected_channel = self.channel_combo.currentData()
        logger.debug(f"Waveform channel changed to: {self.selected_channel}")
        
        # Peak data'yı sıfırla
        self.peak_data = None
        
        # Info güncelle
        if self.selected_channel == 0:
            self.info_label.setText("Tüm kanalların ortalaması gösteriliyor")
        else:
            self.info_label.setText(f"Ch{self.selected_channel} gösteriliyor")
    
    def update_data(self, audio_data: np.ndarray):
        """
        Audio verisini güncelle ve FFT spektrum çiz
        
        Args:
            audio_data: (num_samples, num_channels) veya (num_samples,) şeklinde numpy array
        """
        try:
            # Veri boyutunu kontrol et
            if audio_data.ndim == 1:
                # Tek kanal
                signal_data = audio_data
            elif audio_data.ndim == 2:
                # Çok kanallı
                if self.selected_channel == 0:
                    # Tüm kanalların ortalaması
                    signal_data = np.mean(audio_data, axis=1)
                else:
                    # Seçili kanal
                    ch_idx = self.selected_channel - 1
                    if ch_idx < audio_data.shape[1]:
                        signal_data = audio_data[:, ch_idx]
                    else:
                        logger.warning(f"Channel {self.selected_channel} not available")
                        return
            else:
                logger.error(f"Unexpected audio data shape: {audio_data.shape}")
                return
            
            # Yeterli veri yoksa çık
            if len(signal_data) < self.fft_size:
                return
            
            # Son N sample al
            signal_data = signal_data[-self.fft_size:]
            
            # Windowing (Hann window)
            window = np.hanning(len(signal_data))
            windowed_signal = signal_data * window
            
            # FFT hesapla
            fft_data = np.fft.rfft(windowed_signal)
            fft_magnitude = np.abs(fft_data)
            
            # dB'ye çevir
            fft_db = 20 * np.log10(fft_magnitude + 1e-10)  # Avoid log(0)
            
            # Frekans ekseni
            freqs = np.fft.rfftfreq(len(signal_data), 1.0 / self.sample_rate)
            
            # Sadece 20 Hz - 20 kHz arası göster
            valid_idx = (freqs >= 20) & (freqs <= 20000)
            freqs = freqs[valid_idx]
            fft_db = fft_db[valid_idx]
            
            # Smoothing (moving average) - daha düzgün çizgi
            if len(fft_db) > 10:
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                fft_db_smooth = np.convolve(fft_db, kernel, mode='same')
            else:
                fft_db_smooth = fft_db
            
            # Peak hold güncelle
            if self.peak_data is None or len(self.peak_data) != len(fft_db_smooth):
                self.peak_data = fft_db_smooth.copy()
            else:
                self.peak_data = np.maximum(fft_db_smooth, self.peak_data * self.peak_decay)
            
            # Spektrum çizgisini çiz (line plot)
            if len(freqs) > 0:
                self.spectrum_curve.setData(freqs, fft_db_smooth)
                self.peak_curve.setData(freqs, self.peak_data)
            
        except Exception as e:
            logger.error(f"FFT Spectrum update error: {e}")
    
    def clear(self):
        """FFT Spektrum'u temizle"""
        self.spectrum_curve.clear()
        self.peak_curve.clear()
        self.peak_data = None
