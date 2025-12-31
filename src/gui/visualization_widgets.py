"""
Görselleştirme Widget'ları - Spektrogram & Waveform
Real-time audio visualization için pyqtgraph tabanlı widget'lar
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt
from PySide6.QtGui import QLinearGradient, QColor, QBrush
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
    - Logaritmik X ekseni, koyu tema, gradyan dolgu
    - Crosshair ile interaktif frekans/güç gösterimi
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
        
        # Crosshair data
        self.current_freqs = None
        self.current_fft_db = None
        
        self._init_ui()
        
        logger.info("FFT Spectrum Widget initialized (Dark Theme + Logarithmic)")
    
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
        
        # Info label (crosshair değerlerini gösterir)
        self.info_label = QLabel("Frekans: - Hz | Güç: - dB")
        self.info_label.setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold;")
        control_layout.addWidget(self.info_label)
        
        layout.addLayout(control_layout)
        
        # PyQtGraph plot widget - DARK THEME
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a2e')  # Koyu mavi-siyah arkaplan
        self.plot_widget.setLabel('left', 'Güç (dB)', color='#cccccc')
        self.plot_widget.setLabel('bottom', 'Frekans', units='Hz', color='#cccccc')
        self.plot_widget.setTitle('FFT Spektrum', color='#ffffff', size='11pt')
        self.plot_widget.setYRange(-80, 0)
        
        # Logaritmik X ekseni
        self.plot_widget.setLogMode(x=True, y=False)
        self.plot_widget.setXRange(np.log10(20), np.log10(20000))  # 20 Hz - 20 kHz
        
        # Grid çizgileri - ince ve gri
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getAxis('bottom').setStyle(tickTextOffset=5)
        self.plot_widget.getAxis('left').setStyle(tickTextOffset=5)
        
        # Gradyan dolgu için brush oluştur (Cyan'dan Transparent Mavi'ye)
        gradient = QLinearGradient(0, 0, 0, 1)
        gradient.setCoordinateMode(QLinearGradient.ObjectBoundingMode)
        gradient.setColorAt(0, QColor(0, 255, 255, 180))   # Parlak Cyan (üst)
        gradient.setColorAt(0.5, QColor(0, 150, 255, 100)) # Mavi
        gradient.setColorAt(1, QColor(0, 80, 150, 30))     # Transparent Mavi (alt)
        
        # Ana spektrum eğrisi (gradyan dolgu ile)
        self.spectrum_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=2),  # Cyan çizgi
            fillLevel=-80,
            brush=QBrush(gradient)
        )
        
        # Peak hold çizgisi (açık mor, kesikli)
        self.peak_curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(180, 100, 255), width=1.5, style=Qt.DashLine)
        )
        
        # Crosshair (kılavuz çizgileri)
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#FFFF00', width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#FFFF00', width=1))
        self.plot_widget.addItem(self.vLine, ignoreBounds=True)
        self.plot_widget.addItem(self.hLine, ignoreBounds=True)
        
        # Mouse hareket takibi
        self.proxy = pg.SignalProxy(self.plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self._mouseMoved)
        
        layout.addWidget(self.plot_widget)
    
    def _mouseMoved(self, evt):
        """Mouse hareket ettiğinde crosshair güncelle"""
        pos = evt[0]
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x = mousePoint.x()  # Logaritmik değer
            y = mousePoint.y()
            
            # Crosshair pozisyonunu güncelle
            self.vLine.setPos(x)
            self.hLine.setPos(y)
            
            # Frekans değerini hesapla (logaritmik eksenden)
            freq = 10 ** x
            
            # Eğer veri varsa, o frekanstaki gerçek dB değerini bul
            if self.current_freqs is not None and self.current_fft_db is not None and len(self.current_freqs) > 0:
                # En yakın frekans noktasını bul
                idx = np.argmin(np.abs(self.current_freqs - freq))
                actual_db = self.current_fft_db[idx]
                actual_freq = self.current_freqs[idx]
                self.info_label.setText(f"Frekans: {actual_freq:.0f} Hz | Güç: {actual_db:.1f} dB")
            else:
                self.info_label.setText(f"Frekans: {freq:.0f} Hz | Güç: {y:.1f} dB")
    
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
                # Crosshair için verileri sakla
                self.current_freqs = freqs.copy()
                self.current_fft_db = fft_db_smooth.copy()
                
                self.spectrum_curve.setData(freqs, fft_db_smooth)
                self.peak_curve.setData(freqs, self.peak_data)
            
        except Exception as e:
            logger.error(f"FFT Spectrum update error: {e}")
    
    def clear(self):
        """FFT Spektrum'u temizle"""
        self.spectrum_curve.clear()
        self.peak_curve.clear()
        self.peak_data = None
        self.current_freqs = None
        self.current_fft_db = None


# =============================================================================
# SPATIAL 3D WIDGET - pyqtgraph.opengl based 3D Visualization
# =============================================================================

try:
    import pyqtgraph.opengl as gl
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    logger.warning("pyqtgraph.opengl not available - 3D visualization disabled")


class Spatial3DWidget(QWidget):
    """
    3D Spatial Visualization Widget
    - Grid floor with microphone positions
    - Glowing spheres for sound sources
    - Laser vectors from mic array center to sources
    - Orbiting camera with mouse control
    """
    
    def __init__(self, mic_positions=None):
        super().__init__()
        
        self.mic_positions = mic_positions  # Nx3 array
        self.sources = []  # List of source data
        
        self._init_ui()
        
        logger.info("Spatial3DWidget initialized")
    
    def _init_ui(self):
        """Initialize 3D view"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        if not OPENGL_AVAILABLE:
            # Fallback if OpenGL not available
            fallback_label = QLabel("3D Visualization requires OpenGL support")
            fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_label.setStyleSheet("color: #888; font-size: 12px; padding: 20px;")
            layout.addWidget(fallback_label)
            return
        
        # Create 3D view
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#0a0a14')
        self.view.setCameraPosition(distance=1.5, elevation=30, azimuth=45)
        layout.addWidget(self.view)
        
        # Add grid floor
        self._add_grid_floor()
        
        # Add microphone positions
        self._add_microphone_array()
        
        # Placeholder for source items
        self.source_items = []
        self.laser_items = []
    
    def _add_grid_floor(self):
        """Add reference grid on the floor"""
        if not OPENGL_AVAILABLE:
            return
        
        # Create grid
        grid = gl.GLGridItem()
        grid.setSize(x=2, y=2, z=0)
        grid.setSpacing(x=0.1, y=0.1, z=0.1)
        grid.translate(0, 0, -0.1)  # Slightly below origin
        grid.setColor((0.2, 0.3, 0.4, 0.5))
        self.view.addItem(grid)
        
        # Add axis lines
        axis_data = np.array([
            # X axis (red)
            [[0, 0, 0], [0.3, 0, 0]],
            # Y axis (green)
            [[0, 0, 0], [0, 0.3, 0]],
            # Z axis (blue)
            [[0, 0, 0], [0, 0, 0.3]]
        ])
        
        # X axis
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0.3, 0, 0]]), 
                                    color=(1, 0.3, 0.3, 0.8), width=2)
        self.view.addItem(x_axis)
        
        # Y axis
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0.3, 0]]), 
                                    color=(0.3, 1, 0.3, 0.8), width=2)
        self.view.addItem(y_axis)
        
        # Z axis
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 0.3]]), 
                                    color=(0.3, 0.5, 1, 0.8), width=2)
        self.view.addItem(z_axis)
    
    def _add_microphone_array(self):
        """Add microphone positions as small points"""
        if not OPENGL_AVAILABLE:
            return
        
        if self.mic_positions is None:
            # Default UMA-16 circular array (example)
            n_mics = 16
            radius = 0.04  # 4cm radius
            angles = np.linspace(0, 2 * np.pi, n_mics, endpoint=False)
            self.mic_positions = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.zeros(n_mics)
            ])
        
        # Ensure mic_positions is Nx3
        if self.mic_positions.ndim == 2:
            if self.mic_positions.shape[1] == 2:
                # Add z=0 if only 2D
                z_vals = np.zeros((self.mic_positions.shape[0], 1))
                self.mic_positions = np.hstack([self.mic_positions, z_vals])
        
        # Create microphone scatter points
        mic_colors = np.ones((len(self.mic_positions), 4))
        mic_colors[:, :3] = [0.3, 0.8, 1.0]  # Cyan
        mic_colors[:, 3] = 0.9  # Alpha
        
        self.mic_scatter = gl.GLScatterPlotItem(
            pos=self.mic_positions,
            color=mic_colors,
            size=8,
            pxMode=True
        )
        self.view.addItem(self.mic_scatter)
        
        # Draw microphone array boundary circle
        n_circle = 100
        theta = np.linspace(0, 2 * np.pi, n_circle)
        r = np.max(np.linalg.norm(self.mic_positions[:, :2], axis=1))
        circle_pts = np.column_stack([
            r * np.cos(theta),
            r * np.sin(theta),
            np.zeros(n_circle)
        ])
        
        mic_circle = gl.GLLinePlotItem(pos=circle_pts, color=(0.3, 0.6, 0.8, 0.5), width=1)
        self.view.addItem(mic_circle)
    
    def update_sources(self, sources):
        """
        Update sound source positions and intensities
        
        Args:
            sources: List of dicts with keys:
                - x, y: position in meters
                - power_db: sound level in dB
                - color: RGB tuple (0-255)
                - index: source number
        """
        if not OPENGL_AVAILABLE:
            return
        
        self.sources = sources
        
        # Remove old source items
        for item in self.source_items:
            self.view.removeItem(item)
        for item in self.laser_items:
            self.view.removeItem(item)
        
        self.source_items = []
        self.laser_items = []
        
        if not sources:
            return
        
        # Get mic array center
        mic_center = np.mean(self.mic_positions, axis=0) if self.mic_positions is not None else np.array([0, 0, 0])
        
        for source in sources:
            x = source.get('x', 0)
            y = source.get('y', 0)
            z = source.get('z', 0.5)  # Default z position (distance from array)
            power_db = source.get('power_db', 0)
            color_bgr = source.get('color', (0, 255, 0))
            index = source.get('index', 1)
            
            # Convert BGR to RGB and normalize
            color_rgb = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)
            
            # Size based on power (larger = louder)
            base_size = 15 if index == 1 else 10
            size = base_size + max(0, (power_db + 60) / 5)  # Scale by power
            
            # Alpha based on power
            alpha = min(1.0, 0.5 + (power_db + 60) / 100)
            
            # Source position
            pos = np.array([[x, y, z]])
            
            # Create glowing sphere
            source_scatter = gl.GLScatterPlotItem(
                pos=pos,
                color=(*color_rgb, alpha),
                size=size,
                pxMode=True
            )
            self.view.addItem(source_scatter)
            self.source_items.append(source_scatter)
            
            # Create outer glow (larger, more transparent)
            glow_scatter = gl.GLScatterPlotItem(
                pos=pos,
                color=(*color_rgb, alpha * 0.3),
                size=size * 2,
                pxMode=True
            )
            self.view.addItem(glow_scatter)
            self.source_items.append(glow_scatter)
            
            # Create laser line from mic center to source
            laser_pts = np.array([mic_center, [x, y, z]])
            laser_color = (*color_rgb, 0.6)
            
            laser_line = gl.GLLinePlotItem(
                pos=laser_pts,
                color=laser_color,
                width=2 if index == 1 else 1
            )
            self.view.addItem(laser_line)
            self.laser_items.append(laser_line)
    
    def set_microphone_positions(self, positions):
        """Update microphone positions"""
        self.mic_positions = positions
        
        if OPENGL_AVAILABLE and hasattr(self, 'mic_scatter'):
            # Ensure proper shape
            if positions.ndim == 2:
                if positions.shape[1] == 2:
                    z_vals = np.zeros((positions.shape[0], 1))
                    positions = np.hstack([positions, z_vals])
            
            self.mic_scatter.setData(pos=positions)
    
    def clear(self):
        """Clear all sources"""
        if not OPENGL_AVAILABLE:
            return
        
        for item in self.source_items:
            self.view.removeItem(item)
        for item in self.laser_items:
            self.view.removeItem(item)
        
        self.source_items = []
        self.laser_items = []
        self.sources = []
