"""
Ana GUI Penceresi - UMA-16 Akustik Kamera Sistemi
PyQt5 tabanlı real-time akustik görüntüleme arayüzü

Layout:
    ┌─────────────────────────────────────────────────────┐
    │  Menu Bar & Status Bar                              │
    ├──────────────┬──────────────────────────────────────┤
    │              │                                       │
    │   Control    │        Video + Acoustic Overlay      │
    │   Panel      │                                       │
    │   (Left)     │                                       │
    │              │                                       │
    │  - Connection│                                       │
    │  - Audio     │                                       │
    │  - Beamform  │                                       │
    │  - Visual    │                                       │
    │  - Record    │                                       │
    │              │                                       │
    ├──────────────┴──────────────────────────────────────┤
    │  VU Meters & Channel Status                         │
    └─────────────────────────────────────────────────────┘
"""

import sys
import logging
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QGroupBox, QSlider, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QStatusBar,
    QMenuBar, QFileDialog, QMessageBox,
    QSplitter, QFrame, QProgressBar, QTextEdit, QListWidget,
    QScrollArea, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor, QAction

import numpy as np
import cv2

from .custom_widgets import DoubleRangeSlider
from .visualization_widgets import SpectrogramWidget, WaveformWidget, Spatial3DWidget

# Audio stream thread import
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
from audio.stream_thread import AudioStreamThread

# Beamforming imports
from algorithms.beamforming import (
    load_mic_geometry,
    create_focus_grid,
    das_beamformer,
    das_beamformer_realtime,
    mvdr_beamformer_fast,
    mvdr_beamformer_realtime,
    music_beamformer,
    music_beamformer_realtime,
    power_to_db,
    normalize_power_map,
    BeamformingConfig,
    _precompute_distances
)
from scipy.ndimage import gaussian_filter

# Matplotlib for 3D visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering to image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

# sounddevice for device listing
import sounddevice as sd

logger = logging.getLogger(__name__)


class AcousticCameraGUI(QMainWindow):
    """Ana GUI sınıfı"""
    
    def __init__(self):
        super().__init__()
        
        # Window ayarları
        self.setWindowTitle("UMA-16 Akustik Kamera Sistemi v0.1")
        self.setGeometry(50, 50, 1600, 900)  # Makul boyut
        
        # Durum değişkenleri
        self.is_running = False
        self.is_recording = False
        self.audio_connected = False
        self.video_connected = False
        
        # Timer'lar
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        
        # Recording timer
        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self._update_record_time)
        self.record_start_time = None
        
        # Video capture
        self.video_capture = None
        self.frame_count = 0
        
        # VU Meters
        self.vu_meters = []
        
        # Audio stream thread
        self.audio_thread = None
        
        # Visualization widgets
        self.spectrogram_widget = None
        self.waveform_widget = None
        
        # Visualization throttle (performans için)
        self.viz_update_counter = 0
        self.viz_update_interval = 3  # Her 3 callback'te bir güncelle
        
        # Beamforming variables
        self.beamforming_enabled = False
        self.beamforming_config = None
        self.mic_positions = None
        self.grid_points = None
        self.grid_shape = None
        self.beamforming_counter = 0
        self.beamforming_interval = 1  # Her callback'te beamforming (daha hızlı güncelleme)
        self.latest_heatmap = None  # Cached heatmap for overlay
        self.detected_peak = None  # Legacy: single peak (x, y, z, power_db, grid_row, grid_col)
        self.detected_peaks = []  # Multiple peaks list
        self.cached_distances = None  # Precomputed distances for realtime beamforming
        self.max_freq_bins = 8  # Limit frequency bins for speed
        
        # Performance monitoring
        self.beamforming_times = []  # Track processing times
        
        # Recording variables
        self.video_writer = None
        self.audio_record_buffer = []
        self.spectrogram_record_buffer = []
        self.fft_record_buffer = []
        
        # Device info (for connected device names)
        self.connected_audio_device_name = ""
        self.connected_video_device_name = ""
        
        # GUI bileşenlerini oluştur
        self._init_ui()
        self._init_menubar()
        self._init_statusbar()
        
        # Initialize beamforming (load geometry, create grid)
        self._init_beamforming()
        
        logger.info("GUI başlatıldı")
    
    def _init_ui(self):
        """Ana UI bileşenlerini oluştur"""
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ========== ANA SPLITTER ==========
        # Sol panel ve merkez+sağ panel arasında
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d5c;
                border-radius: 3px;
            }
            QSplitter::handle:hover {
                background-color: #5c5c8a;
            }
        """)
        
        # Sol panel: Kontroller - scrollable
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Yatay scroll kapalı
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(350)
        left_scroll.setMaximumWidth(500)
        left_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        control_panel = self._create_control_panel()
        control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_scroll.setWidget(control_panel)
        
        # Sağ taraf için dikey splitter (merkez + sağ panel)
        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.setHandleWidth(6)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d5c;
                border-radius: 3px;
            }
        """)
        
        # Orta: Video + Overlay + Spektrogram
        center_panel = self._create_center_panel()
        
        # Sağ: Analiz panelleri + VU Meters - scrollable
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(320)
        right_scroll.setMaximumWidth(450)
        right_scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        right_panel = self._create_right_panel()
        right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_scroll.setWidget(right_panel)
        
        # Splitter'lara ekle
        right_splitter.addWidget(center_panel)
        right_splitter.addWidget(right_scroll)
        right_splitter.setStretchFactor(0, 3)  # Merkez daha geniş
        right_splitter.setStretchFactor(1, 1)
        
        self.main_splitter.addWidget(left_scroll)
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 0)  # Sol panel sabit
        self.main_splitter.setStretchFactor(1, 1)  # Sağ taraf esnek
        
        main_layout.addWidget(self.main_splitter)
    
    def _create_control_panel(self) -> QWidget:
        """Sol kontrol panelini oluştur"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. BAĞLANTI AYARLARI
        conn_group = self._create_connection_group()
        layout.addWidget(conn_group)
        
        # 2. PARAMETRELER & ALGORİTMALAR (Beamforming) - Ses Ayarlarından Önce
        params_group = self._create_parameters_algorithms_group()
        layout.addWidget(params_group)
        
        # 3. SES AYARLARI
        audio_group = self._create_audio_group()
        layout.addWidget(audio_group)
        
        # 4. OVERLAY AYARLARI (Ayrı grup)
        overlay_group = self._create_overlay_group()
        layout.addWidget(overlay_group)
        
        # 5. RENK HARİTASI (En altta)
        colormap_group = self._create_colormap_group()
        layout.addWidget(colormap_group)
        
        # 6. KAYIT & DOSYA YÜKLEME
        file_group = self._create_file_operations_group()
        layout.addWidget(file_group)
        
        # Spacer
        layout.addStretch()
        
        return panel
    
    def _create_connection_group(self) -> QGroupBox:
        """Bağlantı ayarları grubu - Sistemdeki gerçek cihazları listeler"""
        group = QGroupBox("🔌 Bağlantı Ayarları")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Mikrofon cihaz seçimi - Sistemdeki cihazları listele
        layout.addWidget(QLabel("Mikrofon:"))
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_audio_devices()
        layout.addWidget(self.audio_device_combo)
        
        # Refresh audio devices button
        refresh_audio_btn = QPushButton("🔄 Mikrofon Listesini Yenile")
        refresh_audio_btn.clicked.connect(lambda: self._populate_audio_devices())
        layout.addWidget(refresh_audio_btn)
        
        # Video cihaz seçimi - Sistemdeki kameraları listele
        layout.addWidget(QLabel("Kamera:"))
        self.video_device_combo = QComboBox()
        self.video_device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_video_devices()
        layout.addWidget(self.video_device_combo)
        
        # Refresh video devices button
        refresh_video_btn = QPushButton("🔄 Kamera Listesini Yenile")
        refresh_video_btn.clicked.connect(lambda: self._populate_video_devices())
        layout.addWidget(refresh_video_btn)
        
        # Bağlantı durumu - tek satırda özet
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_frame.setStyleSheet("background-color: #2b2b3d; border-radius: 5px; padding: 5px;")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setSpacing(2)
        status_layout.setContentsMargins(8, 6, 8, 6)
        
        self.audio_status_label = QLabel("🔴 Mikrofon: Bağlı değil")
        self.video_status_label = QLabel("🔴 Kamera: Bağlı değil")
        status_layout.addWidget(self.audio_status_label)
        status_layout.addWidget(self.video_status_label)
        layout.addWidget(status_frame)
        
        # Başlat/Durdur butonu
        self.start_stop_btn = QPushButton("▶️ BAŞLAT")
        self.start_stop_btn.setMinimumHeight(50)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_stop_btn.clicked.connect(self.toggle_start_stop)
        layout.addWidget(self.start_stop_btn)
        
        group.setLayout(layout)
        return group
    
    def _populate_audio_devices(self):
        """Sistemdeki audio cihazlarını listele"""
        self.audio_device_combo.clear()
        uma_index = -1
        try:
            # Cihaz listesini yenile (cache'i temizle)
            sd._terminate()
            sd._initialize()
            
            devices = sd.query_devices()
            combo_idx = 0
            for i, device in enumerate(devices):
                # Sadece input cihazlarını göster
                if device['max_input_channels'] > 0:
                    name = device['name']
                    channels = device['max_input_channels']
                    display_name = f"{name} ({channels}ch)"
                    self.audio_device_combo.addItem(display_name, i)
                    
                    # UMA-16'yı bul
                    if 'uma' in name.lower():
                        uma_index = combo_idx
                    combo_idx += 1
            
            # UMA-16 varsa onu seç
            if uma_index >= 0:
                self.audio_device_combo.setCurrentIndex(uma_index)
            
            logger.info(f"Mikrofon listesi güncellendi: {combo_idx} cihaz bulundu")
                
        except Exception as e:
            logger.error(f"Mikrofon cihazları listelenemedi: {e}")
            self.audio_device_combo.addItem("Cihaz bulunamadı", -1)
    
    def _populate_video_devices(self):
        """Sistemdeki video cihazlarını listele"""
        self.video_device_combo.clear()
        
        # macOS'ta kamera isimlerini almak için
        available_cameras = []
        for i in range(5):  # İlk 5 kamera index'ini kontrol et
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Backend bilgisini al
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # macOS'ta kamera ismini almak zor, index kullan
                camera_name = f"Kamera {i} ({width}x{height})"
                available_cameras.append((i, camera_name))
                cap.release()
        
        if available_cameras:
            for idx, name in available_cameras:
                self.video_device_combo.addItem(name, idx)
        else:
            self.video_device_combo.addItem("Kamera bulunamadı", -1)
    
    def _create_audio_group(self) -> QGroupBox:
        """Ses işleme ayarları grubu"""
        group = QGroupBox("🎤 Ses Ayarları")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Sample rate
        layout.addWidget(QLabel("Örnekleme Hızı:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["48000 Hz", "44100 Hz", "96000 Hz"])
        self.sample_rate_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.sample_rate_combo)
        
        # Chunk size
        layout.addWidget(QLabel("Buffer Boyutu:"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(512, 8192)
        self.chunk_size_spin.setValue(4096)
        self.chunk_size_spin.setSingleStep(512)
        self.chunk_size_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.chunk_size_spin)
        
        # Filtreleme
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(15)
        self.highpass_check = QCheckBox("Highpass")
        self.highpass_check.setChecked(True)
        self.lowpass_check = QCheckBox("Lowpass")
        self.lowpass_check.setChecked(True)
        filter_layout.addWidget(self.highpass_check)
        filter_layout.addWidget(self.lowpass_check)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Highpass cutoff
        layout.addWidget(QLabel("Highpass Cutoff (Hz):"))
        self.highpass_spin = QSpinBox()
        self.highpass_spin.setRange(10, 1000)
        self.highpass_spin.setValue(100)
        self.highpass_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.highpass_spin)
        
        # Lowpass cutoff
        layout.addWidget(QLabel("Lowpass Cutoff (Hz):"))
        self.lowpass_spin = QSpinBox()
        self.lowpass_spin.setRange(1000, 24000)
        self.lowpass_spin.setValue(10000)
        self.lowpass_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.lowpass_spin)
        
        group.setLayout(layout)
        return group
    
    def _create_parameters_algorithms_group(self) -> QGroupBox:
        """Parametreler & Algoritmalar - Beamforming ayarları"""
        group = QGroupBox("⚙️ Parametreler & Algoritmalar")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # --- GÖRSELLEŞTİRME AÇMA/KAPAMA ---
        layout.addWidget(QLabel("<b>Görselleştirme Kontrol:</b>"))
        
        viz_control_layout = QVBoxLayout()
        viz_control_layout.setSpacing(6)
        
        self.enable_spectrogram_check = QCheckBox("Spektrogram Aktif")
        self.enable_spectrogram_check.setChecked(True)
        self.enable_spectrogram_check.setToolTip("Spektrogramı aç/kapat (CPU tasarrufu)")
        viz_control_layout.addWidget(self.enable_spectrogram_check)
        
        self.enable_fft_check = QCheckBox("FFT Spektrum Aktif")
        self.enable_fft_check.setChecked(True)
        self.enable_fft_check.setToolTip("FFT spektrumunu aç/kapat (CPU tasarrufu)")
        viz_control_layout.addWidget(self.enable_fft_check)
        
        self.enable_beamforming_check = QCheckBox("Beamforming Aktif")
        self.enable_beamforming_check.setChecked(False)  # Başlangıçta kapalı
        self.enable_beamforming_check.setToolTip("Akustik görüntüleme hesaplamasını aç/kapat")
        self.enable_beamforming_check.stateChanged.connect(self._on_beamforming_toggle)
        viz_control_layout.addWidget(self.enable_beamforming_check)
        
        layout.addLayout(viz_control_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3d3d5c;")
        layout.addWidget(separator)
        
        # Algoritma seçimi
        layout.addWidget(QLabel("Algoritma:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "DAS (Delay-and-Sum)",
            "MVDR (Minimum Variance)",
            "MUSIC",
            "CLEAN-SC"
        ])
        self.algorithm_combo.currentTextChanged.connect(self._on_algorithm_changed)
        self.algorithm_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.algorithm_combo)
        
        # Number of sources (for MUSIC algorithm)
        layout.addWidget(QLabel("Kaynak Sayısı (MUSIC için):"))
        self.n_sources_spin = QSpinBox()
        self.n_sources_spin.setRange(1, 10)
        self.n_sources_spin.setValue(1)
        self.n_sources_spin.setToolTip("MUSIC algoritması için beklenen kaynak sayısı")
        self.n_sources_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.n_sources_spin)
        
        # Frekans aralığı - DOUBLE RANGE SLIDER
        layout.addWidget(QLabel("Frekans Aralığı (Hz):"))
        self.freq_range_slider = DoubleRangeSlider(100, 20000)
        self.freq_range_slider.setValues(500, 8000)
        self.freq_range_slider.rangeChanged.connect(self.on_freq_range_changed)
        layout.addWidget(self.freq_range_slider)
        
        # Ses Şiddeti Aralığı (dB) - DOUBLE RANGE SLIDER
        layout.addWidget(QLabel("Ses Şiddeti Aralığı (dB):"))
        self.db_range_slider = DoubleRangeSlider(-60, 0)
        self.db_range_slider.setValues(-40, -10)
        self.db_range_slider.rangeChanged.connect(self.on_db_range_changed)
        layout.addWidget(self.db_range_slider)
        
        # Grid çözünürlüğü
        layout.addWidget(QLabel("Grid Çözünürlüğü (cm):"))
        self.grid_resolution_spin = QDoubleSpinBox()
        self.grid_resolution_spin.setRange(1, 20)
        self.grid_resolution_spin.setValue(5)
        self.grid_resolution_spin.setSingleStep(0.5)
        self.grid_resolution_spin.setDecimals(1)
        self.grid_resolution_spin.setSuffix(" cm")
        self.grid_resolution_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.grid_resolution_spin)
        
        # Z mesafesi (odak mesafesi)
        layout.addWidget(QLabel("Odak Mesafesi (m):"))
        self.focus_distance_spin = QDoubleSpinBox()
        self.focus_distance_spin.setRange(0.3, 5.0)
        self.focus_distance_spin.setValue(1.0)
        self.focus_distance_spin.setSingleStep(0.1)
        self.focus_distance_spin.setDecimals(2)
        self.focus_distance_spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.focus_distance_spin)
        
        # Görselleştirme seçenekleri
        layout.addWidget(QLabel("<b>Gösterim Seçenekleri:</b>"))
        self.show_contours_check = QCheckBox("Kontur çizgileri")
        self.show_peaks_check = QCheckBox("Peak noktaları")
        self.show_peaks_check.setChecked(True)
        self.show_grid_check = QCheckBox("Grid göster")
        layout.addWidget(self.show_contours_check)
        layout.addWidget(self.show_peaks_check)
        layout.addWidget(self.show_grid_check)
        
        group.setLayout(layout)
        return group
    
    def _create_overlay_group(self) -> QGroupBox:
        """Overlay ayarları grubu - ayrı"""
        group = QGroupBox("🎨 Overlay Ayarları")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # Overlay aktif/deaktif
        self.enable_overlay_check = QCheckBox("Video Overlay Aktif")
        self.enable_overlay_check.setChecked(True)
        self.enable_overlay_check.setToolTip("Akustik ısı haritasını video üzerine bindirmeyi aç/kapat")
        layout.addWidget(self.enable_overlay_check)
        
        # Overlay alpha
        layout.addWidget(QLabel("Overlay Şeffaflığı:"))
        alpha_layout = QHBoxLayout()
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)
        self.alpha_value_label = QLabel("60%")
        self.alpha_value_label.setMinimumWidth(40)
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_label.setText(f"{v}%")
        )
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_value_label)
        layout.addLayout(alpha_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_colormap_group(self) -> QGroupBox:
        """Renk haritası grubu - en altta"""
        group = QGroupBox("🌈 Renk Haritası")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        layout.addWidget(QLabel("Renk Paleti:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "jet", "hot", "viridis", "plasma", "inferno",
            "coolwarm", "rainbow", "turbo"
        ])
        self.colormap_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.colormap_combo)
        
        group.setLayout(layout)
        return group
    
    def _create_file_operations_group(self) -> QGroupBox:
        """Dosya işlemleri: Kayıt + Yükleme"""
        group = QGroupBox("💾 Dosya İşlemleri")
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 15, 10, 10)
        
        # --- DOSYA YÜKLEME ---
        layout.addWidget(QLabel("<b>Dosya Yükle:</b>"))
        
        # Ses dosyası yükle
        self.load_audio_btn = QPushButton("📂 Ses Dosyası Yükle (.wav)")
        self.load_audio_btn.clicked.connect(self.load_audio_file)
        layout.addWidget(self.load_audio_btn)
        
        # Video dosyası yükle
        self.load_video_btn = QPushButton("📂 Video Yükle (.mp4)")
        self.load_video_btn.clicked.connect(self.load_video_file)
        layout.addWidget(self.load_video_btn)
        
        # Yüklü dosya bilgisi
        self.loaded_file_label = QLabel("Yüklü dosya: -")
        self.loaded_file_label.setWordWrap(True)
        self.loaded_file_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self.loaded_file_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3d3d5c;")
        layout.addWidget(separator)
        
        # --- KAYIT ---
        layout.addWidget(QLabel("<b>Kayıt:</b>"))
        
        # Kayıt butonu
        self.record_btn = QPushButton("🔴 KAYIT BAŞLAT")
        self.record_btn.setMinimumHeight(45)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        
        # Snapshot butonu
        self.snapshot_btn = QPushButton("📸 Ekran Görüntüsü Al")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        layout.addWidget(self.snapshot_btn)
        
        # Kayıt seçenekleri
        layout.addWidget(QLabel("Kayıt Seçenekleri:"))
        self.record_audio_check = QCheckBox("Ses kaydet (.wav)")
        self.record_audio_check.setChecked(True)
        self.record_video_check = QCheckBox("Video + Overlay kaydet (.mp4)")
        self.record_video_check.setChecked(True)
        self.record_viz_check = QCheckBox("Spektrogram + FFT dahil et")
        self.record_viz_check.setChecked(True)
        self.record_data_check = QCheckBox("Ham veri kaydet (.h5)")
        self.record_data_check.setChecked(False)
        
        layout.addWidget(self.record_audio_check)
        layout.addWidget(self.record_video_check)
        layout.addWidget(self.record_viz_check)
        layout.addWidget(self.record_data_check)
        
        # Kayıt süresi ve dosya bilgisi
        record_info_frame = QFrame()
        record_info_frame.setFrameShape(QFrame.StyledPanel)
        record_info_frame.setStyleSheet("background-color: #2b2b3d; border-radius: 5px;")
        record_info_layout = QVBoxLayout(record_info_frame)
        record_info_layout.setSpacing(4)
        record_info_layout.setContentsMargins(8, 8, 8, 8)
        
        self.record_time_label = QLabel("⏱ Kayıt Süresi: 00:00:00")
        self.record_time_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        self.record_file_label = QLabel("📁 Kayıt dosyası: -")
        self.record_file_label.setWordWrap(True)
        self.record_file_label.setStyleSheet("font-size: 10px; color: #888;")
        
        record_info_layout.addWidget(self.record_time_label)
        record_info_layout.addWidget(self.record_file_label)
        layout.addWidget(record_info_frame)
        
        group.setLayout(layout)
        return group
    
    def _create_center_panel(self) -> QWidget:
        """Orta panel: Video + Overlay + Spektrogram + Waveform"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Video display + Acoustic Overlay
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)  # Aspect ratio korunması için False
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #555; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self._set_placeholder_image()
        layout.addWidget(self.video_label)
        
        # Alt kısım: Spektrogram + Waveform
        analysis_splitter = QSplitter(Qt.Horizontal)
        
        # Spektrogram - Real-time widget
        spectrogram_group = QGroupBox("📊 Spektrogram (Frekans-Zaman)")
        spec_layout = QVBoxLayout()
        spec_layout.setContentsMargins(2, 2, 2, 2)
        self.spectrogram_widget = SpectrogramWidget(sample_rate=48000, window_duration=5.0)
        self.spectrogram_widget.setMinimumSize(400, 200)
        spec_layout.addWidget(self.spectrogram_widget)
        spectrogram_group.setLayout(spec_layout)
        
        # Waveform - Real-time widget
        waveform_group = QGroupBox("📈 FFT Spektrum (Frekans-Genlik)")
        wave_layout = QVBoxLayout()
        wave_layout.setContentsMargins(2, 2, 2, 2)
        self.waveform_widget = WaveformWidget(sample_rate=48000, fft_size=2048)
        self.waveform_widget.setMinimumSize(400, 200)
        wave_layout.addWidget(self.waveform_widget)
        waveform_group.setLayout(wave_layout)
        
        analysis_splitter.addWidget(spectrogram_group)
        analysis_splitter.addWidget(waveform_group)
        
        layout.addWidget(analysis_splitter)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Sağ panel: 3D Uzamsal Konum + Tespit Edilen Kaynaklar + VU Meters (en altta)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 3D Uzamsal Konum (en üstte) - pyqtgraph.opengl ile
        spatial_group = QGroupBox("3D Uzamsal Konum")
        spatial_layout = QVBoxLayout()
        spatial_layout.setContentsMargins(2, 2, 2, 2)
        
        # Spatial3DWidget kullan
        self.spatial_3d_widget = Spatial3DWidget()
        self.spatial_3d_widget.setMinimumSize(280, 280)
        spatial_layout.addWidget(self.spatial_3d_widget)
        spatial_group.setLayout(spatial_layout)
        layout.addWidget(spatial_group)
        
        # Tespit Edilen Ses Kaynakları - Dinamik Widget
        sources_group = QGroupBox("Tespit Edilen Kaynaklar")
        sources_layout = QVBoxLayout()
        
        # Dinamik kaynak container
        self.sources_container = QWidget()
        self.sources_container_layout = QVBoxLayout(self.sources_container)
        self.sources_container_layout.setSpacing(8)
        self.sources_container_layout.setContentsMargins(5, 5, 5, 5)
        
        # Başlangıçta "Kaynak algılanamadı" mesajı
        self.no_source_label = QLabel("Kaynak Algılanamadı")
        self.no_source_label.setAlignment(Qt.AlignCenter)
        self.no_source_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 12px;
                padding: 20px;
                background-color: #2b2b2b;
                border: 1px dashed #555;
            }
        """)
        self.sources_container_layout.addWidget(self.no_source_label)
        
        sources_layout.addWidget(self.sources_container)
        sources_group.setLayout(sources_layout)
        layout.addWidget(sources_group)
        
        # Kaynak widget'ları için liste
        self.source_widgets = []
        
        # VU Meters (en altta)
        vu_meter_group = self._create_vu_meter_group()
        layout.addWidget(vu_meter_group)
        
        return panel
    
    def _create_vu_meter_group(self) -> QGroupBox:
        """VU meter göstergeleri - 4x4 grid (UMA-16v2 fiziksel geometrisine uygun)"""
        group = QGroupBox("Mikrofon Array (16 Ch)")
        layout = QGridLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # UMA-16v2 fiziksel mikrofo düzeni (micgeom.xml'den)
        # TOP ROW:    Ch9  Ch10 Ch7  Ch8
        # ROW 2:      Ch11 Ch12 Ch5  Ch6
        # ROW 3:      Ch13 Ch14 Ch3  Ch4
        # BOTTOM ROW: Ch15 Ch16 Ch1  Ch2
        mic_layout = [
            [9, 10, 7, 8],      # Top row
            [11, 12, 5, 6],     # Second row
            [13, 14, 3, 4],     # Third row
            [15, 16, 1, 2]      # Bottom row
        ]
        
        # VU meter'ları channel sayısı kadar oluştur (indeksleme için)
        self.vu_meters = [None] * 16  # Ch1-Ch16 için placeholder
        
        for row in range(4):
            for col in range(4):
                ch_num = mic_layout[row][col]  # Gerçek channel numarası
                
                # Vertical layout for each channel
                ch_container = QWidget()
                ch_layout = QVBoxLayout(ch_container)
                ch_layout.setSpacing(3)
                ch_layout.setContentsMargins(0, 0, 0, 0)
                
                # Channel label
                ch_label = QLabel(f"Ch{ch_num}")
                ch_label.setAlignment(Qt.AlignCenter)
                ch_label.setStyleSheet("""
                    font-size: 10px;
                    font-weight: bold;
                    color: #aaa;
                """)
                
                # Vertical progress bar (daha uzun ve ince)
                progress = QProgressBar()
                progress.setOrientation(Qt.Vertical)
                progress.setMaximum(100)
                progress.setValue(0)
                progress.setTextVisible(False)
                progress.setMinimumHeight(80)  # Daha uzun
                progress.setMaximumHeight(100)
                progress.setMaximumWidth(25)   # Daha ince
                progress.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #555;
                        border-radius: 3px;
                        background-color: #1e1e1e;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:1, x2:0, y2:0,
                            stop:0 #27ae60, stop:0.5 #f1c40f, stop:1 #e74c3c);
                        border-radius: 2px;
                    }
                """)
                
                ch_layout.addWidget(progress)
                ch_layout.addWidget(ch_label)
                
                layout.addWidget(ch_container, row, col)
                # Channel numarasına göre doğru index'e yerleştir (Ch1=index 0, Ch16=index 15)
                self.vu_meters[ch_num - 1] = progress
        
        group.setLayout(layout)
        return group
    
    def _init_menubar(self):
        """Menu bar oluştur"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("Dosya")
        
        load_config = QAction("Ayar Yükle...", self)
        load_config.triggered.connect(self.load_config)
        file_menu.addAction(load_config)
        
        save_config = QAction("Ayar Kaydet...", self)
        save_config.triggered.connect(self.save_config)
        file_menu.addAction(save_config)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Çıkış", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Araçlar")
        
        calibration = QAction("Kalibrasyon...", self)
        tools_menu.addAction(calibration)
        
        geometry_viewer = QAction("Geometri Görüntüle", self)
        geometry_viewer.triggered.connect(self.show_geometry)
        tools_menu.addAction(geometry_viewer)
        
        # Help menu
        help_menu = menubar.addMenu("Yardım")
        
        about = QAction("Hakkında", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)
    
    def _init_statusbar(self):
        """Status bar oluştur"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # FPS göstergesi
        self.fps_label = QLabel("FPS: 0")
        self.statusbar.addPermanentWidget(self.fps_label)
        
        # CPU kullanımı
        self.cpu_label = QLabel("CPU: 0%")
        self.statusbar.addPermanentWidget(self.cpu_label)
        
        # Durum mesajı
        self.statusbar.showMessage("Hazır")
    
    def _set_placeholder_image(self):
        """Video için placeholder görüntü"""
        placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Metin ekle
        text = "Video Bağlantısı Bekleniyor..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = (800 - text_size[0]) // 2
        text_y = (600 + text_size[1]) // 2
        
        cv2.putText(placeholder, text, (text_x, text_y), 
                   font, 1, (100, 100, 100), 2)
        
        self._display_image(placeholder)
    
    def _display_image(self, image: np.ndarray):
        """Numpy array'i QLabel'da göster - aspect ratio korunur"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        
        # BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(rgb_image.data, width, height, 
                        bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Aspect ratio korunarak ölçeklendir
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_detected_sources(self, sources: list):
        """
        Tespit edilen ses kaynaklarını güncelle - Modern card layout
        
        Args:
            sources: [(x, y, z, db_level), ...] formatında kaynak listesi
        """
        # Önce mevcut widget'ları temizle
        for widget in self.source_widgets:
            widget.deleteLater()
        self.source_widgets.clear()
        
        # "Kaynak algılanamadı" label'ını gizle/göster
        if len(sources) == 0:
            self.no_source_label.show()
        else:
            self.no_source_label.hide()
            
            # Her kaynak için widget oluştur
            for idx, (x, y, z, db) in enumerate(sources, 1):
                source_widget = self._create_source_card(idx, x, y, z, db)
                self.sources_container_layout.addWidget(source_widget)
                self.source_widgets.append(source_widget)
    
    def _create_source_card(self, idx: int, x: float, y: float, z: float, db: float) -> QWidget:
        """Modern card-style source widget with LED bars and compass"""
        import math
        
        widget = QWidget()
        widget.setObjectName(f"source_card_{idx}")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        
        # Color coding based on source index and level
        source_colors = [
            ("#00ff88", "#00cc66"),  # Green (primary)
            ("#ff6b35", "#cc5529"),  # Orange
            ("#00bfff", "#0099cc"),  # Cyan
            ("#ff66ff", "#cc52cc"),  # Magenta
            ("#ffff00", "#cccc00"),  # Yellow
        ]
        primary_color, secondary_color = source_colors[(idx - 1) % len(source_colors)]
        
        # Level indicator - Türkçe
        if db > -20:
            level_text = "Yüksek"
            level_color = "#ff4444"  # Kırmızı
        elif db > -35:
            level_text = "Orta"
            level_color = "#ffaa00"  # Turuncu
        else:
            level_text = "Düşük"
            level_color = "#44cc44"  # Yeşil
        
        # === Header Row ===
        header_layout = QHBoxLayout()
        
        # Source number with accent - Türkçe
        source_num = QLabel(f"{idx}. Ses Kaynağı")
        source_num.setStyleSheet(f"""
            color: {primary_color}; 
            font-weight: bold; 
            font-size: 13px;
            background-color: rgba(0, 255, 136, 0.1);
            padding: 2px 8px;
            border-radius: 3px;
            border-left: 3px solid {primary_color};
        """)
        header_layout.addWidget(source_num)
        
        header_layout.addStretch()
        
        # Level indicator - Türkçe
        level_label = QLabel(f"{level_text}")
        level_label.setStyleSheet(f"""
            color: {level_color}; 
            font-size: 11px;
            font-weight: bold;
            padding: 2px 6px;
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 3px;
        """)
        header_layout.addWidget(level_label)
        
        layout.addLayout(header_layout)
        
        # === Separator Line ===
        separator = QWidget()
        separator.setFixedHeight(1)
        separator.setStyleSheet(f"background-color: {secondary_color}; opacity: 0.5;")
        layout.addWidget(separator)
        
        # === Power Reading with LED Bar ===
        power_layout = QHBoxLayout()
        power_layout.setSpacing(8)
        
        power_icon = QLabel("⚡")
        power_icon.setStyleSheet("font-size: 12px;")
        power_layout.addWidget(power_icon)
        
        power_value = QLabel(f"{db:+.1f} dB")
        power_value.setStyleSheet(f"""
            color: {primary_color}; 
            font-family: 'Consolas', 'Monaco', monospace;
            font-weight: bold; 
            font-size: 13px;
            min-width: 70px;
        """)
        power_layout.addWidget(power_value)
        
        # LED Bar (custom styled progress)
        led_bar = QProgressBar()
        led_bar.setMinimum(-60)
        led_bar.setMaximum(0)
        led_bar.setValue(int(max(-60, min(0, db))))
        led_bar.setTextVisible(False)
        led_bar.setFixedHeight(8)
        led_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 2px;
                background-color: #1a1a2e;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {secondary_color}, 
                    stop:0.7 {primary_color},
                    stop:1 white);
                border-radius: 2px;
            }}
        """)
        power_layout.addWidget(led_bar, stretch=1)
        
        layout.addLayout(power_layout)
        
        # === Position with Compass ===
        pos_layout = QHBoxLayout()
        pos_layout.setSpacing(8)
        
        # Calculate direction
        azimuth_rad = math.atan2(x, z if z != 0 else 0.001)
        azimuth_deg = math.degrees(azimuth_rad)
        elevation_rad = math.atan2(y, math.sqrt(x*x + z*z) if (x*x + z*z) > 0 else 0.001)
        elevation_deg = math.degrees(elevation_rad)
        
        # Compass direction
        if abs(azimuth_deg) < 15:
            compass = "N"
            compass_arrow = "↑"
        elif azimuth_deg >= 15 and azimuth_deg < 75:
            compass = "NE"
            compass_arrow = "↗"
        elif azimuth_deg >= 75:
            compass = "E"
            compass_arrow = "→"
        elif azimuth_deg <= -15 and azimuth_deg > -75:
            compass = "NW"
            compass_arrow = "↖"
        else:
            compass = "W"
            compass_arrow = "←"
        
        # Compass widget
        compass_label = QLabel(f"{compass_arrow}")
        compass_label.setStyleSheet(f"""
            color: {primary_color};
            font-size: 16px;
            font-weight: bold;
            background-color: rgba(0, 0, 0, 0.3);
            border: 1px solid {secondary_color};
            border-radius: 12px;
            padding: 2px 6px;
            min-width: 24px;
        """)
        compass_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pos_layout.addWidget(compass_label)
        
        # Position text
        pos_text = QLabel(f"X:{x*100:+5.0f}  Y:{y*100:+5.0f}  Z:{z*100:5.0f}")
        pos_text.setStyleSheet("""
            color: #aabbcc; 
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10px;
        """)
        pos_layout.addWidget(pos_text)
        
        pos_layout.addStretch()
        
        # Angle display
        angle_text = QLabel(f"∠{azimuth_deg:+4.0f}°")
        angle_text.setStyleSheet(f"""
            color: {primary_color}; 
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 10px;
        """)
        pos_layout.addWidget(angle_text)
        
        layout.addLayout(pos_layout)
        
        # === Card Styling ===
        widget.setStyleSheet(f"""
            QWidget#source_card_{idx} {{
                background-color: #16162a;
                border: 1px solid {secondary_color};
                border-radius: 8px;
                border-left: 4px solid {primary_color};
            }}
            QWidget#source_card_{idx}:hover {{
                background-color: #1e1e3a;
                border-color: {primary_color};
            }}
        """)
        
        return widget
    
    def _create_source_widget(self, idx: int, x: float, y: float, z: float, db: float) -> QWidget:
        """Legacy wrapper - redirects to new card style"""
        return self._create_source_card(idx, x, y, z, db)
    
    def _update_sources_panel(self):
        """
        Update the detected sources panel with current peak information
        Converts detected_peaks to the format expected by update_detected_sources
        """
        try:
            if not hasattr(self, 'detected_peaks') or len(self.detected_peaks) == 0:
                # No sources detected
                self.update_detected_sources([])
                return
            
            # Convert detected_peaks to (x, y, z, db) format
            sources = []
            for peak in self.detected_peaks:
                x = peak.get('x', 0)
                y = peak.get('y', 0)
                z = peak.get('z', self.focus_distance_spin.value())
                db = peak.get('power_db', -60)
                sources.append((x, y, z, db))
            
            # Update the panel
            self.update_detected_sources(sources)
            
        except Exception as e:
            logger.error(f"Sources panel update error: {e}")
    
    def _update_3d_visualization(self):
        """
        Update 3D spatial visualization with detected sources
        Uses pyqtgraph.opengl Spatial3DWidget for real-time rendering
        """
        try:
            # Check if widget exists
            if not hasattr(self, 'spatial_3d_widget') or self.spatial_3d_widget is None:
                return
            
            # Update microphone positions if available
            if hasattr(self, 'mic_positions') and self.mic_positions is not None:
                self.spatial_3d_widget.set_microphone_positions(self.mic_positions)
            
            # Prepare source data for 3D widget
            sources = []
            if hasattr(self, 'detected_peaks') and len(self.detected_peaks) > 0:
                focus_z = self.focus_distance_spin.value() if hasattr(self, 'focus_distance_spin') else 1.0
                
                for peak in self.detected_peaks:
                    source_data = {
                        'x': peak.get('x', 0),
                        'y': peak.get('y', 0),
                        'z': focus_z,  # Z = focus distance
                        'power_db': peak.get('power_db', 0),
                        'color': peak.get('color', (0, 255, 0)),
                        'index': peak.get('index', 1)
                    }
                    sources.append(source_data)
            
            # Update 3D widget with sources
            self.spatial_3d_widget.update_sources(sources)
            
        except Exception as e:
            logger.error(f"3D visualization error: {e}")
    
    # Slot fonksiyonları
    def toggle_start_stop(self):
        """Başlat/Durdur toggle"""
        if not self.is_running:
            self.start_system()
        else:
            self.stop_system()
    
    def start_system(self):
        """Sistemi başlat"""
        logger.info("Sistem başlatılıyor...")
        self.is_running = True
        
        # Seçili video cihazını al
        video_device_id = self.video_device_combo.currentData()
        if video_device_id is None or video_device_id < 0:
            video_device_id = 0  # Default
        
        video_device_text = self.video_device_combo.currentText()
        logger.info(f"Video cihaz açılıyor: {video_device_text} (ID: {video_device_id})")
        
        # Webcam'i aç
        self.video_capture = cv2.VideoCapture(video_device_id)
        
        if not self.video_capture.isOpened():
            logger.error(f"Webcam açılamadı! (ID: {video_device_id})")
            QMessageBox.warning(self, "Webcam Hatası", 
                               f"Video cihazı açılamadı! (ID: {video_device_id})\nLütfen kamera bağlantısını kontrol edin.")
            self.video_status_label.setText("🔴 Kamera: Hata")
            self.video_capture = None
        else:
            # Kamera bilgilerini al
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            backend = self.video_capture.getBackendName()
            
            self.connected_video_device_name = f"{video_device_text}"
            self.video_status_label.setText(f"🟢 Kamera: {width}x{height}@{fps}fps")
            logger.info(f"Webcam başarıyla açıldı: {width}x{height}@{fps}fps")
        
        # Audio stream'i başlat
        self._start_audio_stream()
        
        # UI güncellemeleri
        self.start_stop_btn.setText("⏸️ DURDUR")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        # Timer başlat
        self.update_timer.start(33)  # ~30 FPS
        
        self.statusbar.showMessage("Sistem çalışıyor")
        
    def stop_system(self):
        """Sistemi durdur"""
        logger.info("Sistem durduruluyor...")
        self.is_running = False
        
        # Kayıt devam ediyorsa durdur
        if self.is_recording:
            self.stop_recording()
        
        # Timer durdur
        self.update_timer.stop()
        
        # Audio stream'i durdur
        self._stop_audio_stream()
        
        # Webcam'i kapat
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            logger.info("Webcam kapatıldı")
        
        # UI güncellemeleri
        self.start_stop_btn.setText("▶️ BAŞLAT")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border-radius: 5px;
            }
        """)
        
        self.audio_status_label.setText("🔴 Mikrofon: Bağlı değil")
        self.video_status_label.setText("🔴 Kamera: Bağlı değil")
        self.connected_audio_device_name = ""
        self.connected_video_device_name = ""
        
        self.statusbar.showMessage("Sistem durduruldu")
        self._set_placeholder_image()
    
    def _start_audio_stream(self):
        """Audio stream thread'ini başlat"""
        try:
            # Seçili audio cihazını al
            device_index = self.audio_device_combo.currentData()
            device_name = self.audio_device_combo.currentText()
            
            if device_index is None or device_index < 0:
                # UMA-16'yı bul
                from audio.device_test import find_uma16_device
                device_index = find_uma16_device()
                
                if device_index is None:
                    logger.error("UMA-16 cihazı bulunamadı!")
                    QMessageBox.warning(self, "Audio Hatası", 
                                       "UMA-16 cihazı bulunamadı!\nLütfen cihaz bağlantısını kontrol edin.")
                    self.audio_status_label.setText("🔴 Mikrofon: Cihaz bulunamadı")
                    return
            
            logger.info(f"Audio cihaz açılıyor: {device_name} (ID: {device_index})")
            
            # Cihaz bilgilerini al
            device_info = sd.query_devices(device_index)
            num_channels = min(16, device_info['max_input_channels'])
            
            # Audio thread oluştur
            self.audio_thread = AudioStreamThread(
                device_index=device_index,
                sample_rate=48000,
                buffer_size=4096,
                num_channels=num_channels,
                buffer_duration=5.0,
                gain=10.0  # Digital gain (10x güçlendirme)
            )
            
            # Sinyalleri bağla
            self.audio_thread.channelLevelsReady.connect(self._update_vu_meters)
            self.audio_thread.audioDataReady.connect(self._update_visualizations)
            self.audio_thread.errorOccurred.connect(self._on_audio_error)
            
            # Thread'i başlat
            self.audio_thread.start()
            
            # Bağlı cihaz ismini kaydet - kısa isim oluştur
            self.connected_audio_device_name = device_info['name']
            # İsmi kısalt (ilk 20 karakter)
            short_name = self.connected_audio_device_name[:25] + "..." if len(self.connected_audio_device_name) > 25 else self.connected_audio_device_name
            self.audio_status_label.setText(f"🟢 Mikrofon: {num_channels}ch @ 48kHz")
            logger.info(f"Audio stream başlatıldı: {self.connected_audio_device_name}")
            
        except Exception as e:
            error_msg = f"Audio stream başlatılamadı: {str(e)}"
            logger.error(error_msg)
            QMessageBox.warning(self, "Audio Hatası", error_msg)
            self.audio_status_label.setText("🔴 Mikrofon: Hata")
    
    def _stop_audio_stream(self):
        """Audio stream thread'ini durdur"""
        if self.audio_thread is not None:
            self.audio_thread.stop()
            self.audio_thread.wait()  # Thread'in bitmesini bekle
            self.audio_thread = None
            logger.info("Audio stream durduruldu")
            
            # VU meter'ları sıfırla
            for meter in self.vu_meters:
                meter.setValue(0)
            
            # Visualization widget'ları temizle
            if self.spectrogram_widget is not None:
                self.spectrogram_widget.clear()
            if self.waveform_widget is not None:
                self.waveform_widget.clear()
    
    def _update_vu_meters(self, levels: list):
        """
        VU meter'ları güncelle
        
        Args:
            levels: 0-1 arası normalize RMS değerleri (16 kanal)
        """
        for i, level in enumerate(levels):
            if i < len(self.vu_meters):
                # 0-1 değerini 0-100'e ölçekle
                value = int(level * 100)
                self.vu_meters[i].setValue(value)
    
    def _on_audio_error(self, error_msg: str):
        """Audio stream hata mesajı"""
        logger.error(f"Audio error: {error_msg}")
        self.audio_status_label.setText("🔴 Mikrofon: Hata")
    
    def _update_visualizations(self, audio_data: np.ndarray, sample_rate: int):
        """
        Visualization widget'ları güncelle (throttled, checkbox kontrolü ile)
        
        Args:
            audio_data: (num_samples, num_channels) numpy array - yeni gelen chunk
            sample_rate: Örnekleme hızı
        """
        try:
            # Kayıt için audio buffer'a ekle
            if self.is_recording and self.record_audio_check.isChecked():
                self.audio_record_buffer.append(audio_data.copy())
            
            # Throttle - her N callback'te bir güncelle (performans)
            self.viz_update_counter += 1
            if self.viz_update_counter < self.viz_update_interval:
                return
            self.viz_update_counter = 0
            
            # Spektrogram güncelle (checkbox aktif ise)
            if (self.enable_spectrogram_check.isChecked() and 
                self.spectrogram_widget is not None and 
                self.audio_thread is not None):
                buffer_data = self.audio_thread.get_buffer_data(duration=5.0)
                if buffer_data is not None and len(buffer_data) > 0:
                    self.spectrogram_widget.update_data(buffer_data)
            
            # FFT Spektrum güncelle (checkbox aktif ise)
            if (self.enable_fft_check.isChecked() and 
                self.waveform_widget is not None and 
                self.audio_thread is not None):
                buffer_data = self.audio_thread.get_buffer_data(duration=0.2)  # 0.2 saniye yeterli FFT için
                if buffer_data is not None and len(buffer_data) > 0:
                    self.waveform_widget.update_data(buffer_data)
            
            # Beamforming güncelle (throttled - daha az sıklıkta)
            self.beamforming_counter += 1
            if (self.beamforming_enabled and 
                self.beamforming_counter >= self.beamforming_interval and
                self.audio_thread is not None):
                self.beamforming_counter = 0
                
                # Get buffer for beamforming (0.1 second = ~4800 samples @ 48kHz)
                buffer_data = self.audio_thread.get_buffer_data(duration=0.1)
                if buffer_data is not None and len(buffer_data) > 0:
                    self._run_beamforming(buffer_data, sample_rate)
        
        except Exception as e:
            logger.error(f"Visualization update error: {e}")
    
    def toggle_recording(self):
        """Kayıt toggle"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Kaydı başlat - Video + Ses + Görselleştirmeler"""
        if not self.is_running:
            QMessageBox.warning(self, "Kayıt Hatası", 
                               "Önce sistemi başlatmanız gerekiyor!")
            return
        
        logger.info("Kayıt başlatılıyor...")
        
        # Kayıt dizini oluştur
        records_dir = Path(__file__).parent.parent.parent / 'data' / 'recordings'
        records_dir.mkdir(parents=True, exist_ok=True)
        
        # Zaman damgası ile dosya adı
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_base_name = f"recording_{timestamp}"
        self.record_dir = records_dir / self.record_base_name
        self.record_dir.mkdir(exist_ok=True)
        
        # Video writer başlat (overlay dahil)
        if self.record_video_check.isChecked() and self.video_capture is not None:
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Eğer görselleştirmeler dahil edilecekse, daha geniş bir frame oluştur
            if self.record_viz_check.isChecked():
                # Video + Spektrogram + FFT yan yana
                total_width = width
                total_height = height + 200  # Alt kısma görselleştirmeler
            else:
                total_width = width
                total_height = height
            
            video_path = str(self.record_dir / f"{self.record_base_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (total_width, total_height))
            logger.info(f"Video kayıt başladı: {video_path}")
        
        # Audio buffer'ı temizle
        self.audio_record_buffer = []
        
        # Kayıt başlangıç zamanı
        self.record_start_time = time.time()
        self.is_recording = True
        
        # Timer başlat
        self.record_timer.start(1000)  # Her saniye güncelle
        
        # UI güncelle
        self.record_btn.setText("⏹️ KAYIT DURDUR")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                border: 2px solid #e74c3c;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.record_file_label.setText(f"📁 Kayıt: {self.record_base_name}")
        self.statusbar.showMessage("🔴 Kayıt yapılıyor...")
    
    def stop_recording(self):
        """Kaydı durdur ve dosyaları kaydet"""
        if not self.is_recording:
            return
        
        logger.info("Kayıt durduruluyor...")
        self.is_recording = False
        self.record_timer.stop()
        
        # Video writer'ı kapat
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.info("Video kayıt tamamlandı")
        
        # Audio kaydet
        if self.record_audio_check.isChecked() and len(self.audio_record_buffer) > 0:
            try:
                import soundfile as sf
                audio_data = np.concatenate(self.audio_record_buffer, axis=0)
                audio_path = str(self.record_dir / f"{self.record_base_name}.wav")
                sf.write(audio_path, audio_data, 48000)
                logger.info(f"Audio kayıt tamamlandı: {audio_path}")
            except Exception as e:
                logger.error(f"Audio kayıt hatası: {e}")
        
        self.audio_record_buffer = []
        
        # UI güncelle
        self.record_btn.setText("🔴 KAYIT BAŞLAT")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #c0392b;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
        """)
        
        # Kayıt süresini hesapla
        if self.record_start_time is not None:
            duration = time.time() - self.record_start_time
            mins = int(duration // 60)
            secs = int(duration % 60)
            self.record_file_label.setText(f"📁 Kaydedildi: {self.record_base_name} ({mins}:{secs:02d})")
        
        self.statusbar.showMessage("Kayıt tamamlandı")
        QMessageBox.information(self, "Kayıt Tamamlandı", 
                               f"Kayıt başarıyla tamamlandı!\n\nDosya: {self.record_dir}")
    
    def _update_record_time(self):
        """Kayıt süresini güncelle"""
        if self.record_start_time is not None:
            elapsed = time.time() - self.record_start_time
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            self.record_time_label.setText(f"⏱ Kayıt Süresi: {hours:02d}:{mins:02d}:{secs:02d}")
    
    def _record_frame(self, frame: np.ndarray):
        """Kayıt için frame ekle"""
        if not self.is_recording or self.video_writer is None:
            return
        
        try:
            # Görselleştirmeler dahil edilecekse
            if self.record_viz_check.isChecked():
                # Ana frame'in altına görselleştirme alanı ekle
                h, w = frame.shape[:2]
                viz_height = 200
                
                # Yeni büyük frame oluştur
                combined_frame = np.zeros((h + viz_height, w, 3), dtype=np.uint8)
                combined_frame[:h, :, :] = frame
                
                # Spektrogram ve FFT'yi yakala (widget'lardan)
                # Bu basitleştirilmiş versiyon - asıl görüntüyü almak için grab kullanılabilir
                # Şimdilik sadece placeholder
                combined_frame[h:, :, :] = 30  # Koyu gri arkaplan
                
                # "Spektrogram + FFT" yazısı ekle
                cv2.putText(combined_frame, "Spektrogram + FFT (Kayit)", (10, h + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                self.video_writer.write(combined_frame)
            else:
                self.video_writer.write(frame)
                
        except Exception as e:
            logger.error(f"Frame kayıt hatası: {e}")
    
    def take_snapshot(self):
        """Anlık görüntü al - ekranda görünen her şeyi kaydet"""
        logger.info("Snapshot alınıyor...")
        
        try:
            # Kayıt dizini
            snapshots_dir = Path(__file__).parent.parent.parent / 'data' / 'snapshots'
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Video frame'i al (overlay dahil)
            if self.video_capture is not None and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    # Eğer overlay aktifse ve heatmap varsa, overlay uygula
                    if (self.beamforming_enabled and 
                        self.enable_overlay_check.isChecked() and 
                        self.latest_heatmap is not None):
                        frame = self._apply_overlay_to_frame(frame)
                    
                    # Kaydet
                    snapshot_path = snapshots_dir / f"snapshot_{timestamp}.png"
                    cv2.imwrite(str(snapshot_path), frame)
                    logger.info(f"Snapshot kaydedildi: {snapshot_path}")
                    
                    QMessageBox.information(self, "Snapshot", 
                                           f"Görüntü kaydedildi!\n\n{snapshot_path}")
                else:
                    QMessageBox.warning(self, "Hata", "Frame alınamadı!")
            else:
                QMessageBox.warning(self, "Hata", "Kamera bağlı değil!")
                
        except Exception as e:
            logger.error(f"Snapshot hatası: {e}")
            QMessageBox.warning(self, "Hata", f"Snapshot alınamadı: {e}")
    
    def _apply_overlay_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame'e overlay uygula (snapshot ve kayıt için)"""
        if self.latest_heatmap is None:
            return frame
        
        try:
            video_h, video_w = frame.shape[:2]
            
            # Heatmap'i video boyutuna ölçekle
            heatmap_resized = cv2.resize(
                self.latest_heatmap,
                (video_w, video_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            # RGB ve alpha ayır
            heatmap_rgb = heatmap_resized[:, :, :3]
            heatmap_alpha = heatmap_resized[:, :, 3] / 255.0
            
            # Kullanıcı şeffaflığı
            user_alpha = self.alpha_slider.value() / 100.0
            combined_alpha = heatmap_alpha * user_alpha
            
            # Alpha blending
            alpha_3ch = combined_alpha[:, :, np.newaxis]
            blended = (frame * (1 - alpha_3ch) + heatmap_rgb * alpha_3ch).astype(np.uint8)
            
            return blended
            
        except Exception as e:
            logger.error(f"Overlay uygulama hatası: {e}")
            return frame
    
    def on_freq_range_changed(self, min_val, max_val):
        """Frekans aralığı değiştiğinde"""
        logger.debug(f"Frekans aralığı: {min_val} - {max_val} Hz")
        # Update beamforming config
        if self.beamforming_config is not None:
            self._update_beamforming_config()
    
    def on_db_range_changed(self, min_val, max_val):
        """dB aralığı değiştiğinde"""
        logger.debug(f"dB aralığı: {min_val} - {max_val} dB")
        # dB range is used in heatmap rendering, no need to rebuild config
    
    def _on_algorithm_changed(self, algorithm_name):
        """Algoritma değiştiğinde"""
        logger.info(f"Algoritma değişti: {algorithm_name}")
        # Enable/disable n_sources spinner based on algorithm
        is_music = "MUSIC" in algorithm_name
        self.n_sources_spin.setEnabled(is_music)
        if is_music:
            self.n_sources_spin.setStyleSheet("")
        else:
            self.n_sources_spin.setStyleSheet("color: gray;")
    
    def _on_beamforming_toggle(self, state):
        """Beamforming checkbox toggle"""
        self.beamforming_enabled = (state == Qt.CheckState.Checked.value)
        logger.info(f"Beamforming {'enabled' if self.beamforming_enabled else 'disabled'}")
        
        if self.beamforming_enabled:
            # Update config when enabled
            self._update_beamforming_config()
            self.statusbar.showMessage("🎯 Beamforming aktif - Video overlay başladı")
        else:
            # Clear overlay when disabled
            self.latest_heatmap = None
            self.statusbar.showMessage("Beamforming devre dışı")
    
    def load_audio_file(self):
        """Ses dosyası yükle"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Ses Dosyası Seç", "data/recordings", 
            "Audio Files (*.wav *.mp3)"
        )
        if filename:
            logger.info(f"Ses dosyası yüklendi: {filename}")
            self.loaded_file_label.setText(f"Yüklü: {Path(filename).name}")
            self.statusbar.showMessage(f"Yüklendi: {Path(filename).name}")
    
    def load_video_file(self):
        """Video dosyası yükle"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Video Dosyası Seç", "data/recordings", 
            "Video Files (*.mp4 *.avi)"
        )
        if filename:
            logger.info(f"Video dosyası yüklendi: {filename}")
            self.loaded_file_label.setText(f"Yüklü: {Path(filename).name}")
            self.statusbar.showMessage(f"Yüklendi: {Path(filename).name}")
    
    def update_display(self):
        """Display güncelleme (timer callback)"""
        if not self.is_running:
            return
        
        # Frame timing için zaman damgası
        current_time = time.time()
        
        # Video frame al ve göster
        # Not: Beamforming aktifse, _update_video_overlay() zaten frame'i overlay ile birlikte gösterir
        # Beamforming kapalıysa, sadece plain video göster
        if self.video_capture is not None and self.video_capture.isOpened():
            if not self.beamforming_enabled or self.latest_heatmap is None:
                # Plain video (no overlay)
                ret, frame = self.video_capture.read()
                if ret:
                    self._display_image(frame)
                    self.frame_count += 1
            # else: overlay zaten _update_video_overlay()'de gösterildi
        
        # Gerçek FPS hesaplama - based on actual frame count over time window
        if not hasattr(self, '_fps_frame_times'):
            self._fps_frame_times = []
        
        self._fps_frame_times.append(current_time)
        # Keep only last 1 second of frame times
        self._fps_frame_times = [t for t in self._fps_frame_times if current_time - t < 1.0]
        
        # Calculate FPS from frame count in last second
        if len(self._fps_frame_times) >= 2:
            actual_fps = len(self._fps_frame_times)
            self.fps_label.setText(f"FPS: {actual_fps}")
        
        self._last_frame_time = current_time
        
        # Real CPU usage estimation from beamforming time
        # If beamforming takes X ms out of 33ms frame budget, CPU usage ≈ X/33 * 100
        if self.beamforming_enabled and len(self.beamforming_times) > 0:
            avg_bf_time = np.mean(self.beamforming_times)  # ms
            # Estimate: beamforming time / frame budget (33ms) as percentage
            # Plus base overhead (~10%)
            cpu_usage = min(100, int(10 + (avg_bf_time / 33.0) * 60))
        else:
            cpu_usage = 5  # Minimal when not processing
        self.cpu_label.setText(f"CPU: {cpu_usage}%")
        
        # VU meter'lar audio thread tarafından güncelleniyor - burada dokunma!
    
    def load_config(self):
        """Ayar dosyası yükle"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Ayar Dosyası Seç", "", "YAML Files (*.yaml *.yml)"
        )
        if filename:
            logger.info(f"Config yüklendi: {filename}")
            self.statusbar.showMessage(f"Ayar yüklendi: {Path(filename).name}")
    
    def save_config(self):
        """Ayarları kaydet"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Ayar Dosyası Kaydet", "", "YAML Files (*.yaml)"
        )
        if filename:
            logger.info(f"Config kaydedildi: {filename}")
            self.statusbar.showMessage(f"Ayar kaydedildi: {Path(filename).name}")
    
    def show_geometry(self):
        """Geometri görüntüleyici aç"""
        logger.info("Geometri görüntüleyici açılıyor...")
        QMessageBox.information(self, "Geometri", 
                               "Geometri görüntüleyici yakında eklenecek!")
    
    def show_about(self):
        """Hakkında dialogu"""
        QMessageBox.about(self, "Hakkında",
            """<h2>UMA-16 Akustik Kamera Sistemi</h2>
            <p>Version 0.1</p>
            <p>Real-time akustik kaynak lokalizasyonu ve görselleştirme</p>
            <p><b>Yüksek Lisans Tezi</b><br>
            Emre Göktuğ AKTAŞ<br>
            2024</p>
            <p>miniDSP UMA-16v2 (4x4 Grid) + USB Kamera</p>
            """)
    
    # ============================================================================
    # BEAMFORMING & ACOUSTIC IMAGING
    # ============================================================================
    
    def _init_beamforming(self):
        """Initialize beamforming: load geometry, create grid"""
        try:
            # Load microphone geometry
            micgeom_path = Path(__file__).parent.parent.parent / 'micgeom.xml'
            self.mic_positions = load_mic_geometry(str(micgeom_path))
            logger.info(f"Loaded {len(self.mic_positions)} microphone positions")
            
            # Create beamforming configuration
            self._update_beamforming_config()
            
            logger.info("Beamforming initialized successfully")
            
        except Exception as e:
            logger.error(f"Beamforming initialization failed: {e}")
            self.statusbar.showMessage(f"Beamforming hatası: {e}")
    
    def _update_beamforming_config(self):
        """Update beamforming configuration from GUI parameters"""
        try:
            # Get parameters from GUI
            freq_min, freq_max = self.freq_range_slider.values()
            focus_distance = self.focus_distance_spin.value()
            grid_resolution = self.grid_resolution_spin.value() / 100.0  # cm to m
            
            # Create config
            # Grid size: larger area to cover more of the camera FOV
            # At 1m distance, a typical webcam FOV is ~60-80 degrees
            # This translates to roughly 1.0-1.5m visible area
            self.beamforming_config = BeamformingConfig(
                grid_size_x=1.2,  # 1.2 meters (120 cm) - wider coverage
                grid_size_y=1.2,  # 1.2 meters (120 cm)
                grid_resolution=grid_resolution,
                focus_distance=focus_distance,
                freq_min=float(freq_min),
                freq_max=float(freq_max),
                field_type='near-field',  # Near-field for accurate localization
                sound_speed=343.0
            )
            
            # Create focus grid
            self.grid_points, self.grid_shape = create_focus_grid(self.beamforming_config)
            
            # Precompute distances for realtime beamforming (major speedup)
            if self.mic_positions is not None and self.grid_points is not None:
                self.cached_distances = _precompute_distances(self.mic_positions, self.grid_points)
                logger.info(f"Precomputed distance matrix: {self.cached_distances.shape}")
            
            logger.info(f"Beamforming config updated: {self.grid_shape[0]}x{self.grid_shape[1]} grid, "
                       f"freq=[{freq_min}, {freq_max}] Hz, z={focus_distance}m")
            
        except Exception as e:
            logger.error(f"Failed to update beamforming config: {e}")
    
    def _run_beamforming(self, audio_data: np.ndarray, sample_rate: float):
        """
        Run beamforming on audio data and generate heatmap
        
        Uses optimized realtime beamformers with:
        - Precomputed distances
        - Limited frequency bins
        - Vectorized operations
        
        Args:
            audio_data: (n_samples, n_mics) audio data
            sample_rate: Sampling rate
        """
        import time
        start_time = time.perf_counter()
        
        try:
            # Check if beamforming is ready
            if self.mic_positions is None or self.grid_points is None:
                return
            
            # Get selected algorithm and n_sources
            algorithm = self.algorithm_combo.currentText()
            n_sources = self.n_sources_spin.value()
            
            # Run selected beamformer (using REALTIME optimized versions)
            if "MVDR" in algorithm or "Minimum Variance" in algorithm:
                power_map = mvdr_beamformer_realtime(
                    audio_data,
                    self.mic_positions,
                    self.grid_points,
                    sample_rate,
                    self.beamforming_config,
                    diagonal_loading=1e-3,
                    max_freq_bins=self.max_freq_bins,
                    distances=self.cached_distances
                )
            elif "MUSIC" in algorithm:
                power_map = music_beamformer_realtime(
                    audio_data,
                    self.mic_positions,
                    self.grid_points,
                    sample_rate,
                    self.beamforming_config,
                    n_sources=n_sources,
                    max_freq_bins=self.max_freq_bins,
                    distances=self.cached_distances
                )
            else:  # Default: DAS
                power_map = das_beamformer_realtime(
                    audio_data,
                    self.mic_positions,
                    self.grid_points,
                    sample_rate,
                    self.beamforming_config,
                    max_freq_bins=self.max_freq_bins,
                    distances=self.cached_distances
                )
            
            # Convert to dB
            power_db = power_to_db(power_map)
            
            # Reshape to 2D grid
            power_grid = power_db.reshape(self.grid_shape)
            
            # Find multiple peaks in power grid
            self.detected_peaks = self._detect_multiple_peaks(power_grid, power_db, n_sources)
            # Keep legacy single peak for compatibility
            self.detected_peak = self.detected_peaks[0] if self.detected_peaks else None
            
            # Update 3D spatial visualization (throttled - every 5th frame)
            if not hasattr(self, '_3d_update_counter'):
                self._3d_update_counter = 0
            self._3d_update_counter += 1
            if self._3d_update_counter >= 5:  # Update every 5 frames to reduce overhead
                self._update_3d_visualization()
                # Update detected sources panel
                self._update_sources_panel()
                self._3d_update_counter = 0
            
            # Convert to heatmap (RGB image)
            self.latest_heatmap = self._power_to_heatmap(power_grid)
            
            # Update overlay on video
            self._update_video_overlay()
            
            # Performance monitoring
            elapsed = (time.perf_counter() - start_time) * 1000  # ms
            self.beamforming_times.append(elapsed)
            if len(self.beamforming_times) > 30:
                self.beamforming_times.pop(0)
            
            avg_time = np.mean(self.beamforming_times)
            logger.debug(f"Beamforming ({algorithm}): {elapsed:.1f} ms (avg: {avg_time:.1f} ms)")
            
        except Exception as e:
            logger.error(f"Beamforming error: {e}", exc_info=True)
    
    def _detect_peak(self, power_grid: np.ndarray, power_map_1d: np.ndarray) -> Optional[dict]:
        """
        Detect the brightest point (peak) in the power grid
        
        Args:
            power_grid: (height, width) power in dB
            power_map_1d: (n_points,) power map (1D, before reshape)
            
        Returns:
            dict with keys: x, y, z, power_db, grid_row, grid_col, or None if no peak
        """
        try:
            # Get dB threshold
            db_min, db_max = self.db_range_slider.values()
            threshold = db_min + (db_max - db_min) * 0.2  # 20% above min
            
            # Find global maximum
            max_idx_2d = np.unravel_index(np.argmax(power_grid), power_grid.shape)
            max_power = power_grid[max_idx_2d]
            
            # Check if above threshold
            if max_power < threshold:
                return None
            
            # Convert 2D grid index to 1D index
            grid_row, grid_col = max_idx_2d
            max_idx_1d = grid_row * self.grid_shape[1] + grid_col
            
            # Get 3D position from grid_points
            peak_position = self.grid_points[max_idx_1d]
            
            peak_info = {
                'x': peak_position[0],
                'y': peak_position[1],
                'z': peak_position[2],
                'power_db': max_power,
                'grid_row': grid_row,
                'grid_col': grid_col
            }
            
            logger.debug(f"Peak detected: pos=({peak_info['x']:.2f}, {peak_info['y']:.2f}, {peak_info['z']:.2f}), "
                        f"power={peak_info['power_db']:.1f} dB")
            
            return peak_info
            
        except Exception as e:
            logger.error(f"Peak detection error: {e}")
            return None
    
    def _detect_multiple_peaks(self, power_grid: np.ndarray, power_map_1d: np.ndarray, n_peaks: int = 1) -> list:
        """
        Detect multiple peaks (local maxima) in the power grid
        
        Uses scipy's peak_local_max to find local maxima, then filters by threshold
        and returns the top n_peaks.
        
        Args:
            power_grid: (height, width) power in dB
            power_map_1d: (n_points,) power map (1D, before reshape)
            n_peaks: Maximum number of peaks to detect
            
        Returns:
            List of dicts with keys: x, y, z, power_db, grid_row, grid_col
        """
        from scipy.ndimage import maximum_filter, label
        
        try:
            # Get dB threshold
            db_min, db_max = self.db_range_slider.values()
            threshold = db_min + (db_max - db_min) * 0.2  # 20% above min
            
            # Find local maxima using maximum filter
            # A point is a local maximum if it equals the maximum in its neighborhood
            neighborhood_size = 5  # Size of neighborhood for local max detection
            local_max = maximum_filter(power_grid, size=neighborhood_size)
            
            # Detect peaks: points that are local maxima and above threshold
            peak_mask = (power_grid == local_max) & (power_grid > threshold)
            
            # Get coordinates of all peaks
            peak_coords = np.where(peak_mask)
            
            if len(peak_coords[0]) == 0:
                return []
            
            # Get power values at peak locations
            peak_powers = power_grid[peak_coords]
            
            # Sort by power (descending) and take top n_peaks
            sorted_indices = np.argsort(peak_powers)[::-1]
            top_indices = sorted_indices[:n_peaks]
            
            peaks = []
            colors = [
                (0, 255, 0),    # Green - primary
                (255, 165, 0),  # Orange - secondary
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (255, 0, 0),    # Red
                (0, 0, 255),    # Blue
                (128, 0, 128),  # Purple
                (0, 128, 128),  # Teal
                (128, 128, 0),  # Olive
            ]
            
            for i, idx in enumerate(top_indices):
                grid_row = peak_coords[0][idx]
                grid_col = peak_coords[1][idx]
                power_val = peak_powers[idx]
                
                # Convert 2D grid index to 1D index
                max_idx_1d = grid_row * self.grid_shape[1] + grid_col
                
                # Get 3D position from grid_points
                peak_position = self.grid_points[max_idx_1d]
                
                peak_info = {
                    'x': peak_position[0],
                    'y': peak_position[1],
                    'z': peak_position[2],
                    'power_db': power_val,
                    'grid_row': grid_row,
                    'grid_col': grid_col,
                    'color': colors[i % len(colors)],  # Assign color for visualization
                    'index': i + 1  # 1-based index for display
                }
                peaks.append(peak_info)
            
            logger.debug(f"Detected {len(peaks)} peaks")
            return peaks
            
        except Exception as e:
            logger.error(f"Multi-peak detection error: {e}")
            return []
    
    def _power_to_heatmap(self, power_grid: np.ndarray) -> np.ndarray:
        """
        Convert power grid (dB) to RGB heatmap
        RED = high power (sound source), BLUE = low power (quiet)
        
        Args:
            power_grid: (height, width) power in dB
            
        Returns:
            heatmap: (height, width, 4) RGBA uint8 image (with alpha channel)
        """
        # Get dB range from GUI - use these as reference thresholds
        db_min_ui, db_max_ui = self.db_range_slider.values()
        
        # Use the UI slider values as the normalization range
        # This allows user to control what power levels map to colors
        # Clip power values to the UI range
        power_clipped = np.clip(power_grid, db_min_ui, db_max_ui)
        
        # Normalize to [0, 1] using UI slider range
        normalized = (power_clipped - db_min_ui) / (db_max_ui - db_min_ui + 1e-6)
        normalized = np.clip(normalized, 0, 1)
        
        # Apply gamma correction to enhance contrast
        gamma = 0.7
        normalized_gamma = np.power(normalized, gamma)
        
        # Apply gaussian smoothing for nicer visual
        normalized_smooth = gaussian_filter(normalized_gamma, sigma=1.5)
        
        # Convert to [0, 255] uint8 for colormap
        # HIGH values (loud) should be RED, LOW values (quiet) should be BLUE
        normalized_uint8 = (normalized_smooth * 255).astype(np.uint8)
        
        # Get colormap from GUI
        colormap_name = self.colormap_combo.currentText()
        
        # Apply colormap - JET goes from BLUE (0) to RED (255)
        # So high power → 255 → RED, low power → 0 → BLUE ✓
        colormap_dict = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'coolwarm': cv2.COLORMAP_COOL,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'turbo': cv2.COLORMAP_TURBO
        }
        
        cv_colormap = colormap_dict.get(colormap_name, cv2.COLORMAP_JET)
        heatmap_bgr = cv2.applyColorMap(normalized_uint8, cv_colormap)
        
        # Convert BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Alpha channel: make low power areas more transparent
        # Only show overlay where there's significant sound
        # Use a threshold based on normalized value (bottom 30% is mostly transparent)
        alpha_threshold = 0.25
        alpha = np.zeros_like(normalized_smooth, dtype=np.float32)
        above_threshold = normalized_smooth > alpha_threshold
        # Scale alpha from threshold to 1.0
        alpha[above_threshold] = (normalized_smooth[above_threshold] - alpha_threshold) / (1.0 - alpha_threshold)
        
        # Apply power curve to alpha for better contrast
        alpha = np.power(alpha, 0.6)
        alpha = (alpha * 255).astype(np.uint8)
        
        # Add alpha channel
        heatmap_rgba = np.dstack([heatmap_rgb, alpha])
        
        return heatmap_rgba
    
    def _draw_corner_brackets(self, frame, cx, cy, size, color, thickness=2, gap=8):
        """Draw corner brackets (HUD-style target acquisition)"""
        # Corner length
        corner_len = size // 3
        
        # Top-left corner
        cv2.line(frame, (cx - size, cy - size), (cx - size + corner_len, cy - size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx - size, cy - size), (cx - size, cy - size + corner_len), color, thickness, cv2.LINE_AA)
        
        # Top-right corner
        cv2.line(frame, (cx + size, cy - size), (cx + size - corner_len, cy - size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + size, cy - size), (cx + size, cy - size + corner_len), color, thickness, cv2.LINE_AA)
        
        # Bottom-left corner
        cv2.line(frame, (cx - size, cy + size), (cx - size + corner_len, cy + size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx - size, cy + size), (cx - size, cy + size - corner_len), color, thickness, cv2.LINE_AA)
        
        # Bottom-right corner
        cv2.line(frame, (cx + size, cy + size), (cx + size - corner_len, cy + size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + size, cy + size), (cx + size, cy + size - corner_len), color, thickness, cv2.LINE_AA)
        
        # Center crosshair lines (short lines with gap in middle)
        # Horizontal
        cv2.line(frame, (cx - size + corner_len + 5, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + size - corner_len - 5, cy), color, 1, cv2.LINE_AA)
        # Vertical
        cv2.line(frame, (cx, cy - size + corner_len + 5), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size - corner_len - 5), color, 1, cv2.LINE_AA)
    
    def _draw_diamond_marker(self, frame, cx, cy, size, color, thickness=1):
        """Draw diamond marker for secondary targets"""
        pts = np.array([
            [cx, cy - size],  # Top
            [cx + size, cy],  # Right
            [cx, cy + size],  # Bottom
            [cx - size, cy],  # Left
        ], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    
    def _draw_callout_box(self, frame, cx, cy, texts, color, direction='right'):
        """Draw callout line with text box (HUD style)"""
        # Text measurement
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        
        # Calculate text sizes
        max_width = 0
        total_height = 0
        line_heights = []
        for text in texts:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness_text)
            max_width = max(max_width, w)
            line_heights.append(h + 5)
            total_height += h + 8
        
        # Padding
        padding = 8
        box_width = max_width + padding * 2
        box_height = total_height + padding
        
        # Position calculation
        line_length = 40
        if direction == 'right':
            line_end_x = cx + line_length + 20
            line_end_y = cy - 20
            box_x = line_end_x
            box_y = line_end_y - box_height // 2
        else:  # left
            line_end_x = cx - line_length - 20
            line_end_y = cy - 20
            box_x = line_end_x - box_width
            box_y = line_end_y - box_height // 2
        
        # Create overlay for semi-transparent box
        overlay = frame.copy()
        
        # Draw callout line (angled)
        cv2.line(frame, (cx, cy), (cx + (15 if direction == 'right' else -15), cy - 15), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + (15 if direction == 'right' else -15), cy - 15), (line_end_x, line_end_y), color, 1, cv2.LINE_AA)
        
        # Draw semi-transparent background box
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw box border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), color, 1, cv2.LINE_AA)
        
        # Draw accent line on left/right edge
        if direction == 'right':
            cv2.line(frame, (box_x, box_y), (box_x, box_y + box_height), color, 2, cv2.LINE_AA)
        else:
            cv2.line(frame, (box_x + box_width, box_y), (box_x + box_width, box_y + box_height), color, 2, cv2.LINE_AA)
        
        # Draw text
        text_x = box_x + padding
        text_y = box_y + padding + line_heights[0]
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness_text, cv2.LINE_AA)
            if i < len(texts) - 1:
                text_y += line_heights[i + 1] + 3
    
    def _draw_hud_frame_elements(self, frame, video_w, video_h):
        """Draw HUD frame corner elements and status indicators"""
        color = (0, 200, 180)  # Cyan-ish
        
        # Frame corner decorations
        corner_size = 60
        
        # Top-left
        cv2.line(frame, (10, 10), (10 + corner_size, 10), color, 2, cv2.LINE_AA)
        cv2.line(frame, (10, 10), (10, 10 + corner_size), color, 2, cv2.LINE_AA)
        
        # Top-right
        cv2.line(frame, (video_w - 10, 10), (video_w - 10 - corner_size, 10), color, 2, cv2.LINE_AA)
        cv2.line(frame, (video_w - 10, 10), (video_w - 10, 10 + corner_size), color, 2, cv2.LINE_AA)
        
        # Bottom-left
        cv2.line(frame, (10, video_h - 10), (10 + corner_size, video_h - 10), color, 2, cv2.LINE_AA)
        cv2.line(frame, (10, video_h - 10), (10, video_h - 10 - corner_size), color, 2, cv2.LINE_AA)
        
        # Bottom-right
        cv2.line(frame, (video_w - 10, video_h - 10), (video_w - 10 - corner_size, video_h - 10), color, 2, cv2.LINE_AA)
        cv2.line(frame, (video_w - 10, video_h - 10), (video_w - 10, video_h - 10 - corner_size), color, 2, cv2.LINE_AA)
        
        # Draw scanning lines (subtle horizontal lines)
        for i in range(3):
            y_pos = int(video_h * (0.25 + i * 0.25))
            cv2.line(frame, (15, y_pos), (35, y_pos), color, 1, cv2.LINE_AA)
            cv2.line(frame, (video_w - 35, y_pos), (video_w - 15, y_pos), color, 1, cv2.LINE_AA)
    
    def _update_video_overlay(self):
        """Update video frame with acoustic heatmap overlay - HUD Style"""
        try:
            # Check if we have video and heatmap
            if self.video_capture is None or self.latest_heatmap is None:
                return
            
            # Capture video frame
            ret, frame = self.video_capture.read()
            if not ret:
                return
            
            video_h, video_w = frame.shape[:2]
            
            # Overlay aktif mi kontrol et
            if not self.enable_overlay_check.isChecked():
                # Overlay kapalı - sadece plain video göster
                # Kayıt varsa frame'i kaydet
                if self.is_recording:
                    self._record_frame(frame)
                self._display_image(frame)
                return
            
            heatmap_h, heatmap_w = self.latest_heatmap.shape[:2]
            
            # ============================================================
            # STRATEGY: FULL SCREEN OVERLAY (stretch to entire video)
            # ============================================================
            overlay_w = video_w
            overlay_h = video_h
            x_offset = 0
            y_offset = 0
            
            # Get grid physical size for crosshair mapping
            grid_size_x = self.beamforming_config.grid_size_x  # meters
            grid_size_y = self.beamforming_config.grid_size_y  # meters
            
            # Resize heatmap to overlay dimensions
            heatmap_resized = cv2.resize(
                self.latest_heatmap,
                (overlay_w, overlay_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Extract ROI from video
            roi = frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
            
            # Safety check
            if roi.shape[:2] != heatmap_resized.shape[:2]:
                logger.warning(f"ROI shape mismatch: {roi.shape[:2]} vs {heatmap_resized.shape[:2]}")
                self._display_image(frame)
                return
            
            # Extract RGB and alpha from heatmap
            heatmap_rgb = heatmap_resized[:, :, :3]
            heatmap_alpha = heatmap_resized[:, :, 3] / 255.0  # Normalize to [0, 1]
            
            # Apply user-defined opacity from slider
            user_alpha = self.alpha_slider.value() / 100.0
            combined_alpha = heatmap_alpha * user_alpha
            
            # Alpha blending: output = roi * (1 - alpha) + heatmap * alpha
            alpha_3ch = combined_alpha[:, :, np.newaxis]
            blended = (roi * (1 - alpha_3ch) + heatmap_rgb * alpha_3ch).astype(np.uint8)
            
            # Update frame with blended overlay
            frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = blended
            
            # ============================================================
            # Draw HUD frame elements
            # ============================================================
            self._draw_hud_frame_elements(frame, video_w, video_h)
            
            # ============================================================
            # Draw HUD-style markers at detected peak positions
            # ============================================================
            if self.show_peaks_check.isChecked():
                for peak in self.detected_peaks:
                    peak_x_m = peak['x']  # meters
                    peak_y_m = peak['y']  # meters
                    color = peak.get('color', (0, 255, 0))  # BGR color
                    peak_index = peak.get('index', 1)
                    
                    # Convert from physical coordinates (meters) to pixel coordinates
                    norm_x = (peak_x_m + grid_size_x / 2.0) / grid_size_x  # 0 to 1
                    norm_y = (peak_y_m + grid_size_y / 2.0) / grid_size_y  # 0 to 1
                    
                    # Map to overlay pixel coordinates (Y axis flipped)
                    peak_pixel_x = int(norm_x * overlay_w)
                    peak_pixel_y = int((1.0 - norm_y) * overlay_h)
                    
                    # Convert to absolute video coordinates
                    peak_video_x = x_offset + peak_pixel_x
                    peak_video_y = y_offset + peak_pixel_y
                    
                    # Clamp to video bounds
                    peak_video_x = int(np.clip(peak_video_x, 50, video_w - 50))
                    peak_video_y = int(np.clip(peak_video_y, 50, video_h - 50))
                    
                    # ============================================================
                    # PRIMARY TARGET: Corner Brackets with Callout
                    # ============================================================
                    if peak_index == 1:
                        # Pulsing effect simulation (using time-based alpha would be better)
                        bracket_size = 45
                        
                        # Draw corner brackets (primary target - bright color)
                        self._draw_corner_brackets(frame, peak_video_x, peak_video_y, 
                                                   bracket_size, color, thickness=2)
                        
                        # Draw outer glow brackets (larger, dimmer)
                        glow_color = tuple(int(c * 0.4) for c in color)
                        self._draw_corner_brackets(frame, peak_video_x, peak_video_y, 
                                                   bracket_size + 8, glow_color, thickness=1)
                        
                        # Center dot
                        cv2.circle(frame, (peak_video_x, peak_video_y), 4, color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (peak_video_x, peak_video_y), 6, color, 1, cv2.LINE_AA)
                        
                        # Callout box with info
                        power_text = f"PWR: {peak['power_db']:.1f} dB"
                        pos_text = f"POS: ({peak_x_m*100:.1f}, {peak_y_m*100:.1f}) cm"
                        freq_text = f"SRC: #{peak_index}"
                        
                        # Determine callout direction based on position
                        direction = 'right' if peak_video_x < video_w // 2 else 'left'
                        self._draw_callout_box(frame, peak_video_x, peak_video_y, 
                                              [freq_text, power_text, pos_text], color, direction)
                    
                    # ============================================================
                    # SECONDARY TARGETS: Diamond markers (smaller, dimmer)
                    # ============================================================
                    else:
                        # Dimmer color for secondary targets
                        dim_color = tuple(int(c * 0.7) for c in color)
                        
                        # Diamond marker
                        self._draw_diamond_marker(frame, peak_video_x, peak_video_y, 15, dim_color, 2)
                        self._draw_diamond_marker(frame, peak_video_x, peak_video_y, 20, 
                                                  tuple(int(c * 0.3) for c in color), 1)
                        
                        # Small center dot
                        cv2.circle(frame, (peak_video_x, peak_video_y), 2, dim_color, -1, cv2.LINE_AA)
                        
                        # Small label
                        label = f"#{peak_index}"
                        cv2.putText(frame, label, (peak_video_x + 18, peak_video_y - 8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, label, (peak_video_x + 18, peak_video_y - 8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim_color, 1, cv2.LINE_AA)
            
            # Kayıt için frame'i kaydet
            if self.is_recording:
                self._record_frame(frame)
            
            # Display frame
            self._display_image(frame)
            
        except Exception as e:
            logger.error(f"Video overlay error: {e}", exc_info=True)
    
    def closeEvent(self, event):
        """Pencere kapatılırken"""
        if self.is_running:
            self.stop_system()
        
        reply = QMessageBox.question(
            self, 'Çıkış',
            "Uygulamayı kapatmak istediğinize emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Uygulama kapatılıyor")
            event.accept()
        else:
            event.ignore()


def main():
    """GUI'yi başlat"""
    from PySide6.QtWidgets import QApplication
    
    # Logging ayarla
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = QApplication(sys.argv)
    
    # Dark Theme entegrasyonu (pyqtdarktheme)
    try:
        import qdarktheme
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        logger.info("Dark theme başarıyla yüklendi")
    except ImportError:
        logger.warning("pyqtdarktheme bulunamadı, varsayılan stil kullanılıyor")
        # Fallback: Fusion stili
        app.setStyle('Fusion')
        
        # Manuel dark palette
        from PySide6.QtGui import QPalette, QColor
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        app.setPalette(dark_palette)
    
    # Ana pencereyi oluştur
    window = AcousticCameraGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
