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
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QGroupBox, QSlider, QComboBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QStatusBar,
    QMenuBar, QFileDialog, QMessageBox,
    QSplitter, QFrame, QProgressBar, QTextEdit, QListWidget,
    QScrollArea, QGridLayout
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor, QAction

import numpy as np
import cv2

from .custom_widgets import DoubleRangeSlider

# Audio stream thread import
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
from audio.stream_thread import AudioStreamThread

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
        
        # Video capture
        self.video_capture = None
        self.frame_count = 0
        
        # VU Meters
        self.vu_meters = []
        
        # Audio stream thread
        self.audio_thread = None
        
        # GUI bileşenlerini oluştur
        self._init_ui()
        self._init_menubar()
        self._init_statusbar()
        
        logger.info("GUI başlatıldı")
    
    def _init_ui(self):
        """Ana UI bileşenlerini oluştur"""
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Sol panel: Kontroller - scrollable
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Yatay scroll gerekirse göster
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_panel = self._create_control_panel()
        left_scroll.setWidget(control_panel)
        left_scroll.setMaximumWidth(450)  # Biraz daha geniş
        left_scroll.setMinimumWidth(380)
        main_layout.addWidget(left_scroll)
        
        # Orta: Video + Overlay + Spektrogram
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, stretch=3)
        
        # Sağ: Analiz panelleri + VU Meters - scrollable
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_panel = self._create_right_panel()
        right_scroll.setWidget(right_panel)
        right_scroll.setMaximumWidth(420)
        right_scroll.setMinimumWidth(350)
        main_layout.addWidget(right_scroll)
    
    def _create_control_panel(self) -> QWidget:
        """Sol kontrol panelini oluştur"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. BAĞLANTI AYARLARI
        conn_group = self._create_connection_group()
        layout.addWidget(conn_group)
        
        # 2. SES AYARLARI
        audio_group = self._create_audio_group()
        layout.addWidget(audio_group)
        
        # 3. BEAMFORMING + GÖRSELLEŞTİRME (BİRLEŞTİRİLDİ)
        beamform_group = self._create_beamforming_visualization_group()
        layout.addWidget(beamform_group)
        
        # 4. KAYIT & DOSYA YÜKLEME
        file_group = self._create_file_operations_group()
        layout.addWidget(file_group)
        
        # Spacer
        layout.addStretch()
        
        return panel
    
    def _create_connection_group(self) -> QGroupBox:
        """Bağlantı ayarları grubu"""
        group = QGroupBox("🔌 Bağlantı Ayarları")
        layout = QVBoxLayout()
        
        # Audio cihaz seçimi
        layout.addWidget(QLabel("Audio Cihazı:"))
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.addItems(["UMA16v2 (Auto)", "Manuel Seçim..."])
        layout.addWidget(self.audio_device_combo)
        
        # Video cihaz seçimi
        layout.addWidget(QLabel("Video Cihazı:"))
        self.video_device_combo = QComboBox()
        self.video_device_combo.addItems(["Webcam 0", "Webcam 1", "Webcam 2", "Webcam 3"])
        layout.addWidget(self.video_device_combo)
        
        # Bağlantı durumu
        status_layout = QHBoxLayout()
        self.audio_status_label = QLabel("🔴 Audio: Bağlı değil")
        self.video_status_label = QLabel("🔴 Video: Bağlı değil")
        status_layout.addWidget(self.audio_status_label)
        status_layout.addWidget(self.video_status_label)
        layout.addLayout(status_layout)
        
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
    
    def _create_audio_group(self) -> QGroupBox:
        """Ses işleme ayarları grubu"""
        group = QGroupBox("🎤 Ses Ayarları")
        layout = QVBoxLayout()
        
        # Sample rate
        layout.addWidget(QLabel("Örnekleme Hızı:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["48000 Hz", "44100 Hz", "96000 Hz"])
        layout.addWidget(self.sample_rate_combo)
        
        # Chunk size
        layout.addWidget(QLabel("Buffer Boyutu:"))
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(512, 8192)
        self.chunk_size_spin.setValue(4096)
        self.chunk_size_spin.setSingleStep(512)
        layout.addWidget(self.chunk_size_spin)
        
        # Filtreleme
        filter_layout = QHBoxLayout()
        self.highpass_check = QCheckBox("Highpass")
        self.highpass_check.setChecked(True)
        self.lowpass_check = QCheckBox("Lowpass")
        self.lowpass_check.setChecked(True)
        filter_layout.addWidget(self.highpass_check)
        filter_layout.addWidget(self.lowpass_check)
        layout.addLayout(filter_layout)
        
        # Highpass cutoff
        layout.addWidget(QLabel("Highpass Cutoff (Hz):"))
        self.highpass_spin = QSpinBox()
        self.highpass_spin.setRange(10, 1000)
        self.highpass_spin.setValue(100)
        layout.addWidget(self.highpass_spin)
        
        # Lowpass cutoff
        layout.addWidget(QLabel("Lowpass Cutoff (Hz):"))
        self.lowpass_spin = QSpinBox()
        self.lowpass_spin.setRange(1000, 24000)
        self.lowpass_spin.setValue(10000)
        layout.addWidget(self.lowpass_spin)
        
        group.setLayout(layout)
        return group
    
    def _create_beamforming_visualization_group(self) -> QGroupBox:
        """Beamforming + Görselleştirme ayarları (birleştirildi)"""
        group = QGroupBox("🎯 Beamforming & Görselleştirme")
        layout = QVBoxLayout()
        
        # Algoritma seçimi
        layout.addWidget(QLabel("Algoritma:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems([
            "DAS (Delay-and-Sum)",
            "MVDR (Minimum Variance)",
            "MUSIC",
            "CLEAN-SC"
        ])
        layout.addWidget(self.algorithm_combo)
        
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
        layout.addWidget(self.grid_resolution_spin)
        
        # Z mesafesi (odak mesafesi)
        layout.addWidget(QLabel("Odak Mesafesi (m):"))
        self.focus_distance_spin = QDoubleSpinBox()
        self.focus_distance_spin.setRange(0.3, 5.0)
        self.focus_distance_spin.setValue(1.0)
        self.focus_distance_spin.setSingleStep(0.1)
        self.focus_distance_spin.setDecimals(2)
        layout.addWidget(self.focus_distance_spin)
        
        # --- GÖRSELLEŞTIRME ---
        layout.addWidget(QLabel("─" * 30))
        
        # Renk haritası
        layout.addWidget(QLabel("Renk Haritası:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "jet", "hot", "viridis", "plasma", "inferno",
            "coolwarm", "rainbow", "turbo"
        ])
        layout.addWidget(self.colormap_combo)
        
        # Overlay alpha
        layout.addWidget(QLabel("Overlay Şeffaflığı:"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)
        self.alpha_value_label = QLabel("60%")
        self.alpha_slider.valueChanged.connect(
            lambda v: self.alpha_value_label.setText(f"{v}%")
        )
        layout.addWidget(self.alpha_slider)
        layout.addWidget(self.alpha_value_label)
        
        # Görselleştirme seçenekleri
        self.show_contours_check = QCheckBox("Kontur çizgileri")
        self.show_peaks_check = QCheckBox("Peak noktaları")
        self.show_grid_check = QCheckBox("Grid göster")
        layout.addWidget(self.show_contours_check)
        layout.addWidget(self.show_peaks_check)
        layout.addWidget(self.show_grid_check)
        
        group.setLayout(layout)
        return group
    
    def _create_file_operations_group(self) -> QGroupBox:
        """Dosya işlemleri: Kayıt + Yükleme"""
        group = QGroupBox("💾 Dosya İşlemleri")
        layout = QVBoxLayout()
        
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
        
        layout.addWidget(QLabel("─" * 30))
        
        # --- KAYIT ---
        layout.addWidget(QLabel("<b>Kayıt:</b>"))
        
        # Kayıt butonu
        self.record_btn = QPushButton("🔴 KAYIT BAŞLAT")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_btn)
        
        # Snapshot butonu
        self.snapshot_btn = QPushButton("📸 Snapshot")
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        layout.addWidget(self.snapshot_btn)
        
        # Kayıt seçenekleri
        self.record_audio_check = QCheckBox("Ses kaydet (.wav)")
        self.record_audio_check.setChecked(True)
        self.record_video_check = QCheckBox("Video kaydet (.mp4)")
        self.record_video_check.setChecked(True)
        self.record_data_check = QCheckBox("Veri kaydet (.h5)")
        self.record_data_check.setChecked(False)
        
        layout.addWidget(self.record_audio_check)
        layout.addWidget(self.record_video_check)
        layout.addWidget(self.record_data_check)
        
        # Kayıt süresi
        self.record_time_label = QLabel("Kayıt Süresi: 00:00:00")
        layout.addWidget(self.record_time_label)
        
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
        
        # Spektrogram
        spectrogram_group = QGroupBox("📊 Spektrogram (Frekans-Zaman)")
        spec_layout = QVBoxLayout()
        self.spectrogram_label = QLabel()
        self.spectrogram_label.setMinimumSize(400, 150)
        self.spectrogram_label.setStyleSheet("QLabel { background-color: black; }")
        self.spectrogram_label.setText("Spektrogram buraya gelecek")
        self.spectrogram_label.setAlignment(Qt.AlignCenter)
        spec_layout.addWidget(self.spectrogram_label)
        spectrogram_group.setLayout(spec_layout)
        
        # Waveform
        waveform_group = QGroupBox("📈 Audio Waveform (Frekans-Şiddet)")
        wave_layout = QVBoxLayout()
        self.waveform_label = QLabel()
        self.waveform_label.setMinimumSize(400, 150)
        self.waveform_label.setStyleSheet("QLabel { background-color: black; }")
        self.waveform_label.setText("Waveform buraya gelecek")
        self.waveform_label.setAlignment(Qt.AlignCenter)
        wave_layout.addWidget(self.waveform_label)
        waveform_group.setLayout(wave_layout)
        
        analysis_splitter.addWidget(spectrogram_group)
        analysis_splitter.addWidget(waveform_group)
        
        layout.addWidget(analysis_splitter)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Sağ panel: 3D Uzamsal Konum + Tespit Edilen Kaynaklar + VU Meters (en altta)"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 3D Uzamsal Konum (en üstte)
        spatial_group = QGroupBox("🎯 3D Uzamsal Konum")
        spatial_layout = QVBoxLayout()
        self.spatial_3d_label = QLabel()
        self.spatial_3d_label.setMinimumSize(280, 280)
        self.spatial_3d_label.setStyleSheet("QLabel { background-color: black; border: 1px solid #555; }")
        self.spatial_3d_label.setText("3D pozisyon\ngörseli")
        self.spatial_3d_label.setAlignment(Qt.AlignCenter)
        spatial_layout.addWidget(self.spatial_3d_label)
        spatial_group.setLayout(spatial_layout)
        layout.addWidget(spatial_group)
        
        # Tespit Edilen Ses Kaynakları - Dinamik Widget
        sources_group = QGroupBox("🔊 Tespit Edilen Kaynaklar")
        sources_layout = QVBoxLayout()
        
        # Dinamik kaynak container
        self.sources_container = QWidget()
        self.sources_container_layout = QVBoxLayout(self.sources_container)
        self.sources_container_layout.setSpacing(8)
        self.sources_container_layout.setContentsMargins(5, 5, 5, 5)
        
        # Başlangıçta "Kaynak algılanamadı" mesajı
        self.no_source_label = QLabel("⚠️ Kaynak Algılanamadı")
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
        group = QGroupBox("📊 Mikrofon Array (16 Ch)")
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
        Tespit edilen ses kaynaklarını güncelle
        
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
                source_widget = self._create_source_widget(idx, x, y, z, db)
                self.sources_container_layout.addWidget(source_widget)
                self.source_widgets.append(source_widget)
    
    def _create_source_widget(self, idx: int, x: float, y: float, z: float, db: float) -> QWidget:
        """Tek bir ses kaynağı için widget oluştur"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Renk belirleme (dB seviyesine göre)
        if db > -20:
            color = "#e74c3c"  # Kırmızı - Yüksek ses
            icon = "🔴"
        elif db > -30:
            color = "#f39c12"  # Turuncu - Orta ses
            icon = "🟠"
        else:
            color = "#f1c40f"  # Sarı - Düşük ses
            icon = "🟡"
        
        # Başlık
        title_label = QLabel(f"{icon} Kaynak #{idx}")
        title_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
        layout.addWidget(title_label)
        
        # Konum bilgisi
        pos_label = QLabel(f"📍 Pozisyon: ({x:.2f}m, {y:.2f}m, {z:.2f}m)")
        pos_label.setStyleSheet("color: #bbb; font-size: 10px;")
        layout.addWidget(pos_label)
        
        # dB meter (progress bar)
        db_layout = QHBoxLayout()
        db_label = QLabel(f"{db:.1f} dB")
        db_label.setStyleSheet("color: white; font-size: 10px; min-width: 50px;")
        
        db_meter = QProgressBar()
        db_meter.setMinimum(-60)
        db_meter.setMaximum(0)
        db_meter.setValue(int(db))
        db_meter.setTextVisible(False)
        db_meter.setMaximumHeight(15)
        db_meter.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #1e1e1e;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)
        
        db_layout.addWidget(db_meter, stretch=3)
        db_layout.addWidget(db_label)
        layout.addLayout(db_layout)
        
        # Widget stil
        widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #444;
                border-radius: 5px;
            }
        """)
        
        return widget
    
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
        selected_video = self.video_device_combo.currentText()
        if "Webcam 0" in selected_video:
            video_device_id = 0
        elif "Webcam 1" in selected_video:
            video_device_id = 1
        elif "Webcam 2" in selected_video:
            video_device_id = 2
        else:
            video_device_id = 0  # Default
        
        logger.info(f"Video cihaz açılıyor: {selected_video} (ID: {video_device_id})")
        
        # Webcam'i aç
        self.video_capture = cv2.VideoCapture(video_device_id)
        
        if not self.video_capture.isOpened():
            logger.error(f"Webcam açılamadı! (ID: {video_device_id})")
            QMessageBox.warning(self, "Webcam Hatası", 
                               f"Video cihazı açılamadı! (ID: {video_device_id})\nLütfen kamera bağlantısını kontrol edin.")
            self.video_status_label.setText("🔴 Video: Hata")
            self.video_capture = None
        else:
            # Kamera bilgilerini al
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
            
            self.video_status_label.setText(f"🟢 Video: Bağlı ({width}x{height}@{fps}fps)")
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
        
        self.audio_status_label.setText("🔴 Audio: Bağlı değil")
        self.video_status_label.setText("🔴 Video: Bağlı değil")
        
        self.statusbar.showMessage("Sistem durduruldu")
        self._set_placeholder_image()
    
    def _start_audio_stream(self):
        """Audio stream thread'ini başlat"""
        try:
            # Seçili audio cihazını al
            selected_audio = self.audio_device_combo.currentText()
            
            # UMA-16'yı bul
            from audio.device_test import find_uma16_device
            device_index = find_uma16_device()
            
            if device_index is None:
                logger.error("UMA-16 cihazı bulunamadı!")
                QMessageBox.warning(self, "Audio Hatası", 
                                   "UMA-16 cihazı bulunamadı!\nLütfen cihaz bağlantısını kontrol edin.")
                self.audio_status_label.setText("🔴 Audio: Cihaz bulunamadı")
                return
            
            logger.info(f"UMA-16 bulundu: Device #{device_index}")
            
            # Audio thread oluştur
            self.audio_thread = AudioStreamThread(
                device_index=device_index,
                sample_rate=48000,
                buffer_size=4096,
                num_channels=16,
                buffer_duration=5.0,
                gain=10.0  # Digital gain (10x güçlendirme)
            )
            
            # Sinyalleri bağla
            self.audio_thread.channelLevelsReady.connect(self._update_vu_meters)
            self.audio_thread.errorOccurred.connect(self._on_audio_error)
            
            # Thread'i başlat
            self.audio_thread.start()
            
            self.audio_status_label.setText("🟢 Audio: UMA-16 (16ch @ 48kHz)")
            logger.info("Audio stream başlatıldı")
            
        except Exception as e:
            error_msg = f"Audio stream başlatılamadı: {str(e)}"
            logger.error(error_msg)
            QMessageBox.warning(self, "Audio Hatası", error_msg)
            self.audio_status_label.setText("🔴 Audio: Hata")
    
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
        self.audio_status_label.setText("🔴 Audio: Hata")
    
    def toggle_recording(self):
        """Kayıt toggle"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Kaydı başlat"""
        logger.info("Kayıt başlatıldı")
        self.is_recording = True
        self.record_btn.setText("⏹️ KAYIT DURDUR")
        self.record_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.statusbar.showMessage("Kayıt yapılıyor...")
    
    def stop_recording(self):
        """Kaydı durdur"""
        logger.info("Kayıt durduruldu")
        self.is_recording = False
        self.record_btn.setText("🔴 KAYIT BAŞLAT")
        self.record_btn.setStyleSheet("")
        self.statusbar.showMessage("Kayıt durduruldu")
    
    def take_snapshot(self):
        """Anlık görüntü al"""
        logger.info("Snapshot alındı")
        QMessageBox.information(self, "Snapshot", 
                               "Anlık görüntü kaydedildi!")
    
    def on_freq_range_changed(self, min_val, max_val):
        """Frekans aralığı değiştiğinde"""
        logger.debug(f"Frekans aralığı: {min_val} - {max_val} Hz")
    
    def on_db_range_changed(self, min_val, max_val):
        """dB aralığı değiştiğinde"""
        logger.debug(f"dB aralığı: {min_val} - {max_val} dB")
    
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
        
        # Video frame al
        if self.video_capture is not None and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                # Frame'i göster
                self._display_image(frame)
                self.frame_count += 1
        
        # FPS hesapla (basitleştirilmiş)
        fps = int(1000 / 33)  # ~30 FPS
        self.fps_label.setText(f"FPS: {fps}")
        
        # CPU kullanımı (simüle - gerçek değer için psutil gerekir)
        self.cpu_label.setText(f"CPU: 15%")
        
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
    
    # Stil ayarları
    app.setStyle('Fusion')
    
    # Ana pencereyi oluştur
    window = AcousticCameraGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
