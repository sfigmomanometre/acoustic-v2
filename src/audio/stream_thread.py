"""
Audio Stream Thread - Real-time UMA-16 Audio Capture
QThread tabanlı sürekli ses akışı yakalama
"""

import logging
import numpy as np
import sounddevice as sd
from collections import deque
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class AudioStreamThread(QThread):
    """
    Real-time audio streaming thread
    
    Signals:
        audioDataReady: (audio_data, sample_rate) - Yeni audio verisi hazır
        channelLevelsReady: (rms_levels) - Kanal seviyeleri (16 değer, 0-1 arası)
        errorOccurred: (error_msg) - Hata oluştu
    """
    
    audioDataReady = Signal(np.ndarray, int)  # (audio_chunk, sample_rate)
    channelLevelsReady = Signal(list)  # [ch1_level, ch2_level, ..., ch16_level]
    errorOccurred = Signal(str)
    
    def __init__(self, 
                 device_index: int = 0,
                 sample_rate: int = 48000,
                 buffer_size: int = 4096,
                 num_channels: int = 16,
                 buffer_duration: float = 5.0,
                 gain: float = 10.0):
        """
        Args:
            device_index: Audio cihaz index (UMA-16)
            sample_rate: Örnekleme hızı (Hz)
            buffer_size: Callback buffer boyutu (samples)
            num_channels: Kanal sayısı (16 for UMA-16)
            buffer_duration: Circular buffer süresi (saniye)
            gain: Sinyal güçlendirme çarpanı (digital gain)
        """
        super().__init__()
        
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.gain = gain
        
        # Circular buffer (ringbuffer) - son N saniyeyi tutar
        self.buffer_samples = int(sample_rate * buffer_duration)
        self.audio_buffer = deque(maxlen=self.buffer_samples)
        
        # InputStream objesi
        self.stream = None
        
        # Thread kontrol
        self._running = False
        
        # Peak hold için
        self.peak_levels = np.zeros(num_channels)
        self.peak_decay = 0.92  # Her frame'de %92'ye düş (daha hızlı düşme, ama hala smooth)
        
        logger.info(f"AudioStreamThread initialized: {sample_rate}Hz, {num_channels}ch, buffer={buffer_size}, gain={gain}x")
    
    def run(self):
        """Thread ana döngüsü"""
        try:
            self._running = True
            logger.info("Audio stream başlatılıyor...")
            
            # sounddevice InputStream oluştur
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=self.num_channels,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                callback=self._audio_callback,
                dtype='float32'
            )
            
            with self.stream:
                logger.info("Audio stream aktif")
                # Thread durdurulana kadar bekle
                while self._running:
                    self.msleep(10)  # 10ms sleep
            
            logger.info("Audio stream durduruldu")
            
        except Exception as e:
            error_msg = f"Audio stream hatası: {str(e)}"
            logger.error(error_msg)
            self.errorOccurred.emit(error_msg)
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice callback fonksiyonu
        
        Args:
            indata: Audio verisi (frames x channels) numpy array
            frames: Frame sayısı
            time_info: Timing bilgisi
            status: Callback durumu
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Audio verisini kopyala (callback thread-safe değil)
        audio_chunk = indata.copy()
        
        # Circular buffer'a ekle
        for sample in audio_chunk:
            self.audio_buffer.append(sample)
        
        # RMS seviyeleri hesapla (her kanal için)
        rms_levels = self._calculate_rms(audio_chunk)
        
        # Peak hold güncelle
        self.peak_levels = np.maximum(self.peak_levels * self.peak_decay, rms_levels)
        
        # Sinyalleri emit et
        self.audioDataReady.emit(audio_chunk, self.sample_rate)
        self.channelLevelsReady.emit(self.peak_levels.tolist())
    
    def _calculate_rms(self, audio_data: np.ndarray) -> np.ndarray:
        """
        RMS (Root Mean Square) hesapla - her kanal için
        
        Args:
            audio_data: (frames, channels) shape
            
        Returns:
            (channels,) shape - 0-1 arası normalize RMS değerleri
        """
        # Her kanal için RMS
        rms = np.sqrt(np.mean(audio_data ** 2, axis=0))
        
        # Digital gain uygula (düşük sinyalleri güçlendir)
        rms = rms * self.gain
        
        # 0-1 arasına normalize et (clipping ile)
        rms = np.clip(rms, 0.0, 1.0)
        
        return rms
    
    def stop(self):
        """Thread'i durdur"""
        logger.info("Audio stream durduruluyor...")
        self._running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_buffer_data(self, duration: float = 1.0) -> np.ndarray:
        """
        Circular buffer'dan son N saniyeyi al
        
        Args:
            duration: İstenen süre (saniye)
            
        Returns:
            (samples, channels) shape numpy array
        """
        num_samples = int(self.sample_rate * duration)
        
        if len(self.audio_buffer) < num_samples:
            # Yeterli veri yoksa mevcut tüm veriyi al
            num_samples = len(self.audio_buffer)
        
        if num_samples == 0:
            return np.zeros((0, self.num_channels))
        
        # Son N sample'ı al
        buffer_list = list(self.audio_buffer)
        recent_data = buffer_list[-num_samples:]
        
        return np.array(recent_data)
    
    def get_db_levels(self) -> np.ndarray:
        """
        RMS seviyelerini dB cinsinden döndür
        
        Returns:
            (channels,) shape - dB değerleri (-60 to 0 range)
        """
        # RMS'i dB'ye çevir
        # dB = 20 * log10(rms / reference)
        # reference = 1.0 (full scale)
        
        db_levels = 20 * np.log10(self.peak_levels + 1e-10)  # +epsilon log(0) için
        
        # -60 ile 0 arasına kliple
        db_levels = np.clip(db_levels, -60, 0)
        
        return db_levels
