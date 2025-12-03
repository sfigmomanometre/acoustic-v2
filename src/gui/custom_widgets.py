"""
Custom Qt Widgets
Özel GUI bileşenleri
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont


class DoubleRangeSlider(QWidget):
    """
    Çift kollu range slider - tek slider üzerinde iki handle
    """
    
    # Signal: (min_value, max_value)
    rangeChanged = Signal(int, int)
    
    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        
        self._minimum = minimum
        self._maximum = maximum
        self._low = minimum
        self._high = maximum
        
        self._pressing_control = None  # 'low', 'high', None
        self._track_height = 8
        self._handle_radius = 8
        
        self.setMinimumHeight(50)
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Custom çizim"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Boyutları hesapla
        width = self.width()
        height = self.height()
        track_y = height // 2
        margin = self._handle_radius + 2
        track_width = width - 2 * margin
        
        # Track arka plan (gri)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(180, 180, 180))
        track_rect = QRect(margin, track_y - self._track_height // 2, 
                          track_width, self._track_height)
        painter.drawRoundedRect(track_rect, 4, 4)
        
        # Seçili aralık (mavi)
        low_pos = self._value_to_pos(self._low, margin, track_width)
        high_pos = self._value_to_pos(self._high, margin, track_width)
        
        painter.setBrush(QColor(41, 128, 185))
        selected_rect = QRect(low_pos, track_y - self._track_height // 2,
                             high_pos - low_pos, self._track_height)
        painter.drawRoundedRect(selected_rect, 4, 4)
        
        # Low handle (sol kol)
        painter.setBrush(QColor(52, 152, 219))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPoint(low_pos, track_y), 
                           self._handle_radius, self._handle_radius)
        
        # High handle (sağ kol)
        painter.drawEllipse(QPoint(high_pos, track_y),
                           self._handle_radius, self._handle_radius)
        
        # Değer yazıları - kontur ile daha görünür
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        low_text = str(self._low)
        high_text = str(self._high)
        
        # Sol kol değeri (üstte) - kontur + beyaz yazı
        low_rect = QRect(low_pos - 40, 2, 80, 18)
        # Siyah kontur (outline)
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    painter.drawText(low_rect.adjusted(dx, dy, dx, dy), 
                                    Qt.AlignCenter, low_text)
        # Beyaz yazı
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(low_rect, Qt.AlignCenter, low_text)
        
        # Sağ kol değeri (altta) - kontur + beyaz yazı
        high_rect = QRect(high_pos - 40, height - 20, 80, 18)
        # Siyah kontur
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    painter.drawText(high_rect.adjusted(dx, dy, dx, dy), 
                                    Qt.AlignCenter, high_text)
        # Beyaz yazı
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(high_rect, Qt.AlignCenter, high_text)
    
    def mousePressEvent(self, event):
        """Mouse basıldığında"""
        if event.button() == Qt.LeftButton:
            pos = event.pos().x()
            margin = self._handle_radius + 2
            track_width = self.width() - 2 * margin
            track_y = self.height() // 2
            
            low_pos = self._value_to_pos(self._low, margin, track_width)
            high_pos = self._value_to_pos(self._high, margin, track_width)
            
            # Hangi handle'a daha yakınız?
            dist_low = abs(pos - low_pos)
            dist_high = abs(pos - high_pos)
            
            if dist_low < dist_high and dist_low < self._handle_radius * 2:
                self._pressing_control = 'low'
            elif dist_high < self._handle_radius * 2:
                self._pressing_control = 'high'
            else:
                self._pressing_control = None
    
    def mouseReleaseEvent(self, event):
        """Mouse bırakıldığında"""
        self._pressing_control = None
    
    def mouseMoveEvent(self, event):
        """Mouse hareket ettiğinde"""
        if self._pressing_control:
            pos = event.pos().x()
            margin = self._handle_radius + 2
            track_width = self.width() - 2 * margin
            
            value = self._pos_to_value(pos, margin, track_width)
            
            if self._pressing_control == 'low':
                if value <= self._high:
                    self._low = value
                    self.rangeChanged.emit(self._low, self._high)
                    self.update()
            elif self._pressing_control == 'high':
                if value >= self._low:
                    self._high = value
                    self.rangeChanged.emit(self._low, self._high)
                    self.update()
    
    def _value_to_pos(self, value, margin, track_width):
        """Değeri pixel pozisyonuna çevir"""
        ratio = (value - self._minimum) / (self._maximum - self._minimum)
        return int(margin + ratio * track_width)
    
    def _pos_to_value(self, pos, margin, track_width):
        """Pixel pozisyonunu değere çevir"""
        ratio = max(0, min(1, (pos - margin) / track_width))
        return int(self._minimum + ratio * (self._maximum - self._minimum))
    
    def setRange(self, minimum, maximum):
        """Min-max aralığını ayarla"""
        self._minimum = minimum
        self._maximum = maximum
        self.update()
    
    def values(self):
        """Mevcut değerleri döndür"""
        return (self._low, self._high)
    
    def setValues(self, low, high):
        """Değerleri ayarla"""
        self._low = low
        self._high = high
        self.update()
