"""
Custom Qt Widgets
Özel GUI bileşenleri
"""

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont


class DoubleRangeSlider(QWidget):
    """
    Çift kollu range slider - HORIZONTAL layout with values on sides
    Layout: [min_value] ---[handle]=====[handle]--- [max_value]
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
        self._track_height = 6
        self._handle_radius = 7
        
        # Label margins on left/right for value display
        self._label_width = 45  # Width reserved for value labels on each side
        
        self.setMinimumHeight(30)
        self.setMaximumHeight(35)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        
    def paintEvent(self, event):
        """Custom çizim - horizontal layout with side labels"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        track_y = height // 2
        
        # Track area (between the two value labels)
        track_start = self._label_width
        track_end = width - self._label_width
        track_width = track_end - track_start
        
        # Draw value labels on sides
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        
        # Left label (low value)
        painter.setPen(QColor(100, 180, 255))
        low_rect = QRect(0, 0, self._label_width - 5, height)
        painter.drawText(low_rect, Qt.AlignRight | Qt.AlignVCenter, str(self._low))
        
        # Right label (high value)
        painter.setPen(QColor(255, 180, 100))
        high_rect = QRect(width - self._label_width + 5, 0, self._label_width - 5, height)
        painter.drawText(high_rect, Qt.AlignLeft | Qt.AlignVCenter, str(self._high))
        
        # Track background (gray)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(80, 80, 80))
        track_rect = QRect(track_start, track_y - self._track_height // 2, 
                          track_width, self._track_height)
        painter.drawRoundedRect(track_rect, 3, 3)
        
        # Calculate handle positions
        low_pos = self._value_to_pos(self._low, track_start, track_width)
        high_pos = self._value_to_pos(self._high, track_start, track_width)
        
        # Selected range (blue)
        painter.setBrush(QColor(41, 128, 185))
        selected_rect = QRect(low_pos, track_y - self._track_height // 2,
                             high_pos - low_pos, self._track_height)
        painter.drawRoundedRect(selected_rect, 3, 3)
        
        # Low handle (left - blue)
        painter.setBrush(QColor(52, 152, 219))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPoint(low_pos, track_y), 
                           self._handle_radius, self._handle_radius)
        
        # High handle (right - orange)
        painter.setBrush(QColor(230, 126, 34))
        painter.drawEllipse(QPoint(high_pos, track_y),
                           self._handle_radius, self._handle_radius)
    
    def mousePressEvent(self, event):
        """Mouse basıldığında"""
        if event.button() == Qt.LeftButton:
            pos = event.pos().x()
            track_start = self._label_width
            track_width = self.width() - 2 * self._label_width
            
            low_pos = self._value_to_pos(self._low, track_start, track_width)
            high_pos = self._value_to_pos(self._high, track_start, track_width)
            
            dist_low = abs(pos - low_pos)
            dist_high = abs(pos - high_pos)
            
            if dist_low < dist_high and dist_low < self._handle_radius * 3:
                self._pressing_control = 'low'
            elif dist_high < self._handle_radius * 3:
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
            track_start = self._label_width
            track_width = self.width() - 2 * self._label_width
            
            value = self._pos_to_value(pos, track_start, track_width)
            
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
    
    def _value_to_pos(self, value, track_start, track_width):
        """Değeri pixel pozisyonuna çevir"""
        ratio = (value - self._minimum) / (self._maximum - self._minimum + 1e-6)
        return int(track_start + ratio * track_width)
    
    def _pos_to_value(self, pos, track_start, track_width):
        """Pixel pozisyonunu değere çevir"""
        ratio = max(0, min(1, (pos - track_start) / (track_width + 1e-6)))
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

