#!/usr/bin/env python3
"""
UMA-16 Akustik Kamera GUI Başlatıcı
"""

import sys
import os
import logging
from pathlib import Path

# macOS: OpenGL widget'larının Qt layer içinde render edilmesi için
os.environ.setdefault('QT_MAC_WANTS_LAYER', '1')

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from gui.main_window import main

if __name__ == '__main__':
    main()
