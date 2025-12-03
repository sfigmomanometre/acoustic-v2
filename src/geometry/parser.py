"""
Mikrofon Geometrisi Parser
XML formatındaki mikrofon pozisyonlarını okuyup Acoular MicGeom objesine dönüştürür.
"""

import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

# Acoular import - eğer kurulu değilse uyarı ver
try:
    from acoular import MicGeom
    ACOULAR_AVAILABLE = True
except ImportError:
    ACOULAR_AVAILABLE = False
    logging.warning("Acoular kütüphanesi bulunamadı. 'pip install acoular' ile kurabilirsiniz.")


class MicGeometryParser:
    """
    XML formatındaki mikrofon dizisi geometrisini parse eder.
    
    XML Format Örneği:
    <MicArray>
        <Mic id="0" x="0.0" y="0.044" z="0.0"/>
        <Mic id="1" x="0.031" y="0.031" z="0.0"/>
        ...
    </MicArray>
    """
    
    def __init__(self, xml_path: str):
        """
        Args:
            xml_path: XML dosyasının yolu
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML dosyası bulunamadı: {xml_path}")
        
        self.positions = None
        self.num_mics = 0
        self.logger = logging.getLogger(__name__)
        
    def parse(self) -> np.ndarray:
        """
        XML dosyasını parse edip mikrofon pozisyonlarını döndürür.
        
        Returns:
            np.ndarray: (3, num_mics) şeklinde pozisyon matrisi
                       [x_positions]
                       [y_positions]  
                       [z_positions]
        """
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            
            # Mikrofon elemanlarını bul (hem <Mic> hem <pos> tag'lerini destekle)
            mic_elements = root.findall('.//Mic')
            if not mic_elements:
                mic_elements = root.findall('.//pos')  # Alternatif format
            
            if not mic_elements:
                raise ValueError("XML'de mikrofon elemanları bulunamadı (Mic veya pos)")
            
            self.num_mics = len(mic_elements)
            self.logger.info(f"{self.num_mics} mikrofon pozisyonu bulundu")
            
            # Pozisyonları array'e çevir
            positions_list = []
            
            for mic in mic_elements:
                # ID veya Name
                mic_id = mic.get('id', mic.get('Name', 'unknown'))
                x = float(mic.get('x', 0.0))
                y = float(mic.get('y', 0.0))
                z = float(mic.get('z', 0.0))
                
                positions_list.append([x, y, z])
                self.logger.debug(f"Mic {mic_id}: ({x:.4f}, {y:.4f}, {z:.4f})")
            
            # (num_mics, 3) -> (3, num_mics) formatına çevir (Acoular formatı)
            self.positions = np.array(positions_list).T
            
            self.logger.info(f"Geometri başarıyla parse edildi. Shape: {self.positions.shape}")
            return self.positions
            
        except ET.ParseError as e:
            self.logger.error(f"XML parse hatası: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Geometri parse hatası: {e}")
            raise
    
    def to_acoular(self) -> Optional['MicGeom']:
        """
        Acoular MicGeom objesi oluşturur.
        
        Returns:
            MicGeom: Acoular mikrofon geometrisi objesi
        """
        if not ACOULAR_AVAILABLE:
            self.logger.error("Acoular kurulu değil!")
            return None
        
        if self.positions is None:
            self.parse()
        
        # Acoular MicGeom objesi oluştur
        mic_geom = MicGeom()
        mic_geom.mpos_tot = self.positions
        
        self.logger.info(f"Acoular MicGeom objesi oluşturuldu: {self.num_mics} mikrofon")
        return mic_geom
    
    def get_array_info(self) -> dict:
        """
        Mikrofon dizisi hakkında bilgi döndürür.
        
        Returns:
            dict: Geometri bilgileri
        """
        if self.positions is None:
            self.parse()
        
        center = np.mean(self.positions, axis=1)
        
        # Merkezden maksimum mesafe (yaklaşık yarıçap)
        distances = np.linalg.norm(self.positions - center[:, np.newaxis], axis=0)
        max_distance = np.max(distances)
        min_distance = np.min(distances)
        
        # Bounding box
        bbox = {
            'x_min': np.min(self.positions[0, :]),
            'x_max': np.max(self.positions[0, :]),
            'y_min': np.min(self.positions[1, :]),
            'y_max': np.max(self.positions[1, :]),
            'z_min': np.min(self.positions[2, :]),
            'z_max': np.max(self.positions[2, :]),
        }
        
        return {
            'num_microphones': self.num_mics,
            'center': center.tolist(),
            'max_radius': float(max_distance),
            'min_radius': float(min_distance),
            'bounding_box': bbox,
            'is_planar': self._is_planar(),
            'estimated_type': self._estimate_array_type()
        }
    
    def _is_planar(self, tolerance: float = 1e-4) -> bool:
        """Dizinin düzlemsel olup olmadığını kontrol eder."""
        if self.positions is None:
            return False
        
        z_range = np.max(self.positions[2, :]) - np.min(self.positions[2, :])
        return z_range < tolerance
    
    def _estimate_array_type(self) -> str:
        """
        Mikrofon dizisinin tipini tahmin eder.
        
        Returns:
            str: 'circular', 'linear', 'grid', 'random'
        """
        if self.positions is None:
            return "unknown"
        
        # Basit bir tahmin: mikrofonlar arası mesafelerin std'sine bak
        center = np.mean(self.positions, axis=1)
        distances = np.linalg.norm(self.positions - center[:, np.newaxis], axis=0)
        
        # Eğer tüm mesafeler birbirine yakınsa -> dairesel
        if np.std(distances) < 0.01:
            return "circular"
        
        # Eğer z değişimi yoksa ve x veya y sabit -> linear
        if self._is_planar():
            x_range = np.max(self.positions[0, :]) - np.min(self.positions[0, :])
            y_range = np.max(self.positions[1, :]) - np.min(self.positions[1, :])
            
            if min(x_range, y_range) < 0.01:
                return "linear"
            
            return "grid"
        
        return "3d_array"
    
    def visualize(self, save_path: Optional[str] = None):
        """
        Mikrofon dizisini 3D olarak görselleştirir.
        
        Args:
            save_path: Kaydedilecek dosya yolu (None ise sadece göster)
        """
        if self.positions is None:
            self.parse()
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 5))
            
            # 3D görünüm
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(self.positions[0, :], 
                       self.positions[1, :], 
                       self.positions[2, :],
                       c='red', s=100, marker='o', label='Mikrofonlar')
            
            # Mikrofon numaralarını ekle
            for i in range(self.num_mics):
                ax1.text(self.positions[0, i], 
                        self.positions[1, i], 
                        self.positions[2, i], 
                        f'  {i}', fontsize=8)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title('Mikrofon Dizisi - 3D Görünüm')
            ax1.legend()
            
            # 2D Top view (XY düzlemi)
            ax2 = fig.add_subplot(122)
            ax2.scatter(self.positions[0, :], 
                       self.positions[1, :],
                       c='blue', s=100, marker='o', label='Mikrofonlar')
            
            for i in range(self.num_mics):
                ax2.text(self.positions[0, i], 
                        self.positions[1, i], 
                        f'  {i}', fontsize=8)
            
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Mikrofon Dizisi - Üstten Görünüm')
            ax2.axis('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Görselleştirme kaydedildi: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib kurulu değil, görselleştirme yapılamadı")


# Modül test fonksiyonu
if __name__ == "__main__":
    # Basit test
    logging.basicConfig(level=logging.INFO)
    
    # XML dosyası yolu (proje kök dizininden)
    xml_path = "config/micgeom.xml"
    
    parser = MicGeometryParser(xml_path)
    positions = parser.parse()
    
    print("\n=== Mikrofon Dizisi Bilgileri ===")
    info = parser.get_array_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== Pozisyon Matrisi ===")
    print(f"Shape: {positions.shape}")
    print(positions)
    
    # Görselleştir
    parser.visualize()
    
    # Acoular objesi oluştur
    if ACOULAR_AVAILABLE:
        mic_geom = parser.to_acoular()
        print(f"\nAcoular MicGeom objesi: {mic_geom}")
