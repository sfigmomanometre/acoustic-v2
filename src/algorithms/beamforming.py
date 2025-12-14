"""
Beamforming Algorithms for Acoustic Source Localization

Near-field & Far-field beamforming implementations:
- DAS (Delay-and-Sum) / Conventional Beamforming
- MVDR (Minimum Variance Distortionless Response) - future
- MUSIC (Multiple Signal Classification) - future

Author: Acoustic Camera Project
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Try to import joblib for better parallelization
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Get number of CPU cores
N_CORES = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)


@dataclass
class BeamformingConfig:
    """Beamforming configuration parameters"""
    
    # Grid parameters
    grid_size_x: float = 0.6  # meters (60 cm)
    grid_size_y: float = 0.6  # meters (60 cm)
    grid_resolution: float = 0.02  # meters (2 cm)
    focus_distance: float = 1.0  # Z distance in meters
    
    # Frequency parameters
    freq_min: float = 500.0  # Hz
    freq_max: float = 8000.0  # Hz
    freq_bins: int = 10  # Number of frequency bins
    
    # Field type
    field_type: str = 'near-field'  # 'near-field' or 'far-field'
    
    # Speed of sound
    sound_speed: float = 343.0  # m/s at 20°C
    
    def __post_init__(self):
        """Validate configuration"""
        if self.field_type not in ['near-field', 'far-field']:
            raise ValueError(f"field_type must be 'near-field' or 'far-field', got {self.field_type}")
        
        if self.grid_resolution <= 0:
            raise ValueError(f"grid_resolution must be positive, got {self.grid_resolution}")
        
        if self.focus_distance <= 0:
            raise ValueError(f"focus_distance must be positive, got {self.focus_distance}")


def load_mic_geometry(xml_path: str) -> np.ndarray:
    """
    Load microphone positions from XML file
    
    Args:
        xml_path: Path to micgeom.xml file
        
    Returns:
        mic_positions: (n_mics, 3) array of [x, y, z] positions in meters
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # UMA-16v2: Microphone order matches audio channel order
        # Ch1-Ch16 mapping from XML comments
        mic_names = [
            'Mic1', 'Mic2', 'Mic3', 'Mic4',
            'Mic5', 'Mic6', 'Mic7', 'Mic8',
            'Mic9', 'Mic10', 'Mic11', 'Mic12',
            'Mic13', 'Mic14', 'Mic15', 'Mic16'
        ]
        
        mic_positions = []
        for mic_name in mic_names:
            # Find microphone in XML
            mic_elem = root.find(f".//pos[@Name='{mic_name}']")
            if mic_elem is None:
                raise ValueError(f"Microphone {mic_name} not found in {xml_path}")
            
            x = float(mic_elem.get('x'))
            y = float(mic_elem.get('y'))
            z = float(mic_elem.get('z'))
            mic_positions.append([x, y, z])
        
        mic_positions = np.array(mic_positions)
        logger.info(f"Loaded {len(mic_positions)} microphone positions from {xml_path}")
        logger.debug(f"Microphone array bounds: "
                    f"X=[{mic_positions[:, 0].min():.3f}, {mic_positions[:, 0].max():.3f}], "
                    f"Y=[{mic_positions[:, 1].min():.3f}, {mic_positions[:, 1].max():.3f}]")
        
        return mic_positions
    
    except Exception as e:
        logger.error(f"Failed to load microphone geometry: {e}")
        raise


def create_focus_grid(config: BeamformingConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Create 2D grid of focus points in space
    
    Args:
        config: BeamformingConfig with grid parameters
        
    Returns:
        grid_points: (n_points, 3) array of [x, y, z] coordinates
        grid_shape: (n_rows, n_cols) tuple for reshaping power map
    """
    # Calculate number of grid points
    n_x = int(config.grid_size_x / config.grid_resolution) + 1
    n_y = int(config.grid_size_y / config.grid_resolution) + 1
    
    # Create grid centered at origin
    x = np.linspace(-config.grid_size_x / 2, config.grid_size_x / 2, n_x)
    y = np.linspace(-config.grid_size_y / 2, config.grid_size_y / 2, n_y)
    
    # Meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Z coordinate (focus distance)
    Z = np.full_like(X, config.focus_distance)
    
    # Stack into (n_points, 3) array
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    grid_shape = (n_y, n_x)
    
    logger.info(f"Created focus grid: {n_x}x{n_y} = {len(grid_points)} points, "
                f"resolution={config.grid_resolution}m, z={config.focus_distance}m")
    
    return grid_points, grid_shape


def compute_steering_vectors(
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    frequency: float,
    sound_speed: float = 343.0,
    field_type: str = 'near-field'
) -> np.ndarray:
    """
    Compute steering vectors for beamforming
    
    Near-field: Uses spherical wave model (accounts for distance and angle)
    Far-field: Uses plane wave model (accounts for angle only)
    
    Args:
        mic_positions: (n_mics, 3) array of microphone positions
        grid_points: (n_points, 3) array of focus point positions
        frequency: Frequency in Hz
        sound_speed: Speed of sound in m/s
        field_type: 'near-field' or 'far-field'
        
    Returns:
        steering_vectors: (n_points, n_mics) complex array
    """
    n_mics = len(mic_positions)
    n_points = len(grid_points)
    
    # Wavenumber
    k = 2 * np.pi * frequency / sound_speed
    
    if field_type == 'near-field':
        # Spherical wave model
        # For each grid point, compute distance to each microphone
        # Broadcasting: (n_points, 1, 3) - (1, n_mics, 3) = (n_points, n_mics, 3)
        distances = np.linalg.norm(
            grid_points[:, np.newaxis, :] - mic_positions[np.newaxis, :, :],
            axis=2
        )
        
        # Steering vector: exp(+j * k * distance)
        # Positive phase for time-reversal beamforming (compensates for propagation delay)
        # Note: We omit 1/distance normalization - kept constant for all grid points
        steering_vectors = np.exp(1j * k * distances)
        
    else:  # far-field
        # Plane wave model
        # Assume source direction is from origin to grid point
        # Only angle matters, not distance
        directions = grid_points / (np.linalg.norm(grid_points, axis=1, keepdims=True) + 1e-10)
        
        # Time delays based on projection onto direction
        # delays[i, j] = dot(mic_position[j], direction[i]) / c
        delays = np.dot(directions, mic_positions.T) / sound_speed
        
        # Steering vector: exp(-j * 2π * f * delay)
        steering_vectors = np.exp(-1j * 2 * np.pi * frequency * delays)
    
    return steering_vectors


def das_beamformer(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Delay-and-Sum (DAS) Beamformer - Frequency Domain Implementation
    
    Processes multiple frequency bins and averages power across frequency band.
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain microphone signals
        mic_positions: (n_mics, 3) microphone positions in meters
        grid_points: (n_points, 3) focus grid points in meters
        sample_rate: Sampling rate in Hz
        config: BeamformingConfig
        freq_range: Optional (freq_min, freq_max) tuple to override config
        
    Returns:
        power_map: (n_points,) array of acoustic power at each grid point
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # FFT
    fft_data = np.fft.rfft(mic_signals, axis=0)  # (n_freqs, n_mics)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Select frequency bins in range
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    selected_freqs = freqs[freq_mask]
    selected_fft = fft_data[freq_mask, :]  # (n_selected_freqs, n_mics)
    
    if len(selected_freqs) == 0:
        logger.warning(f"No frequencies in range [{f_min}, {f_max}] Hz")
        return np.zeros(n_points)
    
    logger.debug(f"Processing {len(selected_freqs)} frequency bins "
                f"from {selected_freqs[0]:.1f} to {selected_freqs[-1]:.1f} Hz")
    
    # Initialize power map
    power_map = np.zeros(n_points, dtype=np.float64)
    
    # Process each frequency bin
    for freq_idx, freq in enumerate(selected_freqs):
        # Get FFT data for this frequency: (n_mics,)
        X = selected_fft[freq_idx, :]
        
        # Compute steering vectors for this frequency: (n_points, n_mics)
        A = compute_steering_vectors(
            mic_positions, 
            grid_points, 
            freq,
            config.sound_speed,
            config.field_type
        )
        
        # Beamforming: Y = A @ X
        # Y[i] = sum_over_mics(A[i, m] * X[m])
        # Broadcasting: (n_points, n_mics) @ (n_mics,) = (n_points,)
        Y = np.sum(A * X[np.newaxis, :], axis=1)
        
        # Accumulate power
        power_map += np.abs(Y) ** 2
    
    # Average over frequency bins
    power_map /= len(selected_freqs)
    
    elapsed = (time.perf_counter() - start_time) * 1000  # ms
    logger.debug(f"DAS beamforming: {n_points} grid points, "
                f"{len(selected_freqs)} freqs, {elapsed:.2f} ms")
    
    return power_map


def power_to_db(power_map: np.ndarray, reference: float = 1.0) -> np.ndarray:
    """
    Convert power map to dB scale
    
    Args:
        power_map: Linear power values
        reference: Reference power (default: 1.0)
        
    Returns:
        power_db: Power in dB (10 * log10(power / reference))
    """
    # Avoid log(0)
    power_map_safe = np.maximum(power_map, 1e-10)
    power_db = 10 * np.log10(power_map_safe / reference)
    return power_db


def compute_covariance_matrix(
    mic_signals: np.ndarray,
    sample_rate: float,
    freq_range: Tuple[float, float],
    diagonal_loading: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Cross-Spectral Matrix (CSM) from microphone signals.
    
    The CSM is the frequency-domain covariance matrix, computed as:
    R(f) = X(f) @ X(f)^H
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain microphone signals
        sample_rate: Sampling rate in Hz
        freq_range: (freq_min, freq_max) frequency range of interest
        diagonal_loading: Small value added to diagonal for numerical stability
        
    Returns:
        csm: (n_freqs, n_mics, n_mics) Cross-Spectral Matrix for each frequency
        freqs: (n_freqs,) frequency bins
        fft_data: (n_freqs, n_mics) FFT of signals
    """
    n_samples, n_mics = mic_signals.shape
    
    # FFT
    fft_data = np.fft.rfft(mic_signals, axis=0)  # (n_freqs, n_mics)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    # Select frequency range
    freq_min, freq_max = freq_range
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    selected_freqs = freqs[freq_mask]
    selected_fft = fft_data[freq_mask, :]  # (n_selected_freqs, n_mics)
    
    n_freqs = len(selected_freqs)
    
    # Compute CSM for each frequency bin
    # R(f) = X(f) @ X(f)^H where X is column vector of mic signals
    csm = np.zeros((n_freqs, n_mics, n_mics), dtype=np.complex128)
    
    for i in range(n_freqs):
        X = selected_fft[i, :]  # (n_mics,)
        # Outer product: X @ X^H
        csm[i] = np.outer(X, np.conj(X))
        # Diagonal loading for numerical stability
        csm[i] += diagonal_loading * np.eye(n_mics)
    
    logger.debug(f"Computed CSM: {n_freqs} frequency bins, {n_mics}x{n_mics} matrix")
    
    return csm, selected_freqs, selected_fft


def mvdr_beamformer(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    diagonal_loading: float = 1e-3
) -> np.ndarray:
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer
    Also known as Capon Beamformer.
    
    MVDR minimizes output power while maintaining unity gain in the look direction:
    
    w_MVDR = (R^-1 @ a) / (a^H @ R^-1 @ a)
    P_MVDR = 1 / (a^H @ R^-1 @ a)
    
    Where:
    - R is the covariance (cross-spectral) matrix
    - a is the steering vector
    - w is the weight vector
    
    Advantages over DAS:
    - Higher resolution (narrower main lobe)
    - Better interference rejection
    - Adaptive to the acoustic environment
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain microphone signals
        mic_positions: (n_mics, 3) microphone positions in meters
        grid_points: (n_points, 3) focus grid points in meters
        sample_rate: Sampling rate in Hz
        config: BeamformingConfig
        freq_range: Optional (freq_min, freq_max) tuple to override config
        diagonal_loading: Regularization parameter (higher = more stable, lower = sharper)
        
    Returns:
        power_map: (n_points,) array of acoustic power at each grid point
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute Cross-Spectral Matrix
    csm, selected_freqs, selected_fft = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(selected_freqs) == 0:
        logger.warning(f"No frequencies in range [{f_min}, {f_max}] Hz")
        return np.zeros(n_points)
    
    n_freqs = len(selected_freqs)
    
    logger.debug(f"MVDR: Processing {n_freqs} frequency bins "
                f"from {selected_freqs[0]:.1f} to {selected_freqs[-1]:.1f} Hz")
    
    # Initialize power map
    power_map = np.zeros(n_points, dtype=np.float64)
    
    # Process each frequency bin
    for freq_idx, freq in enumerate(selected_freqs):
        # Get CSM for this frequency
        R = csm[freq_idx]  # (n_mics, n_mics)
        
        # Compute inverse of CSM
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            R_inv = np.linalg.pinv(R)
            logger.warning(f"CSM singular at {freq:.1f} Hz, using pseudo-inverse")
        
        # Compute steering vectors for this frequency: (n_points, n_mics)
        A = compute_steering_vectors(
            mic_positions,
            grid_points,
            freq,
            config.sound_speed,
            config.field_type
        )
        
        # MVDR power for each grid point
        # P_MVDR(θ) = 1 / (a^H @ R^-1 @ a)
        for point_idx in range(n_points):
            a = A[point_idx, :]  # Steering vector for this point (n_mics,)
            
            # a^H @ R^-1 @ a
            denominator = np.real(np.conj(a) @ R_inv @ a)
            
            # Avoid division by zero
            if denominator > 1e-10:
                power_map[point_idx] += 1.0 / denominator
            else:
                power_map[point_idx] += 0.0
    
    # Average over frequency bins
    power_map /= n_freqs
    
    elapsed = (time.perf_counter() - start_time) * 1000  # ms
    logger.debug(f"MVDR beamforming: {n_points} grid points, "
                f"{n_freqs} freqs, {elapsed:.2f} ms")
    
    return power_map


def mvdr_beamformer_fast(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    diagonal_loading: float = 1e-3
) -> np.ndarray:
    """
    Optimized MVDR Beamformer using vectorized operations.
    
    This version processes all grid points simultaneously for each frequency,
    which is significantly faster than the loop-based version.
    
    Args:
        Same as mvdr_beamformer
        
    Returns:
        power_map: (n_points,) array of acoustic power at each grid point
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute Cross-Spectral Matrix
    csm, selected_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(selected_freqs) == 0:
        logger.warning(f"No frequencies in range [{f_min}, {f_max}] Hz")
        return np.zeros(n_points)
    
    n_freqs = len(selected_freqs)
    
    # Initialize power map
    power_map = np.zeros(n_points, dtype=np.float64)
    
    # Process each frequency bin
    for freq_idx, freq in enumerate(selected_freqs):
        R = csm[freq_idx]  # (n_mics, n_mics)
        
        # Compute inverse
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        
        # Steering vectors: (n_points, n_mics)
        A = compute_steering_vectors(
            mic_positions,
            grid_points,
            freq,
            config.sound_speed,
            config.field_type
        )
        
        # Vectorized MVDR: P = 1 / (a^H @ R^-1 @ a) for all points
        # A @ R_inv: (n_points, n_mics) @ (n_mics, n_mics) = (n_points, n_mics)
        AR_inv = A @ R_inv
        
        # Element-wise multiply and sum: (a^H @ R^-1 @ a) for each point
        # Sum of (A_conj * AR_inv) along axis 1
        denominator = np.real(np.sum(np.conj(A) * AR_inv, axis=1))  # (n_points,)
        
        # MVDR power
        valid_mask = denominator > 1e-10
        power_map[valid_mask] += 1.0 / denominator[valid_mask]
    
    # Average over frequency bins
    power_map /= n_freqs
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"MVDR (fast): {n_points} grid points, {n_freqs} freqs, {elapsed:.2f} ms")
    
    return power_map


def music_beamformer(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    n_sources: int = 1,
    diagonal_loading: float = 1e-6
) -> np.ndarray:
    """
    MUSIC (Multiple Signal Classification) Beamformer
    
    MUSIC uses eigenvalue decomposition to separate signal and noise subspaces,
    providing super-resolution source localization.
    
    P_MUSIC(θ) = 1 / (a^H @ E_n @ E_n^H @ a)
    
    Where E_n contains the noise subspace eigenvectors.
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain microphone signals
        mic_positions: (n_mics, 3) microphone positions in meters
        grid_points: (n_points, 3) focus grid points in meters
        sample_rate: Sampling rate in Hz
        config: BeamformingConfig
        freq_range: Optional (freq_min, freq_max) tuple
        n_sources: Estimated number of sources (for subspace separation)
        diagonal_loading: Regularization parameter
        
    Returns:
        power_map: (n_points,) MUSIC pseudo-spectrum
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute CSM
    csm, selected_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(selected_freqs) == 0:
        return np.zeros(n_points)
    
    n_freqs = len(selected_freqs)
    power_map = np.zeros(n_points, dtype=np.float64)
    
    for freq_idx, freq in enumerate(selected_freqs):
        R = csm[freq_idx]
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace: eigenvectors corresponding to smallest eigenvalues
        # Assume n_sources largest eigenvalues correspond to signals
        E_n = eigenvectors[:, n_sources:]  # (n_mics, n_mics - n_sources)
        
        # Steering vectors
        A = compute_steering_vectors(
            mic_positions, grid_points, freq,
            config.sound_speed, config.field_type
        )
        
        # MUSIC pseudo-spectrum: P = 1 / |a^H @ E_n|^2
        # A @ E_n: (n_points, n_mics) @ (n_mics, n_noise) = (n_points, n_noise)
        projection = A @ E_n
        
        # Sum of squared magnitudes along noise subspace
        denominator = np.sum(np.abs(projection) ** 2, axis=1)  # (n_points,)
        
        valid_mask = denominator > 1e-10
        power_map[valid_mask] += 1.0 / denominator[valid_mask]
    
    power_map /= n_freqs
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"MUSIC: {n_points} grid points, {n_freqs} freqs, "
                f"{n_sources} sources, {elapsed:.2f} ms")
    
    return power_map


# ============================================================================
# OPTIMIZED PARALLEL BEAMFORMING IMPLEMENTATIONS
# ============================================================================

def _precompute_distances(mic_positions: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
    """
    Precompute distances from each grid point to each microphone.
    This is reused across all frequencies.
    
    Args:
        mic_positions: (n_mics, 3) microphone positions
        grid_points: (n_points, 3) grid points
        
    Returns:
        distances: (n_points, n_mics) distance matrix
    """
    # Broadcasting: (n_points, 1, 3) - (1, n_mics, 3) = (n_points, n_mics, 3)
    distances = np.linalg.norm(
        grid_points[:, np.newaxis, :] - mic_positions[np.newaxis, :, :],
        axis=2
    )
    return distances


def _compute_steering_vectors_from_distances(
    distances: np.ndarray,
    frequency: float,
    sound_speed: float = 343.0
) -> np.ndarray:
    """
    Compute steering vectors from precomputed distances (near-field only).
    Much faster than recomputing distances every time.
    
    Args:
        distances: (n_points, n_mics) precomputed distances
        frequency: Frequency in Hz
        sound_speed: Speed of sound in m/s
        
    Returns:
        steering_vectors: (n_points, n_mics) complex array
    """
    k = 2 * np.pi * frequency / sound_speed
    return np.exp(1j * k * distances)


def das_beamformer_parallel(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Parallel DAS Beamformer using all CPU cores.
    
    Optimizations:
    1. Precompute distances once (reused for all frequencies)
    2. Process frequency bins in parallel using joblib
    3. Vectorized operations throughout
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain microphone signals
        mic_positions: (n_mics, 3) microphone positions in meters
        grid_points: (n_points, 3) focus grid points in meters
        sample_rate: Sampling rate in Hz
        config: BeamformingConfig
        freq_range: Optional (freq_min, freq_max) tuple
        n_jobs: Number of parallel jobs (-1 = all cores)
        
    Returns:
        power_map: (n_points,) array of acoustic power at each grid point
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    if n_jobs == -1:
        n_jobs = N_CORES
    
    # FFT
    fft_data = np.fft.rfft(mic_signals, axis=0)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Select frequency bins
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    selected_freqs = freqs[freq_mask]
    selected_fft = fft_data[freq_mask, :]
    
    if len(selected_freqs) == 0:
        return np.zeros(n_points)
    
    # Precompute distances (only for near-field)
    if config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    else:
        distances = None
    
    def process_frequency(freq_idx):
        """Process a single frequency bin."""
        freq = selected_freqs[freq_idx]
        X = selected_fft[freq_idx, :]
        
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        Y = np.sum(A * X[np.newaxis, :], axis=1)
        return np.abs(Y) ** 2
    
    # Parallel processing
    if HAS_JOBLIB and len(selected_freqs) > 4:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_frequency)(i) for i in range(len(selected_freqs))
        )
        power_map = np.sum(results, axis=0)
    else:
        # Fallback to sequential if joblib not available
        power_map = np.zeros(n_points, dtype=np.float64)
        for i in range(len(selected_freqs)):
            power_map += process_frequency(i)
    
    power_map /= len(selected_freqs)
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"DAS (parallel, {n_jobs} cores): {n_points} points, "
                f"{len(selected_freqs)} freqs, {elapsed:.2f} ms")
    
    return power_map


def mvdr_beamformer_parallel(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    diagonal_loading: float = 1e-3,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Parallel MVDR Beamformer using all CPU cores.
    
    Optimizations:
    1. Precompute distances once
    2. Process frequency bins in parallel
    3. Vectorized matrix operations
    
    Args:
        Same as mvdr_beamformer_fast plus n_jobs
        
    Returns:
        power_map: (n_points,) array of acoustic power at each grid point
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    if n_jobs == -1:
        n_jobs = N_CORES
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute CSM
    csm, selected_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(selected_freqs) == 0:
        return np.zeros(n_points)
    
    n_freqs = len(selected_freqs)
    
    # Precompute distances
    if config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    else:
        distances = None
    
    def process_frequency(freq_idx):
        """Process a single frequency bin for MVDR."""
        freq = selected_freqs[freq_idx]
        R = csm[freq_idx]
        
        # Compute inverse
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        
        # Steering vectors
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        # Vectorized MVDR
        AR_inv = A @ R_inv
        denominator = np.real(np.sum(np.conj(A) * AR_inv, axis=1))
        
        result = np.zeros(n_points, dtype=np.float64)
        valid_mask = denominator > 1e-10
        result[valid_mask] = 1.0 / denominator[valid_mask]
        return result
    
    # Parallel processing
    if HAS_JOBLIB and n_freqs > 4:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_frequency)(i) for i in range(n_freqs)
        )
        power_map = np.sum(results, axis=0)
    else:
        power_map = np.zeros(n_points, dtype=np.float64)
        for i in range(n_freqs):
            power_map += process_frequency(i)
    
    power_map /= n_freqs
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"MVDR (parallel, {n_jobs} cores): {n_points} points, "
                f"{n_freqs} freqs, {elapsed:.2f} ms")
    
    return power_map


def music_beamformer_parallel(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    n_sources: int = 1,
    diagonal_loading: float = 1e-6,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Parallel MUSIC Beamformer using all CPU cores.
    
    Args:
        Same as music_beamformer plus n_jobs
        
    Returns:
        power_map: (n_points,) MUSIC pseudo-spectrum
    """
    import time
    start_time = time.perf_counter()
    
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    if n_jobs == -1:
        n_jobs = N_CORES
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute CSM
    csm, selected_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(selected_freqs) == 0:
        return np.zeros(n_points)
    
    n_freqs = len(selected_freqs)
    
    # Precompute distances
    if config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    else:
        distances = None
    
    def process_frequency(freq_idx):
        """Process a single frequency bin for MUSIC."""
        freq = selected_freqs[freq_idx]
        R = csm[freq_idx]
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace
        E_n = eigenvectors[:, n_sources:]
        
        # Steering vectors
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        # MUSIC pseudo-spectrum
        projection = A @ E_n
        denominator = np.sum(np.abs(projection) ** 2, axis=1)
        
        result = np.zeros(n_points, dtype=np.float64)
        valid_mask = denominator > 1e-10
        result[valid_mask] = 1.0 / denominator[valid_mask]
        return result
    
    # Parallel processing
    if HAS_JOBLIB and n_freqs > 4:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(process_frequency)(i) for i in range(n_freqs)
        )
        power_map = np.sum(results, axis=0)
    else:
        power_map = np.zeros(n_points, dtype=np.float64)
        for i in range(n_freqs):
            power_map += process_frequency(i)
    
    power_map /= n_freqs
    
    elapsed = (time.perf_counter() - start_time) * 1000
    logger.debug(f"MUSIC (parallel, {n_jobs} cores): {n_points} points, "
                f"{n_freqs} freqs, {n_sources} sources, {elapsed:.2f} ms")
    
    return power_map


# ============================================================================
# ULTRA-FAST BEAMFORMING (Reduced frequency bins, coarser grid)
# ============================================================================

def das_beamformer_realtime(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    max_freq_bins: int = 10,
    distances: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Ultra-fast DAS beamformer for real-time operation.
    
    Optimizations:
    1. Limit number of frequency bins
    2. Accept precomputed distances
    3. Fully vectorized - no loops
    
    Args:
        mic_signals: (n_samples, n_mics) time-domain signals
        mic_positions: (n_mics, 3) microphone positions
        grid_points: (n_points, 3) grid points
        sample_rate: Sampling rate
        config: BeamformingConfig
        freq_range: Frequency range
        max_freq_bins: Maximum frequency bins to process (for speed)
        distances: Precomputed distances (optional, for repeated calls)
        
    Returns:
        power_map: (n_points,) power at each grid point
    """
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # FFT
    fft_data = np.fft.rfft(mic_signals, axis=0)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Select and subsample frequency bins
    freq_mask = (freqs >= f_min) & (freqs <= f_max)
    freq_indices = np.where(freq_mask)[0]
    
    if len(freq_indices) == 0:
        return np.zeros(n_points)
    
    # Subsample to max_freq_bins
    if len(freq_indices) > max_freq_bins:
        step = len(freq_indices) // max_freq_bins
        freq_indices = freq_indices[::step][:max_freq_bins]
    
    selected_freqs = freqs[freq_indices]
    selected_fft = fft_data[freq_indices, :]
    
    # Precompute distances if not provided
    if distances is None and config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    
    # Initialize power map
    power_map = np.zeros(n_points, dtype=np.float64)
    
    # Process all frequencies (vectorized)
    for freq_idx, freq in enumerate(selected_freqs):
        X = selected_fft[freq_idx, :]
        
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        Y = np.sum(A * X[np.newaxis, :], axis=1)
        power_map += np.abs(Y) ** 2
    
    power_map /= len(selected_freqs)
    
    return power_map


def mvdr_beamformer_realtime(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    diagonal_loading: float = 1e-3,
    max_freq_bins: int = 10,
    distances: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Ultra-fast MVDR beamformer for real-time operation.
    
    Args:
        Same as das_beamformer_realtime plus diagonal_loading
        
    Returns:
        power_map: (n_points,) power at each grid point
    """
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute CSM
    csm, all_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(all_freqs) == 0:
        return np.zeros(n_points)
    
    # Subsample frequencies
    if len(all_freqs) > max_freq_bins:
        step = len(all_freqs) // max_freq_bins
        indices = list(range(0, len(all_freqs), step))[:max_freq_bins]
        selected_freqs = all_freqs[indices]
        csm = csm[indices]
    else:
        selected_freqs = all_freqs
    
    n_freqs = len(selected_freqs)
    
    # Precompute distances
    if distances is None and config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    
    power_map = np.zeros(n_points, dtype=np.float64)
    
    for freq_idx, freq in enumerate(selected_freqs):
        R = csm[freq_idx]
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R)
        
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        AR_inv = A @ R_inv
        denominator = np.real(np.sum(np.conj(A) * AR_inv, axis=1))
        
        valid_mask = denominator > 1e-10
        power_map[valid_mask] += 1.0 / denominator[valid_mask]
    
    power_map /= n_freqs
    
    return power_map


def music_beamformer_realtime(
    mic_signals: np.ndarray,
    mic_positions: np.ndarray,
    grid_points: np.ndarray,
    sample_rate: float,
    config: BeamformingConfig,
    freq_range: Optional[Tuple[float, float]] = None,
    n_sources: int = 1,
    diagonal_loading: float = 1e-6,
    max_freq_bins: int = 10,
    distances: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Ultra-fast MUSIC beamformer for real-time operation.
    
    Args:
        Same as das_beamformer_realtime plus n_sources and diagonal_loading
        
    Returns:
        power_map: (n_points,) MUSIC pseudo-spectrum
    """
    n_samples, n_mics = mic_signals.shape
    n_points = len(grid_points)
    
    # Frequency range
    if freq_range is not None:
        f_min, f_max = freq_range
    else:
        f_min, f_max = config.freq_min, config.freq_max
    
    # Compute CSM
    csm, all_freqs, _ = compute_covariance_matrix(
        mic_signals, sample_rate, (f_min, f_max), diagonal_loading
    )
    
    if len(all_freqs) == 0:
        return np.zeros(n_points)
    
    # Subsample frequencies
    if len(all_freqs) > max_freq_bins:
        step = len(all_freqs) // max_freq_bins
        indices = list(range(0, len(all_freqs), step))[:max_freq_bins]
        selected_freqs = all_freqs[indices]
        csm = csm[indices]
    else:
        selected_freqs = all_freqs
    
    n_freqs = len(selected_freqs)
    
    # Precompute distances
    if distances is None and config.field_type == 'near-field':
        distances = _precompute_distances(mic_positions, grid_points)
    
    power_map = np.zeros(n_points, dtype=np.float64)
    
    for freq_idx, freq in enumerate(selected_freqs):
        R = csm[freq_idx]
        
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        E_n = eigenvectors[:, n_sources:]
        
        if config.field_type == 'near-field':
            A = _compute_steering_vectors_from_distances(distances, freq, config.sound_speed)
        else:
            A = compute_steering_vectors(mic_positions, grid_points, freq,
                                        config.sound_speed, config.field_type)
        
        projection = A @ E_n
        denominator = np.sum(np.abs(projection) ** 2, axis=1)
        
        valid_mask = denominator > 1e-10
        power_map[valid_mask] += 1.0 / denominator[valid_mask]
    
    power_map /= n_freqs
    
    return power_map


def normalize_power_map(
    power_map: np.ndarray,
    db_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Normalize power map to [0, 1] range for visualization
    
    Args:
        power_map: Power map (linear or dB scale)
        db_range: Optional (min_db, max_db) for clipping. If None, use percentiles.
        
    Returns:
        normalized: Values in [0, 1]
    """
    # Convert to dB if needed
    if power_map.max() > 100:  # Heuristic: likely linear scale
        power_db = power_to_db(power_map)
    else:
        power_db = power_map
    
    # Determine range
    if db_range is not None:
        vmin, vmax = db_range
    else:
        # Adaptive: use percentiles
        vmin = np.percentile(power_db, 10)
        vmax = np.percentile(power_db, 95)
    
    # Clip and normalize
    power_clipped = np.clip(power_db, vmin, vmax)
    normalized = (power_clipped - vmin) / (vmax - vmin + 1e-6)
    
    return normalized


# ============================================================================
# Test and Validation Functions
# ============================================================================

def create_synthetic_signal(
    mic_positions: np.ndarray,
    source_position: np.ndarray,
    frequency: float,
    sample_rate: float,
    n_samples: int,
    sound_speed: float = 343.0,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Create synthetic microphone signals from a point source
    
    Args:
        mic_positions: (n_mics, 3) microphone positions
        source_position: (3,) source position [x, y, z]
        frequency: Source frequency in Hz
        sample_rate: Sampling rate in Hz
        n_samples: Number of samples
        sound_speed: Speed of sound in m/s
        noise_level: Noise amplitude (relative to signal)
        
    Returns:
        signals: (n_samples, n_mics) synthetic signals
    """
    n_mics = len(mic_positions)
    
    # Compute distances from source to each microphone
    distances = np.linalg.norm(mic_positions - source_position, axis=1)
    
    # Time delays (propagation time)
    delays = distances / sound_speed
    
    # Amplitude decay (1/r for spherical spreading)
    amplitudes = 1.0 / (distances + 0.1)  # Avoid division by zero
    
    # Generate time vector
    t = np.arange(n_samples) / sample_rate
    
    # Generate signals for each microphone
    signals = np.zeros((n_samples, n_mics))
    
    for i in range(n_mics):
        # Phase-shifted sinusoid
        phase = 2 * np.pi * frequency * (t - delays[i])
        signals[:, i] = amplitudes[i] * np.sin(phase)
    
    # Add noise
    if noise_level > 0:
        noise = noise_level * np.random.randn(n_samples, n_mics)
        signals += noise
    
    return signals


def test_beamforming(
    source_position: np.ndarray = np.array([0.0, 0.0, 1.0]),
    frequency: float = 2000.0,
    plot: bool = False
):
    """
    Test beamforming with synthetic data
    
    Args:
        source_position: Known source position [x, y, z]
        frequency: Test frequency in Hz
        plot: Whether to plot results (requires matplotlib)
    """
    logger.info("=" * 60)
    logger.info("BEAMFORMING TEST")
    logger.info("=" * 60)
    
    # Load microphone geometry
    xml_path = Path(__file__).parent.parent.parent / 'micgeom.xml'
    mic_positions = load_mic_geometry(str(xml_path))
    
    # Configuration
    config = BeamformingConfig(
        grid_size_x=0.6,
        grid_size_y=0.6,
        grid_resolution=0.02,
        focus_distance=source_position[2],
        freq_min=frequency - 100,
        freq_max=frequency + 100,
        field_type='near-field'
    )
    
    # Create focus grid
    grid_points, grid_shape = create_focus_grid(config)
    
    # Generate synthetic signals
    sample_rate = 48000
    n_samples = 4096
    
    logger.info(f"Generating synthetic signal from source at {source_position} @ {frequency} Hz")
    
    signals = create_synthetic_signal(
        mic_positions,
        source_position,
        frequency,
        sample_rate,
        n_samples,
        noise_level=0.05
    )
    
    # Run beamformer
    logger.info("Running DAS beamformer...")
    power_map = das_beamformer(
        signals,
        mic_positions,
        grid_points,
        sample_rate,
        config
    )
    
    # Find peak
    peak_idx = np.argmax(power_map)
    peak_position = grid_points[peak_idx]
    peak_power_db = power_to_db(power_map)[peak_idx]
    
    # Calculate localization error
    error = np.linalg.norm(peak_position - source_position)
    
    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info(f"  True source position: {source_position}")
    logger.info(f"  Detected peak position: {peak_position}")
    logger.info(f"  Localization error: {error * 100:.2f} cm")
    logger.info(f"  Peak power: {peak_power_db:.1f} dB")
    logger.info("=" * 60)
    
    # Optional: plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            
            power_db = power_to_db(power_map)
            power_grid = power_db.reshape(grid_shape)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extent for imshow
            extent = [
                -config.grid_size_x / 2, config.grid_size_x / 2,
                -config.grid_size_y / 2, config.grid_size_y / 2
            ]
            
            im = ax.imshow(
                power_grid,
                origin='lower',
                extent=extent,
                cmap='jet',
                aspect='equal'
            )
            
            # Plot microphones
            ax.scatter(mic_positions[:, 0], mic_positions[:, 1],
                      c='white', marker='o', s=50, edgecolors='black',
                      label='Microphones', zorder=10)
            
            # Plot true source
            ax.scatter(source_position[0], source_position[1],
                      c='lime', marker='*', s=300, edgecolors='black',
                      label='True Source', zorder=11)
            
            # Plot detected peak
            ax.scatter(peak_position[0], peak_position[1],
                      c='red', marker='x', s=200, linewidths=3,
                      label='Detected Peak', zorder=12)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'DAS Beamforming Test - {frequency} Hz\n'
                        f'Error: {error * 100:.2f} cm')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(im, ax=ax, label='Power (dB)')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    return error < 0.05  # Pass if error < 5 cm


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    success = test_beamforming(
        source_position=np.array([0.1, -0.15, 1.0]),  # 10cm right, 15cm down, 1m away
        frequency=2000.0,
        plot=True  # Set to True if matplotlib is available
    )
    
    if success:
        logger.info("✅ TEST PASSED")
    else:
        logger.warning("⚠️ TEST FAILED - Localization error too large")
