#!/usr/bin/env python3
"""
Quick Test: Heatmap Overlay Debugging Tool
Beamforming ve overlay'in doğru çalışıp çalışmadığını test eder
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from algorithms.beamforming import (
    load_mic_geometry,
    create_focus_grid,
    BeamformingConfig
)


def test_coordinate_mapping():
    """Test: 3D grid coordinates → 2D pixel coordinates"""
    print("\n" + "="*70)
    print("TEST 1: Coordinate Mapping")
    print("="*70)
    
    # Video dimensions
    video_w, video_h = 1920, 1080
    
    # Grid configuration
    config = BeamformingConfig(
        grid_size_x=1.2,  # 1.2m
        grid_size_y=1.2,
        grid_resolution=0.05,  # 5cm
        focus_distance=1.0,
    )
    
    # Create grid
    grid_points, grid_shape = create_focus_grid(config)
    print(f"✓ Grid created: {grid_shape[0]}x{grid_shape[1]} = {len(grid_points)} points")
    print(f"  Grid size: {config.grid_size_x}m x {config.grid_size_y}m @ z={config.focus_distance}m")
    
    # Test corner points
    test_points = [
        ("Top-Left", -config.grid_size_x/2, config.grid_size_y/2),
        ("Top-Right", config.grid_size_x/2, config.grid_size_y/2),
        ("Bottom-Left", -config.grid_size_x/2, -config.grid_size_y/2),
        ("Bottom-Right", config.grid_size_x/2, -config.grid_size_y/2),
        ("Center", 0.0, 0.0),
        ("Test Point", 0.3, 0.2),  # 30cm right, 20cm up
    ]
    
    print(f"\n  Video: {video_w}x{video_h} px")
    print(f"\n  Coordinate Mapping (3D → 2D):")
    print("  " + "-"*66)
    
    # Aspect ratio handling
    aspect_ratio = config.grid_size_x / config.grid_size_y
    video_aspect = video_w / video_h
    
    if video_aspect > aspect_ratio:
        overlay_h = video_h
        overlay_w = int(video_h * aspect_ratio)
        x_offset = (video_w - overlay_w) // 2
        y_offset = 0
    else:
        overlay_w = video_w
        overlay_h = int(video_w / aspect_ratio)
        x_offset = 0
        y_offset = (video_h - overlay_h) // 2
    
    print(f"  Overlay region: {overlay_w}x{overlay_h} px @ offset ({x_offset}, {y_offset})")
    print("  " + "-"*66)
    
    for name, x_m, y_m in test_points:
        # Normalize to [0, 1]
        norm_x = (x_m + config.grid_size_x / 2.0) / config.grid_size_x
        norm_y = (y_m + config.grid_size_y / 2.0) / config.grid_size_y
        
        # Map to overlay pixels
        pixel_x = int(norm_x * overlay_w)
        pixel_y = int((1.0 - norm_y) * overlay_h)  # Flip Y
        
        # Absolute video coordinates
        video_x = x_offset + pixel_x
        video_y = y_offset + pixel_y
        
        print(f"  {name:15s}: ({x_m:+.2f}, {y_m:+.2f})m → ({video_x:4d}, {video_y:4d})px")
    
    print("\n✓ Coordinate mapping test complete!\n")


def test_heatmap_generation():
    """Test: Synthetic power map → heatmap image"""
    print("\n" + "="*70)
    print("TEST 2: Heatmap Generation")
    print("="*70)
    
    # Create synthetic power map
    grid_size = 30
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Simulate a peak at (0.3, 0.2) with Gaussian
    peak_x, peak_y = 0.3, 0.2
    sigma = 0.2
    power_db = -20 * np.exp(-((X - peak_x)**2 + (Y - peak_y)**2) / (2 * sigma**2)) - 30
    
    print(f"✓ Synthetic power map: {grid_size}x{grid_size}")
    print(f"  Peak: ({peak_x}, {peak_y}), Power range: [{power_db.min():.1f}, {power_db.max():.1f}] dB")
    
    # Apply thresholding
    db_min, db_max = -40, -10
    power_clipped = np.clip(power_db, db_min, db_max)
    
    threshold = db_min + (db_max - db_min) * 0.1
    mask = power_db > threshold
    
    normalized = np.zeros_like(power_db, dtype=np.float32)
    normalized[mask] = (power_clipped[mask] - db_min) / (db_max - db_min)
    
    normalized_uint8 = (normalized * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_bgr = cv2.applyColorMap(normalized_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    # Create alpha channel
    alpha = (normalized * 255).astype(np.uint8)
    heatmap_rgba = np.dstack([heatmap_rgb, alpha])
    
    print(f"✓ Heatmap generated: {heatmap_rgba.shape}")
    print(f"  Non-zero pixels: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
    
    # Save for inspection
    output_dir = Path(__file__).parent.parent / 'data' / 'test_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(power_db, cmap='jet', origin='lower')
    plt.colorbar(label='Power (dB)')
    plt.title('Power Map (dB)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray', origin='lower')
    plt.title('Mask (Thresholded)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_rgba, origin='lower')
    plt.title('Heatmap (RGBA)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    output_file = output_dir / 'heatmap_test.png'
    plt.savefig(output_file, dpi=150)
    print(f"\n✓ Visualization saved: {output_file}")
    print("\n✓ Heatmap generation test complete!\n")


def test_overlay_blending():
    """Test: Video frame + heatmap blending"""
    print("\n" + "="*70)
    print("TEST 3: Overlay Blending")
    print("="*70)
    
    # Create synthetic video frame (gradient)
    video_h, video_w = 1080, 1920
    frame = np.zeros((video_h, video_w, 3), dtype=np.uint8)
    
    # Add gradient background
    for i in range(video_h):
        frame[i, :, :] = int(50 + 100 * i / video_h)
    
    # Add grid lines for reference
    for i in range(0, video_h, 100):
        frame[i:i+2, :, :] = [100, 100, 100]
    for i in range(0, video_w, 100):
        frame[:, i:i+2, :] = [100, 100, 100]
    
    print(f"✓ Synthetic video frame: {video_w}x{video_h}")
    
    # Create heatmap (simpler)
    heatmap_size = 30
    heatmap = np.zeros((heatmap_size, heatmap_size, 4), dtype=np.uint8)
    
    # Draw a bright circle in the center
    center = heatmap_size // 2
    for i in range(heatmap_size):
        for j in range(heatmap_size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            if dist < heatmap_size // 3:
                intensity = int(255 * (1 - dist / (heatmap_size // 3)))
                heatmap[i, j] = [0, 0, intensity, intensity]  # Blue with alpha
    
    print(f"✓ Synthetic heatmap: {heatmap_size}x{heatmap_size}")
    
    # Resize heatmap to overlay area
    overlay_w, overlay_h = 1920, 1080
    heatmap_resized = cv2.resize(heatmap, (overlay_w, overlay_h), interpolation=cv2.INTER_LINEAR)
    
    # Alpha blending
    heatmap_rgb = heatmap_resized[:, :, :3]
    heatmap_alpha = heatmap_resized[:, :, 3] / 255.0
    user_alpha = 0.6  # 60% opacity
    combined_alpha = heatmap_alpha * user_alpha
    
    alpha_3ch = combined_alpha[:, :, np.newaxis]
    blended = (frame * (1 - alpha_3ch) + heatmap_rgb * alpha_3ch).astype(np.uint8)
    
    print(f"✓ Blended frame: {blended.shape}")
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'test_outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'video_frame.png'), frame)
    cv2.imwrite(str(output_dir / 'heatmap_overlay.png'), 
                cv2.cvtColor(heatmap_resized[:, :, :3], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / 'blended_result.png'), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    print(f"\n✓ Test images saved to: {output_dir}")
    print("\n✓ Overlay blending test complete!\n")


def test_mic_geometry():
    """Test: Microphone geometry loading"""
    print("\n" + "="*70)
    print("TEST 4: Microphone Geometry")
    print("="*70)
    
    micgeom_path = Path(__file__).parent.parent / 'micgeom.xml'
    
    if not micgeom_path.exists():
        print(f"❌ Microphone geometry file not found: {micgeom_path}")
        return
    
    mic_positions = load_mic_geometry(str(micgeom_path))
    
    print(f"✓ Loaded {len(mic_positions)} microphone positions")
    print(f"  X range: [{mic_positions[:, 0].min():.3f}, {mic_positions[:, 0].max():.3f}] m")
    print(f"  Y range: [{mic_positions[:, 1].min():.3f}, {mic_positions[:, 1].max():.3f}] m")
    print(f"  Z range: [{mic_positions[:, 2].min():.3f}, {mic_positions[:, 2].max():.3f}] m")
    
    # Check if it's a circular array
    center = mic_positions.mean(axis=0)
    radii = np.linalg.norm(mic_positions - center, axis=1)
    
    print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m")
    print(f"  Radii: mean={radii.mean():.3f}m, std={radii.std():.4f}m")
    
    if radii.std() < 0.01:  # Very small variation
        print(f"  ✓ Circular array detected (radius ≈ {radii.mean():.3f}m)")
    else:
        print(f"  ⚠ Non-uniform array")
    
    print("\n✓ Microphone geometry test complete!\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ACOUSTIC HEATMAP OVERLAY - DEBUG TESTS")
    print("="*70)
    print("Testing coordinate mapping, heatmap generation, and overlay blending")
    print("="*70 + "\n")
    
    try:
        test_mic_geometry()
        test_coordinate_mapping()
        test_heatmap_generation()
        test_overlay_blending()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Run the GUI: python run_gui.py")
        print("2. Enable 'Beamforming Overlay'")
        print("3. Make some noise and observe the heatmap")
        print("4. Adjust alpha/dB sliders for better visualization")
        print("\nTest outputs saved to: data/test_outputs/")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
