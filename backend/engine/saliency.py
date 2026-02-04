import cv2
import numpy as np

try:
    import torch
    import pysaliency
    PYSALIENCY_AVAILABLE = True
except ImportError:
    PYSALIENCY_AVAILABLE = False
    # Silently use lightweight fallback - no console spam

class VantageEngine:
    def __init__(self):
        # Don't load heavy models - use lightweight fallback
        self.model = None
        
    def get_saliency_map(self, frame):
        """Granular saliency detection - pixel-level analysis"""
        h, w = frame.shape[:2]
        
        # Process at higher resolution for granularity (was 160, now 320)
        if w > 320:
            frame = cv2.resize(frame, (320, int(h * 320 / w)), interpolation=cv2.INTER_CUBIC)
            h, w = frame.shape[:2]
        
        # Convert to LAB color space for better color contrast detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection for granularity
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = (edges1.astype(np.float32) + edges2.astype(np.float32)) / 2
        edges = cv2.GaussianBlur(edges, (7, 7), 1.5)
        
        # Color contrast (L channel variance)
        l_blur = cv2.GaussianBlur(l_channel.astype(np.float32), (15, 15), 0)
        color_contrast = np.abs(l_channel.astype(np.float32) - l_blur)
        color_contrast = cv2.GaussianBlur(color_contrast, (5, 5), 0)
        
        # Center bias (weaker for more natural attention)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 1.5)**2))
        
        # Granular combination - more weight on actual visual features
        saliency = edges * 0.5 + color_contrast * 0.3 + center_bias * 0.2
        return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    
    def get_motion_map(self, prev_frame, curr_frame):
        """Ultra-fast motion detection - frame difference only"""
        # Ensure frames have same shape
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Use simple frame difference instead of optical flow for speed
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Simple absolute difference - much faster than optical flow
        diff = np.abs(curr_gray - prev_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        return cv2.normalize(diff, None, 0, 1, cv2.NORM_MINMAX)

    def calculate_metrics(self, saliency_map, motion_map):
        """Calculate metrics efficiently - ensure shapes match"""
        # Ensure both maps have the same shape
        if saliency_map.shape != motion_map.shape:
            # Resize motion_map to match saliency_map
            motion_map = cv2.resize(motion_map, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Simplified calculations
        flat_s = saliency_map.flatten() + 1e-7
        entropy = -np.sum(flat_s * np.log2(flat_s))
        
        eps = 1e-7
        p = saliency_map.flatten() + eps
        q = motion_map.flatten() + eps
        kl_div = np.sum(p * np.log(p / q))
        
        return {"entropy": float(entropy), "conflict": float(kl_div)}
