import cv2
import numpy as np

try:
    import torch
    import pysaliency
    PYSALIENCY_AVAILABLE = True
except ImportError:
    PYSALIENCY_AVAILABLE = False
    print("Warning: pysaliency not available. Using lightweight fallback.")

class VantageEngine:
    def __init__(self):
        # Don't load heavy models - use lightweight fallback
        self.model = None
        
    def get_saliency_map(self, frame):
        """Lightweight saliency detection - fast and CPU-friendly"""
        # Very lightweight: simple edge + center bias
        h, w = frame.shape[:2]
        
        # Downscale even more for speed
        if w > 160:
            frame = cv2.resize(frame, (160, int(h * 160 / w)), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fast edge detection with small kernel
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (7, 7), 0)
        
        # Simple center bias
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 2)**2))
        
        # Combine
        saliency = edges * 0.5 + center_bias * 0.5
        return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    
    def get_motion_map(self, prev_frame, curr_frame):
        """Lightweight motion detection - simplified optical flow"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Very fast optical flow with minimal parameters
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,
            levels=1,  # Minimal levels
            winsize=5,  # Small window
            iterations=1,  # Single iteration
            poly_n=3,
            poly_sigma=1.0,
            flags=0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = cv2.GaussianBlur(mag, (5, 5), 0)  # Small blur
        return cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    def calculate_metrics(self, saliency_map, motion_map):
        """Calculate metrics efficiently"""
        # Simplified calculations
        flat_s = saliency_map.flatten() + 1e-7
        entropy = -np.sum(flat_s * np.log2(flat_s))
        
        eps = 1e-7
        p = saliency_map.flatten() + eps
        q = motion_map.flatten() + eps
        kl_div = np.sum(p * np.log(p / q))
        
        return {"entropy": float(entropy), "conflict": float(kl_div)}
