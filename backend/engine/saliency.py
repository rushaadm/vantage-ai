import cv2
import numpy as np

class VantageEngine:
    """Fast, efficient saliency detection"""
    
    def __init__(self):
        pass
        
    def get_saliency_map(self, frame):
        """Fast saliency detection using color contrast and edges"""
        h, w = frame.shape[:2]
        
        # Resize for speed if needed
        if w > 240:
            scale = 240 / w
            frame = cv2.resize(frame, (240, int(h * scale)), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
        
        # Convert to LAB for better color analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Color contrast (simplified)
        l_blur = cv2.GaussianBlur(l_channel, (9, 9), 0)
        color_contrast = np.abs(l_channel - l_blur)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32) / 255.0
        
        # Center bias (weaker)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 2)**2))
        
        # Combine
        saliency = color_contrast * 0.5 + edges * 0.3 + center_bias * 0.2
        return cv2.normalize(saliency, None, 0, 1, cv2.NORM_MINMAX)
    
    def get_motion_map(self, prev_frame, curr_frame):
        """Ultra-fast motion detection"""
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        diff = np.abs(curr_gray - prev_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        return cv2.normalize(diff, None, 0, 1, cv2.NORM_MINMAX)

    def calculate_metrics(self, saliency_map, motion_map):
        """Calculate metrics efficiently"""
        if saliency_map.shape != motion_map.shape:
            motion_map = cv2.resize(motion_map, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        flat_s = saliency_map.flatten() + 1e-7
        entropy = -np.sum(flat_s * np.log2(flat_s))
        
        p = saliency_map.flatten() + 1e-7
        q = motion_map.flatten() + 1e-7
        conflict = np.sum(p * np.log(p / q))
        
        return {"entropy": float(entropy), "conflict": float(conflict)}
