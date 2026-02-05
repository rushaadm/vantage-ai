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
        """Optical flow-based motion detection (Farneback method)"""
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Farneback Optical Flow - more accurate than frame difference
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,      # Image pyramid scale
            levels=3,           # Number of pyramid levels
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Size of pixel neighborhood
            poly_sigma=1.2,     # Gaussian sigma
            flags=0
        )
        
        # Calculate magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Normalize and blur to simulate human peripheral vision
        magnitude = cv2.GaussianBlur(magnitude, (15, 15), 0)
        return cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    def get_flicker_map(self, prev_frame, curr_frame):
        """Flicker detection - temporal brightness changes"""
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert to LAB color space for better luminance analysis
        prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB)
        curr_lab = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2LAB)
        
        # Extract L channel (luminance)
        prev_l = prev_lab[:, :, 0].astype(np.float32)
        curr_l = curr_lab[:, :, 0].astype(np.float32)
        
        # Calculate flicker as absolute change in luminance
        flicker = np.abs(curr_l - prev_l)
        
        # Normalize and blur
        flicker = cv2.GaussianBlur(flicker, (9, 9), 0)
        return cv2.normalize(flicker, None, 0, 1, cv2.NORM_MINMAX)
    
    def detect_saccades(self, prev_saliency, curr_saliency, threshold=0.3):
        """Detect saccadic eye movements (rapid shifts in attention)"""
        # Calculate center of mass for each saliency map
        def get_center_of_mass(sal_map):
            h, w = sal_map.shape
            y_coords, x_coords = np.ogrid[:h, :w]
            total = np.sum(sal_map)
            if total == 0:
                return (w/2, h/2)
            cx = np.sum(sal_map * x_coords) / total
            cy = np.sum(sal_map * y_coords) / total
            return (cx, cy)
        
        prev_center = get_center_of_mass(prev_saliency)
        curr_center = get_center_of_mass(curr_saliency)
        
        # Calculate distance and velocity
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Normalize by image size
        h, w = prev_saliency.shape
        normalized_distance = distance / np.sqrt(w*w + h*h)
        
        # Saccade if movement is rapid (high normalized distance)
        is_saccade = normalized_distance > threshold
        return {
            "is_saccade": bool(is_saccade),
            "distance": float(normalized_distance),
            "velocity": float(normalized_distance * 30)  # Assuming 30 fps
        }

    def calculate_metrics(self, saliency_map, motion_map, flicker_map=None, prev_saliency=None):
        """Calculate comprehensive eye-tracking metrics"""
        if saliency_map.shape != motion_map.shape:
            motion_map = cv2.resize(motion_map, (saliency_map.shape[1], saliency_map.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 1. Shannon Entropy (attention distribution)
        flat_s = saliency_map.flatten() + 1e-7
        flat_s_norm = flat_s / np.sum(flat_s)  # Normalize to probabilities
        entropy = -np.sum(flat_s_norm * np.log2(flat_s_norm))
        
        # 2. KL Divergence / Conflict (motion-saliency alignment)
        p = saliency_map.flatten() + 1e-7
        q = motion_map.flatten() + 1e-7
        p_norm = p / np.sum(p)
        q_norm = q / np.sum(q)
        conflict = np.sum(p_norm * np.log(p_norm / q_norm))
        
        # 3. Spatial Coherence (how clustered attention is)
        # Calculate variance of attention distribution
        h, w = saliency_map.shape
        y_coords, x_coords = np.ogrid[:h, :w]
        total_attention = np.sum(saliency_map)
        if total_attention > 0:
            center_x = np.sum(saliency_map * x_coords) / total_attention
            center_y = np.sum(saliency_map * y_coords) / total_attention
            # Calculate spread (standard deviation)
            spread_x = np.sqrt(np.sum(saliency_map * (x_coords - center_x)**2) / total_attention)
            spread_y = np.sqrt(np.sum(saliency_map * (y_coords - center_y)**2) / total_attention)
            spatial_coherence = 1.0 / (1.0 + (spread_x + spread_y) / (w + h))  # Normalized coherence
        else:
            spatial_coherence = 0.0
        
        # 4. Temporal Stability (if previous frame available)
        temporal_stability = 0.0
        saccade_info = None
        if prev_saliency is not None:
            if prev_saliency.shape == saliency_map.shape:
                # Correlation between frames
                prev_flat = prev_saliency.flatten()
                curr_flat = saliency_map.flatten()
                correlation = np.corrcoef(prev_flat, curr_flat)[0, 1]
                temporal_stability = max(0, correlation) if not np.isnan(correlation) else 0.0
                
                # Saccade detection
                saccade_info = self.detect_saccades(prev_saliency, saliency_map)
        
        # 5. Flicker Sensitivity (if flicker map available)
        flicker_sensitivity = 0.0
        if flicker_map is not None:
            if flicker_map.shape == saliency_map.shape:
                # How much flicker aligns with saliency
                flicker_saliency_correlation = np.corrcoef(
                    flicker_map.flatten(), saliency_map.flatten()
                )[0, 1]
                flicker_sensitivity = max(0, flicker_saliency_correlation) if not np.isnan(flicker_saliency_correlation) else 0.0
        
        # 6. Visual Complexity (edge density + color variance)
        # Edge density
        edges = cv2.Canny((saliency_map * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Color variance (if we had original frame, use that; otherwise estimate from saliency)
        color_variance = float(np.std(saliency_map))
        visual_complexity = (edge_density + color_variance) / 2.0
        
        metrics = {
            "entropy": float(entropy),
            "conflict": float(conflict),
            "spatial_coherence": float(spatial_coherence),
            "temporal_stability": float(temporal_stability),
            "flicker_sensitivity": float(flicker_sensitivity),
            "visual_complexity": float(visual_complexity),
            "edge_density": float(edge_density)
        }
        
        if saccade_info:
            metrics.update(saccade_info)
        
        return metrics
