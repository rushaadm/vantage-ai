from fpdf import FPDF
import os
from datetime import datetime

class AuditReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'VANTAGE AI: VISUAL ATTENTION ANALYSIS REPORT', ln=True, align='C', border=1)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_section_header(self, title):
        self.ln(5)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(2)
    
    def add_stat_row(self, label, value, unit=""):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(100, 6, label, border=0)
        self.set_font('Helvetica', 'B', 10)
        self.cell(0, 6, f"{value}{unit}", ln=True, border=0)
    
    def add_comprehensive_report(self, data):
        """Generate comprehensive formal report with all statistics"""
        stats = data.get('stats', {})
        clarity_score = stats.get('clarity_score', 0)
        engagement_score = stats.get('engagement_score', 0)
        attention_stability = stats.get('attention_stability', 0)
        ai_narrative = data.get('ai_suggestions', '')
        processing_time = data.get('processing_time', 0)
        fps = data.get('fps', 0)
        frame_count = data.get('frame_count', 0)
        processed_frames = data.get('processed_frames', 0)
        duration = data.get('duration', 0)
        
        # Title Page
        self.add_page()
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(0, 0, 0)
        self.cell(0, 20, 'VISUAL ATTENTION ANALYSIS REPORT', ln=True, align='C')
        self.ln(10)
        
        self.set_font('Helvetica', 'B', 48)
        self.set_text_color(0, 150, 0)  # Green for score
        self.cell(0, 30, f'{clarity_score:.0f}', ln=True, align='C')
        self.set_font('Helvetica', 'B', 18)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, 'Clarity Score / 100', ln=True, align='C')
        self.ln(15)
        
        # Executive Summary
        self.add_section_header('EXECUTIVE SUMMARY')
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 6, 
            f"This report presents a comprehensive analysis of visual attention patterns in the submitted video content. "
            f"The analysis processed {processed_frames} frames from a {duration:.2f}-second video ({frame_count} total frames at {fps:.2f} FPS). "
            f"Processing completed in {processing_time:.2f} seconds. The overall clarity score of {clarity_score:.0f}/100 indicates "
            f"{'excellent' if clarity_score >= 80 else 'good' if clarity_score >= 60 else 'moderate' if clarity_score >= 40 else 'poor'} "
            f"visual attention coherence.")
        self.ln(5)
        
        # Key Metrics
        self.add_section_header('KEY METRICS')
        self.add_stat_row('Clarity Score:', f'{clarity_score:.2f}', '/100')
        self.add_stat_row('Engagement Score:', f'{engagement_score:.2f}', '/100')
        self.add_stat_row('Attention Stability:', f'{attention_stability:.2f}', '/100')
        self.add_stat_row('Video Duration:', f'{duration:.2f}', ' seconds')
        self.add_stat_row('Frame Rate:', f'{fps:.2f}', ' FPS')
        self.add_stat_row('Total Frames:', f'{frame_count:,}')
        self.add_stat_row('Processed Frames:', f'{processed_frames:,}')
        self.add_stat_row('Processing Time:', f'{processing_time:.2f}', ' seconds')
        self.add_stat_row('Fixation Rate:', f'{stats.get("fixation_rate", 0):.2f}', ' fixations/frame')
        self.add_stat_row('Total Fixations:', f'{stats.get("total_fixations", 0):,}')
        self.ln(5)
        
        # Entropy Statistics
        self.add_section_header('ENTROPY ANALYSIS')
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5, 
            'Entropy measures the distribution of visual attention across the frame. Higher entropy indicates more '
            'distributed attention, while lower entropy suggests focused attention on specific areas.')
        self.ln(3)
        self.add_stat_row('Average Entropy:', f'{stats.get("avg_entropy", 0):.4f}')
        self.add_stat_row('Maximum Entropy:', f'{stats.get("max_entropy", 0):.4f}')
        self.add_stat_row('Minimum Entropy:', f'{stats.get("min_entropy", 0):.4f}')
        self.add_stat_row('Standard Deviation:', f'{stats.get("std_entropy", 0):.4f}')
        self.ln(5)
        
        # Conflict Statistics
        self.add_section_header('MOTION CONFLICT ANALYSIS')
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5,
            'Conflict (KL Divergence) measures the discrepancy between static saliency predictions and actual motion patterns. '
            'Lower conflict values indicate better alignment between expected and actual attention patterns.')
        self.ln(3)
        self.add_stat_row('Average Conflict:', f'{stats.get("avg_conflict", 0):.4f}')
        self.add_stat_row('Maximum Conflict:', f'{stats.get("max_conflict", 0):.4f}')
        self.add_stat_row('Minimum Conflict:', f'{stats.get("min_conflict", 0):.4f}')
        self.add_stat_row('Standard Deviation:', f'{stats.get("std_conflict", 0):.4f}')
        self.ln(5)
        
        # Statistical Distribution
        self.add_section_header('STATISTICAL DISTRIBUTION')
        entropy_range = stats.get("max_entropy", 0) - stats.get("min_entropy", 0)
        conflict_range = stats.get("max_conflict", 0) - stats.get("min_conflict", 0)
        self.add_stat_row('Entropy Range:', f'{entropy_range:.4f}')
        self.add_stat_row('Conflict Range:', f'{conflict_range:.4f}')
        self.add_stat_row('Entropy Coefficient of Variation:', 
                         f'{(stats.get("std_entropy", 0) / stats.get("avg_entropy", 1) * 100):.2f}' if stats.get("avg_entropy", 0) > 0 else 'N/A', '%')
        self.add_stat_row('Conflict Coefficient of Variation:', 
                         f'{(stats.get("std_conflict", 0) / stats.get("avg_conflict", 1) * 100):.2f}' if stats.get("avg_conflict", 0) > 0 else 'N/A', '%')
        self.ln(5)
        
        # AI Recommendations
        self.add_section_header('AI-POWERED RECOMMENDATIONS')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, ai_narrative if ai_narrative else "No AI recommendations available.")
        self.ln(5)
        
        # Methodology
        self.add_section_header('METHODOLOGY')
        self.set_font('Helvetica', '', 9)
        self.multi_cell(0, 5,
            'This analysis employs a dual-stream visual attention model based on established eye-tracking research:\n\n'
            '1. Static Saliency Detection: Color contrast and edge detection for predicting attention based on visual features (Itti & Koch, 2001)\n'
            '2. Dynamic Motion Analysis: Frame difference for detecting motion patterns\n'
            '3. Entropy Calculation: Shannon entropy to measure attention distribution (Tatler et al., 2011)\n'
            '4. Conflict Measurement: KL Divergence to assess alignment between static and dynamic attention patterns\n\n'
            'SCORE CALCULATIONS:\n'
            '- Clarity Score: 100 - (avg_conflict × 20), measures visual hierarchy clarity (Itti & Koch, 2001)\n'
            '- Attention Stability: 100 - (std_entropy × 30), measures consistency of attention patterns (Tatler et al., 2011)\n'
            '- Engagement Score: (avg_saliency × 60) + (fixation_rate × 8), combines saliency intensity and fixation rate (Yarbus, 1967)\n\n'
            'REFERENCES:\n'
            '- Itti, L., & Koch, C. (2001). Computational modelling of visual attention. Nature Reviews Neuroscience, 2(3), 194-203.\n'
            '- Yarbus, A. L. (1967). Eye movements and vision. Plenum Press.\n'
            '- Tatler, B. W., Hayhoe, M. M., Land, M. F., & Ballard, D. H. (2011). Eye guidance in natural vision: Reinterpreting salience. Journal of Vision, 11(5), 5.')
        self.ln(5)
        
        # Conclusion
        self.add_section_header('CONCLUSION')
        self.set_font('Helvetica', '', 10)
        score_interpretation = (
            "excellent visual coherence" if clarity_score >= 80 else
            "good visual coherence" if clarity_score >= 60 else
            "moderate visual coherence requiring optimization" if clarity_score >= 40 else
            "poor visual coherence requiring significant improvement"
        )
        self.multi_cell(0, 6,
            f'The analysis reveals {score_interpretation} in the video content. '
            f'Key areas for improvement include reducing motion conflict and optimizing attention distribution. '
            f'Consider implementing the AI-generated recommendations to enhance visual clarity and viewer engagement.')

# Integration with Gemini
def get_ai_suggestions(stats):
    """Generate AI suggestions using Gemini API"""
    try:
        import google.generativeai as genai
        
        # Use provided API key
        api_key = "AIzaSyBFjEpZYCRtvqApOSCqrie4TfhXP08Xc_c"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are an expert in visual attention and eye-tracking research. Analyze these eye-tracking statistics from a video and provide 3-5 specific, actionable recommendations for improving visual content.

Statistics:
- Clarity Score: {stats.get('clarity_score', 0)}/100 (measures visual hierarchy clarity based on motion-saliency conflict)
- Engagement Score: {stats.get('engagement_score', 0)}/100 (measures viewer engagement based on saliency and fixations)
- Attention Stability: {stats.get('attention_stability', 0)}/100 (measures attention consistency across frames)
- Average Saliency: {stats.get('avg_saliency', 0)} (average attention intensity)
- Fixation Rate: {stats.get('fixation_rate', 0)} fixations per frame
- Total Fixations: {stats.get('total_fixations', 0)} (total attention points detected)
- Average Entropy: {stats.get('avg_entropy', 0)} (attention distribution)
- Average Conflict: {stats.get('avg_conflict', 0)} (motion-saliency alignment)

Provide specific recommendations based on eye-tracking research principles (Itti & Koch, 2001; Yarbus, 1967; Tatler et al., 2011). Be concise and actionable."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"AI suggestions error: {e}")
        import traceback
        traceback.print_exc()
        return f"AI analysis unavailable: {str(e)}. Please check API configuration."
