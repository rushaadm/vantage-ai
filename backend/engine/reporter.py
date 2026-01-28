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
        clarity_score = data.get('clarity_score', 0)
        ai_narrative = data.get('ai_narrative', '')
        processing_time = data.get('processing_time', 0)
        fps = data.get('fps', 0)
        frame_count = data.get('frame_count', 0)
        processed_frames = data.get('processed_frames', 0)
        duration = stats.get('duration', 0)
        
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
        self.add_stat_row('Overall Clarity Score:', f'{clarity_score:.2f}', '/100')
        self.add_stat_row('Video Duration:', f'{duration:.2f}', ' seconds')
        self.add_stat_row('Frame Rate:', f'{fps:.2f}', ' FPS')
        self.add_stat_row('Total Frames:', f'{frame_count:,}')
        self.add_stat_row('Processed Frames:', f'{processed_frames:,}')
        self.add_stat_row('Processing Time:', f'{processing_time:.2f}', ' seconds')
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
            'This analysis employs a dual-stream visual attention model combining:\n'
            '1. Static Saliency Detection: DeepGaze IIE model for predicting attention based on visual features\n'
            '2. Dynamic Motion Analysis: Farneback Optical Flow for detecting motion patterns\n'
            '3. Entropy Calculation: Shannon entropy to measure attention distribution\n'
            '4. Conflict Measurement: KL Divergence to assess alignment between static and dynamic attention patterns\n\n'
            'The clarity score is calculated as: 100 - (average_conflict × 10), normalized to 0-100 scale.')
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
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return "AI suggestions unavailable. Please set GEMINI_API_KEY in .env"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""Analyze these visual attention statistics from a video analysis:
- Clarity Score: {stats.get('clarity_score', 0)}/100
- Average Entropy: {stats.get('avg_entropy', 0):.4f}
- Average Conflict: {stats.get('avg_conflict', 0):.4f}
- Video Duration: {stats.get('duration', 0):.2f} seconds
- FPS: {stats.get('fps', 0):.2f}

Provide 3-5 specific, actionable recommendations to improve visual clarity and attention distribution for social media content creation. Be scientific and detailed."""
    response = model.generate_content(prompt)
    return response.text
