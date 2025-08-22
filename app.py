import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
import json
from flask_cv_analyzer import CVAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import Counter
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_plot_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze CV
        analyzer = CVAnalyzer(filepath)
        
        # Get summary
        summary = analyzer.get_summary()
        
        # Generate word cloud
        word_cloud = analyzer.generate_word_cloud_figure()
        word_cloud_b64 = create_plot_base64(word_cloud)
        
        # Generate top words chart
        top_words_fig = analyzer.plot_top_words_figure()
        top_words_b64 = create_plot_base64(top_words_fig)
        
        # Generate year mentions chart
        year_mentions_fig = analyzer.plot_year_mentions_figure()
        year_mentions_b64 = create_plot_base64(year_mentions_fig) if year_mentions_fig else None
        
        # Get keyword analysis
        keyword_lists = {
            "Programming Languages": ["python", "sql", "java", "javascript", "r", "scala", "c++", "c#"],
            "Data & Analytics": ["data", "analytics", "machine learning", "ai", "statistics", "visualization", "tableau", "power bi"],
            "Cloud Platforms": ["aws", "azure", "gcp", "google cloud", "cloud"],
            "Databases": ["mysql", "postgresql", "mongodb", "oracle", "sql server", "redis"],
            "Management Skills": ["leadership", "project management", "team", "strategy", "planning"],
            "Technical Skills": ["api", "microservices", "docker", "kubernetes", "git", "ci/cd"]
        }
        
        keyword_results = analyzer.analyze_keywords_data(keyword_lists)
        keyword_chart_fig = analyzer.plot_keywords_figure(keyword_lists)
        keyword_chart_b64 = create_plot_base64(keyword_chart_fig)
        
        # Get top words data for table
        top_words_data = analyzer.word_counts.most_common(20)
        
        # Get AI analysis (will use mock data if no API key)
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        ai_analysis = analyzer.get_ai_analysis(api_key=openai_api_key)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'word_cloud': word_cloud_b64,
            'top_words_chart': top_words_b64,
            'year_mentions_chart': year_mentions_b64,
            'keyword_chart': keyword_chart_b64,
            'keyword_results': keyword_results,
            'top_words_data': top_words_data,
            'ai_analysis': ai_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_pdf_report(analysis_data, filename):
    """Generate a professional PDF report from analysis data"""
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2563eb')
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#1f2937')
    )
    normal_style = styles['Normal']
    
    # Title
    story.append(Paragraph("CV Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Paragraph(f"<b>Filename:</b> {filename}", normal_style))
    story.append(Spacer(1, 20))
    
    # Summary Statistics
    if 'summary' in analysis_data:
        story.append(Paragraph("Executive Summary", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Words', f"{analysis_data['summary']['total_words']:,}"],
            ['Unique Words', f"{analysis_data['summary']['unique_words']:,}"],
            ['Years Mentioned', str(analysis_data['summary']['years_mentioned'])],
            ['Document Length', f"{analysis_data['summary']['raw_text_length']:,} characters"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
    
    # AI Analysis Section
    if 'ai_analysis' in analysis_data:
        ai_data = analysis_data['ai_analysis']
        
        # Professional Summary
        story.append(Paragraph("AI Professional Summary", heading_style))
        story.append(Paragraph(ai_data.get('professional_summary', 'N/A'), normal_style))
        story.append(Spacer(1, 15))
        
        # Experience Assessment
        story.append(Paragraph("Experience Assessment", heading_style))
        exp_data = [
            ['Assessment', 'Rating'],
            ['Experience Level', ai_data.get('experience_level', 'N/A')],
            ['Estimated Salary Range', f"${ai_data.get('estimated_salary_range', 'N/A')}"],
            ['Skills Match', f"{ai_data.get('skills_match_percentage', 0)}%"]
        ]
        
        exp_table = Table(exp_data)
        exp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(exp_table)
        story.append(Spacer(1, 20))
        
        # Key Strengths
        story.append(Paragraph("Key Strengths", heading_style))
        for i, strength in enumerate(ai_data.get('key_strengths', []), 1):
            story.append(Paragraph(f"{i}. {strength}", normal_style))
        story.append(Spacer(1, 15))
        
        # Areas to Clarify
        story.append(Paragraph("Areas to Clarify in Interview", heading_style))
        for i, flag in enumerate(ai_data.get('red_flags', []), 1):
            story.append(Paragraph(f"{i}. {flag}", normal_style))
        story.append(Spacer(1, 15))
        
        # Cultural Fit Indicators
        story.append(Paragraph("Cultural Fit Assessment", heading_style))
        cultural_data = [['Trait', 'Rating']]
        for trait, rating in ai_data.get('cultural_fit_indicators', {}).items():
            cultural_data.append([trait.replace('_', ' ').title(), rating])
        
        cultural_table = Table(cultural_data)
        cultural_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(cultural_table)
        story.append(Spacer(1, 20))
        
        # Interview Questions
        story.append(Paragraph("Suggested Interview Questions", heading_style))
        for i, question in enumerate(ai_data.get('interview_questions', []), 1):
            story.append(Paragraph(f"{i}. {question}", normal_style))
        story.append(Spacer(1, 20))
        
        # HR Recommendation
        story.append(Paragraph("HR Recommendation", heading_style))
        story.append(Paragraph(ai_data.get('hr_recommendation', 'N/A'), normal_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        # Get analysis data from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No analysis data provided'}), 400
        
        filename = data.get('filename', 'Unknown CV')
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(data, filename)
        
        # Create safe filename for download
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        pdf_filename = f"CV_Analysis_Report_{safe_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"PDF Generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/<data_type>')
def export_data(data_type):
    try:
        # This would need to store analysis results in session or database
        # For now, return error
        return jsonify({'error': 'Export functionality requires session storage'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
