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
