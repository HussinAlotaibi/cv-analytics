import os
import re
import io
import pdfplumber
import PyPDF2
import docx
from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from unidecode import unidecode
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Tokenization setup
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
EN_STOP = set(stopwords.words("english"))

# Custom stopwords for CV analysis
CUSTOM_STOP = {
    'cv', 'resume', 'responsible', 'worked', 'using', 'project', 'projects', 
    'tool', 'performed', 'objective', 'summary', 'include', 'data', 
    'management', 'experience', 'senior', 'lead', 'team', 'develop', 
    'developed', 'design', 'designed', 'implement', 'implemented', 'build', 
    'built', 'created', 'skills', 'skill', 'abilities', 'ability', 
    'proficient', 'knowledge', 'strong', 'understanding', 'etc'
}

# Year extraction
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b|\b(19\d{2}|20\d{2})\s*(?:-|–|to)\s*(19\d{2}|20\d{2}|present|now)\b", re.IGNORECASE)

def read_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def read_pdf(path: str) -> str:
    # Primary: pdfplumber; fallback: PyPDF2
    try:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        out = "\n".join(text)
        if out.strip():
            return out
    except Exception as e:
        print(f"pdfplumber failed: {e}, trying PyPDF2...")
    
    text = []
    with open(path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text.append(p.extract_text() or "")
    return "\n".join(text)

def read_any(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext == ".pdf":
        return read_pdf(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".txt":
        return read_txt(path)
    else:
        try:
            return read_txt(path)
        except:
            return ""

def tokenize(text: str, extra_stop: set = None):
    text = unidecode(text or "")
    text = re.sub(r"\s+", " ", text)
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    
    # Remove pure numbers
    tokens = [t for t in tokens if not t.isdigit()]
    
    # Apply stopwords
    stop = EN_STOP.copy()
    stop.update(CUSTOM_STOP)
    if extra_stop:
        stop.update({w.lower() for w in extra_stop})
    
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    return tokens, text

def extract_year_mentions(raw_text: str):
    years = []
    found_patterns = YEAR_RE.findall(raw_text)
    
    for pattern in found_patterns:
        for year_str in pattern:
            if year_str and year_str.lower() not in ("present", "now"):
                try:
                    y = int(year_str)
                    if 1950 <= y <= datetime.now().year + 1:
                        years.append(y)
                except ValueError:
                    pass
    return years

class CVAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.raw_text = read_any(file_path)
        self.tokens, self.normalized_text = tokenize(self.raw_text)
        self.word_counts = Counter(self.tokens)
        self.year_mentions = extract_year_mentions(self.raw_text)
        
    def get_summary(self):
        return {
            'file_name': self.file_name,
            'total_words': len(self.tokens),
            'unique_words': len(set(self.tokens)),
            'raw_text_length': len(self.raw_text),
            'years_mentioned': len(self.year_mentions)
        }
    
    def generate_word_cloud_figure(self, max_words=300, width=1600, height=900):
        """Generate word cloud and return matplotlib figure"""
        wc_stop = STOPWORDS.union(EN_STOP).union({w.lower() for w in CUSTOM_STOP})
        
        cloud = WordCloud(
            width=width, height=height,
            background_color="white",
            max_words=max_words,
            collocations=False,
            stopwords=wc_stop
        ).generate(" ".join(self.tokens))
        
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {self.file_name}", fontsize=16, pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_top_words_figure(self, n=20):
        """Generate top words chart and return matplotlib figure"""
        top_words = self.word_counts.most_common(n)
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(words)), counts, color='skyblue')
        ax.set_xlabel("Words", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Top {n} Words in {self.file_name}", fontsize=14)
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_year_mentions_figure(self):
        """Generate year mentions chart and return matplotlib figure"""
        if not self.year_mentions:
            return None
        
        year_counts = Counter(self.year_mentions)
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years, counts, marker='o', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Mentions", fontsize=12)
        ax.set_title(f"Year Mentions in {self.file_name}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for year, count in zip(years, counts):
            ax.text(year, count + 0.1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def analyze_keywords_data(self, keyword_lists):
        """Analyze keyword frequency and return data"""
        def count_keywords(text: str, keywords: list):
            text_lower = text.lower()
            counts = {}
            for keyword in keywords:
                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
                counts[keyword] = len(re.findall(pattern, text_lower))
            return counts
        
        results = {}
        for category, keywords in keyword_lists.items():
            results[category] = count_keywords(self.raw_text, keywords)
        
        return results
    
    def plot_keywords_figure(self, keyword_lists):
        """Generate keyword analysis chart and return matplotlib figure"""
        results = self.analyze_keywords_data(keyword_lists)
        
        all_keywords = []
        all_counts = []
        
        for category, keyword_counts in results.items():
            for keyword, count in keyword_counts.items():
                all_keywords.append(f"{category}: {keyword}")
                all_counts.append(count)
        
        # Sort by count
        sorted_data = sorted(zip(all_keywords, all_counts), key=lambda x: x[1], reverse=True)
        # Only show keywords with count > 0
        sorted_data = [(k, c) for k, c in sorted_data if c > 0]
        
        if not sorted_data:
            return None
            
        sorted_keywords, sorted_counts = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(sorted_keywords) * 0.4)))
        bars = ax.barh(range(len(sorted_keywords)), sorted_counts, color='lightcoral')
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Keywords", fontsize=12)
        ax.set_title(f"Keyword Analysis for {self.file_name}", fontsize=14)
        ax.set_yticks(range(len(sorted_keywords)))
        ax.set_yticklabels(sorted_keywords)
        ax.invert_yaxis()
        
        # Add value labels
        for i, count in enumerate(sorted_counts):
            ax.text(count + 0.1, i, str(count), va='center')
        
        plt.tight_layout()
        return fig
