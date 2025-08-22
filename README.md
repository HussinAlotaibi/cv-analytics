# CV Analytics Dashboard 🚀

A powerful Python Flask web application for comprehensive CV analysis featuring word clouds, statistical charts, and keyword extraction.

## ✨ Features

- **📁 Multi-format Support**: Upload PDF, DOCX, or TXT files
- **🎨 Word Cloud Generation**: Beautiful visual representation of most frequent terms
- **📊 Statistical Analysis**: Top words bar charts and frequency analysis
- **📅 Timeline Analysis**: Year mentions tracking with line charts
- **🔍 Keyword Extraction**: Categorized analysis of technical skills, programming languages, etc.
- **💾 Data Export**: Download analysis results as CSV files
- **📱 Responsive Design**: Works perfectly on desktop and mobile devices
- **🔒 Secure**: Files are processed temporarily and automatically deleted

## 🛠️ Technologies Used

- **Backend**: Flask (Python 3.11+)
- **CV Processing**: pdfplumber, PyPDF2, python-docx
- **Data Visualization**: matplotlib, wordcloud
- **Text Processing**: NLTK, unidecode
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **File Handling**: Drag & drop interface with AJAX

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/cv-analytics-flask.git
cd cv-analytics-flask
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
```
http://127.0.0.1:5000
```

1. Clone the repository:

   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```
   cd cv-analytics
   ```

3. Install the dependencies:

   ```
   npm install
   ```

### Running the Application

To start the development server, run:

```
npm run dev
```

The application will be available at `http://localhost:3000`.

### Project Structure

```
cv-analytics
├── src
│   ├── app
│   ├── components
│   ├── hooks
│   ├── lib
│   ├── styles
│   └── types
├── package.json
├── tsconfig.json
├── next.config.js
├── next-env.d.ts
├── .eslintrc.json
├── .gitignore
└── README.md
```

### Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

### License

This project is licensed under the MIT License. See the LICENSE file for details.