import subprocess
from flask import Flask, render_template, request
app = Flask(__name__)
import os
        
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'File not uploaded'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'File not uploaded'
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return f'File uploaded successfully'
    else:
        return 'Only .csv files are allowed'

@app.route('/sentiment_analysis', methods=['POST'])
def run_script():
    subprocess.Popen(['python', 'sentiment_analysis.py'])
    
    return render_template('index.html')  

@app.route('/web_scraper')
def web_scraper():
    return render_template('web_scraper.html')



if __name__ == '__main__':
    app.run(debug=True)
