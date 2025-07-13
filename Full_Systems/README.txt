Application Malware Scanning Sandbox Using AI/ML
===============================================

Overview
--------
This project is a machine learning-based cybersecurity tool to detect and classify malware in PE files and URLs. It uses Random Forest (99.39% accuracy), SVM, and Neural Networks, with a Flask web app for real-time scanning and reporting. It supports behavioral analysis, scalability, and AES-256 encryption, compliant with GDPR/ISO 27001.

Key Features
------------
- ML-driven detection of ransomware, trojans, spyware
- Flask-based web interface with Matplotlib/Seaborn visualizations
- Scalable for 1,000 concurrent scans
- Secure PDF reports with file hashes and mitigation steps
- Supports online/offline modes
- Future-ready for AI enhancements (e.g., threat prediction)

Prerequisites
-------------
Hardware:
- Dual-core processor (2.0 GHz+)
- 4GB RAM
- 256GB SSD

Software:
- Ubuntu 20.04+/Windows 10/11
- Python 3.8+
- VS Code/PyCharm
- Dependencies: Flask, Scikit-learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn

Installation
-----------
1. Create a project directory (e.g., malware-scanning-sandbox).
2. Copy project files (app.py, config.py, etc.) into the directory.
3. Set up a virtual environment:
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
4. Create requirements.txt with:
   flask
   scikit-learn
   tensorflow
   pandas
   numpy
   matplotlib
   seaborn
5. Install dependencies:
   pip install -r requirements.txt
6. Place VirusShare/Kaggle datasets in data/ directory.
7. Configure paths in config.py for datasets and models.
8. Run:
   python app.py

Usage
-----
- Access http://localhost:5000 in a browser.
- Log in with MFA-enabled credentials.
- Upload PE files (.exe, .pdf, .zip, .docx, .rar, max 50MB) or URLs.
- View results with visualizations; download PDF reports.
- Use offline mode for secure environments.

Project Structure
----------------
malware-scanning-sandbox/
├── app.py            # Main Flask app
├── config.py         # Configuration
├── data/            # Datasets
├── models/          # ML models
├── static/          # CSS, JS, images
├── templates/       # HTML templates
├── scripts/         # Automation scripts
├── logs/            # Audit logs
├── reports/         # PDF reports
└── requirements.txt # Dependencies

Testing
------
- Web interface: Loads correctly, supports uploads (WI-001 to WI-003).
- File uploads: Identifies legitimate/malicious files (FU-001 to FU-005).
- Directory scans: Handles mixed/large directories (DS-001 to DS-004).
- ML performance: Random Forest at 99.39% accuracy (ML-001 to ML-003).
- Security: Robust input validation (ST-001 to ST-003).

Future Enhancements
------------------
- Cloud deployment
- Mobile support (iOS/Android)
- Network traffic analysis
- AI-driven threat prediction

Acknowledgments
---------------
- Supervisor: Mr. Chamara Dissanayake
- Datasets: Kaggle
- Tools: Flask, Scikit-learn, TensorFlow/Keras, Pandas, NumPy, Matplotlib, Seaborn

Contact
-------
Wanniachchige M Fonseka (Plymouth Index: 10899263)
10899263@students.plymouth.ac.uk