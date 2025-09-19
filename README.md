Specific Heart Disease Prediction App
A collaborative project by our team to build a machine learning web application that predicts specific heart conditions based on 13 clinical parameters. This tool is built with a Python Flask backend and a simple HTML/Tailwind CSS frontend.

🌟 Features
Multi-Class Prediction: Predicts one of several conditions (e.g., Coronary Artery Disease, Cardiomyopathy, Arrhythmia) or "No Disease Detected".

Web-Based Interface: Simple and intuitive UI for entering patient data.

Self-Contained Model: The backend trains a RandomForestClassifier on an embedded dataset, so no external model files are needed.

Dynamic Results: The result box is color-coded (red for a diagnosed condition, green for no disease) for clear visual feedback.

👥 Our Team | Contributors
This project was brought to life by the collaborative efforts of:

[Enter Team Member 1 Name] - GitHub: @username

[Enter Team Member 2 Name] - GitHub: @username

[Enter Team Member 3 Name] - GitHub: @username

(Add more members as needed)

🛠️ Technologies Used
Backend:

Python

Flask & Flask-CORS

Pandas

Scikit-learn

Frontend:

HTML

Tailwind CSS

JavaScript (Fetch API)

🚀 Local Setup and Installation
To run this project on your local machine, follow these steps:

1. Clone the Repository:

git clone [https://github.com/pap83/Heart-Disease-Prediction.git](https://github.com/pap83/Heart-Disease-Prediction.git)
cd Heart-Disease-Prediction

2. Create and Activate a Virtual Environment:

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

3. Install Dependencies:

pip install Flask Flask-CORS pandas scikit-learn

▶️ How to Run
Start the Backend Server:
With the virtual environment active, run the Flask app:

python app.py

The server will start on http://127.0.0.1:5000.

Open the Frontend:
Open the index.html file directly in your web browser.