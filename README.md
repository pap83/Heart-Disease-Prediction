Specific Heart Disease Prediction App
A machine learning web application to predict specific heart conditions based on 13 clinical parameters. This tool is built with a Python Flask backend and a simple HTML/Tailwind CSS frontend.

🌟 Features
Multi-Class Prediction: Predicts one of several conditions (e.g., Coronary Artery Disease, Cardiomyopathy, Arrhythmia) or "No Disease Detected".

Web-Based Interface: Simple and intuitive UI for entering patient data.

Self-Contained Model: The backend trains a RandomForestClassifier on an embedded dataset, so no external model files are needed.

Dynamic Results: The result box is color-coded (red for a diagnosed condition, green for no disease) for clear visual feedback.

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

# Activate on macOS/Linux
# source venv/bin/activate

3. Install Dependencies:
The required Python packages are listed in the app.py imports. Install them with pip:

pip install Flask Flask-CORS pandas scikit-learn

▶️ How to Run
Start the Backend Server:
Once your virtual environment is active and dependencies are installed, run the Flask app from the terminal:

python app.py

The server will start on http://127.0.0.1:5000.

Open the Frontend:
Navigate to the project folder on your computer and open the index.html file directly in your web browser.

Use the Application:
Fill in the medical data in the form and click the "Predict" button to see the result.

⚠️ Disclaimer
This tool is for educational and demonstrational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.