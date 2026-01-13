Diabetes Prediction using Machine Learning

This project implements a Diabetes Prediction System using a Support Vector Machine (SVM) classifier. The model predicts whether a person is diabetic based on key medical parameters such as glucose level, BMI, age, insulin level, etc.

The system is trained on the Pima Indians Diabetes Dataset and allows users to input data via the terminal to get real-time predictions.

⸻

Features
	•	Uses SVM (Support Vector Machine) for classification
	•	Data preprocessing using StandardScaler
	•	Accepts comma-separated user input
	•	Outputs whether the person is Diabetic or Not Diabetic
	•	Includes a requirements.txt for easy environment setup
Dataset

The dataset used contains the following attributes=> Feature : Description
Pregnancies: Number of pregnancies
Glucose : Plasma glucose concentration
BloodPressure : Diastolic blood pressure
SkinThickness : Triceps skin fold thickness
Insulin : Serum insulin
BMI : Body Mass Index
DiabetesPedigreeFunction : Genetic diabetes risk
Age : Ageief the patient
Outcome : 0 = Non-Diabetic, 1 = Diabetic

_________

Installation

Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:
pip install -r requirements.txt

_____

How to Run

Navigate to the model directory:
cd model

Run the script:
python3 diabetes_prediction.py

Enter values in this format when prompted:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age

Example:
5,166,72,19,175,25.8,0.587,51

Output:
The person is diabetic
or
The person is not diabetic

_____

Technologies Used
	•	Python
	•	NumPy
	•	Pandas
	•	Scikit-Learn
	•	Support Vector Machine (SVM)

______

Project Structure:

Diabetes Prediction/
│
├── Data/
│   └── diabetes.csv
│
├── model/
│   └── diabetes_prediction.py
│
├── requirements.txt
└── README.md

_____

Model Accuracy

The model prints accuracy for:
	•	Training Data
	•	Testing Data

This allows validation of model performance.

____

Author

Vansh Saxena
Machine Learning | Python | Data Science
