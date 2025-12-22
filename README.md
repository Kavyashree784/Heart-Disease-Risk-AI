# Heart Attack Prediction System

## Overview
This project is a machine learning application designed to predict the likelihood of a heart attack based on physiological data. It utilizes advanced algorithms including **XGBoost**, **Random Forest**, and **Logistic Regression** to analyze health metrics such as blood pressure, heart rate, and cholesterol levels.

## Features
* **Predictive Analysis:** Uses historical data to predict heart disease risk.
* **Interactive Interface:** A user-friendly web app built with Streamlit.
* **High Accuracy:** Optimized using ensemble learning techniques.

## Dataset
The model is trained on the **Cleveland Heart Disease Dataset**, considering 14 standard clinical features including Age, Sex, Chest Pain Type, and more.

## Installation & Usage
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd Heart-Attack-Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Train the model:**
    ```bash
    python train_model.py
    ```
4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Model Performance
The system was evaluated using Accuracy, Precision, and Recall. **XGBoost** demonstrated the highest performance, making it the primary model for the live application.

## Technologies
* Python
* Scikit-Learn & XGBoost
* Streamlit
* Pandas & NumPy