# Football Match Outcome Predictor ⚽🤖

**Live Application:** https://football-predictor-8kokbmvbgpgxixqxdrxlnf.streamlit.app/

## Overview
This project is an end-to-end Machine Learning web application that predicts the outcome of football matches (Home Win, Away Win, or Draw) based on historical team statistics. 

The model was trained on multi-season match data using **XGBoost**, utilizing key performance indicators such as shots on target, corners, fouls, and card counts. The frontend is built and deployed using **Streamlit**, allowing users to dynamically adjust match statistics to simulate real-time predictions.

## Tech Stack
* **Language:** Python 3.13
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Manipulation:** Pandas, NumPy
* **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
* **Model Serialization:** Joblib

## How It Works
1. **Data Processing:** The script (`train.py`) loads historical match data, cleans missing values, and encodes the target variable (Full Time Result: H, D, A).
2. **Model Training:** An XGBoost Classifier is trained on the numerical features to understand the correlations between match stats and the final outcome. The trained model is saved as `model.pkl`.
3. **Inference:** The Streamlit app (`app.py`) loads the pre-trained model and accepts user input via interactive sliders to output a predicted probability for the match.

## How to Run Locally
If you want to run this project on your own machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone [https://github.com/shaileshthakur855-pixel/football-predictor.git](https://github.com/shaileshthakur855-pixel/football-predictor.git)
