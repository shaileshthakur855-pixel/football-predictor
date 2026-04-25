import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

def train_model():
    print("Loading data...")
    if not os.path.exists('matches.csv'):
        print("Error: matches.csv not found.")
        return

    df = pd.read_csv('matches.csv', encoding='latin1')
    
    # Selecting relevant features for a 'what-if' predictor
    features = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    target = 'FTR'
    
    # Drop rows where stats are missing (older seasons)
    df_clean = df.dropna(subset=features + [target])
    
    print(f"Original data size: {len(df)}")
    print(f"Cleaned data size: {len(df_clean)}")
    
    X = df_clean[features]
    y = df_clean[target]
    
    # Encode target (H, D, A)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train XGBoost
    print("Training model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")
    
    # Save model and encoder
    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'encoder': le, 'features': features}, f)
    
    print("Done!")

if __name__ == "__main__":
    train_model()
