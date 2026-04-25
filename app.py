import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stSlider > div > div > div > div {
        color: #00ff88;
    }
    .result-card {
        padding: 2rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 2rem;
    }
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .win-h { color: #00ff88; }
    .win-a { color: #ff4b4b; }
    .draw { color: #ffd700; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists('model.pkl'):
        return None
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    # Header Image
    st.image("header.png", use_container_width=True)
    
    st.title("⚽ Football Match Predictor")
    st.markdown("Predict match results based on live or expected team statistics using **XGBoost**.")

    data = load_model()
    
    if data is None:
        st.error("Model not found! Please run `python train.py` first to generate `model.pkl`.")
        return

    model = data['model']
    encoder = data['encoder']
    features = data['features']

    # Sidebar for inputs
    st.sidebar.header("Match Statistics")
    st.sidebar.markdown("Adjust the sliders to simulate match stats.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Home Team")
        hs = st.slider("Shots", 0, 40, 10, key="hs")
        hst = st.slider("Shots on Target", 0, 20, 4, key="hst")
        hc = st.slider("Corners", 0, 20, 5, key="hc")
        hf = st.slider("Fouls", 0, 30, 10, key="hf")
        hy = st.slider("Yellow Cards", 0, 10, 1, key="hy")
        hr = st.slider("Red Cards", 0, 5, 0, key="hr")

    with col2:
        st.subheader("🚩 Away Team")
        as_stats = st.slider("Shots", 0, 40, 10, key="as")
        ast = st.slider("Shots on Target", 0, 20, 4, key="ast")
        ac = st.slider("Corners", 0, 20, 5, key="ac")
        af = st.slider("Fouls", 0, 30, 10, key="af")
        ay = st.slider("Yellow Cards", 0, 10, 1, key="ay")
        ar = st.slider("Red Cards", 0, 5, 0, key="ar")

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Predict Match Outcome", use_container_width=True, type="primary"):
        # Prediction logic
        input_data = pd.DataFrame([[hs, as_stats, hst, ast, hc, ac, hf, af, hy, ay, hr, ar]], columns=features)
        
        # Probability distribution
        probs = model.predict_proba(input_data)[0]
        classes = encoder.classes_
        
        # Result mapping
        result_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        prediction_idx = np.argmax(probs)
        result_code = classes[prediction_idx]
        result_text = result_map[result_code]
        
        # Visual Output
        st.divider()
        
        color_class = "win-h" if result_code == 'H' else "win-a" if result_code == 'A' else "draw"
        
        st.markdown(f"""
            <div class="result-card">
                <div class="prediction-text {color_class}">{result_text}</div>
                <p>Confidence: {probs[prediction_idx]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # Probabilities Chart
        st.subheader("Probability Breakdown")
        prob_df = pd.DataFrame({
            'Result': [result_map[c] for c in classes],
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index('Result'))

        # Metadata / Stats
        cols = st.columns(3)
        cols[0].metric("Home Attack Index", f"{hs + hst*2}")
        cols[1].metric("Away Attack Index", f"{as_stats + ast*2}")
        cols[2].metric("Total Aggression", f"{hf + af + (hy+ay)*5 + (hr+ar)*15}")
    else:
        st.info("👈 Adjust the statistics on the left and click the button to see the prediction!")

if __name__ == "__main__":
    main()
