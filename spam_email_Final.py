import streamlit as st
import joblib 
import pandas as pd

# Load the model
spam = joblib.load('Spam_Emial_Detector.joblib')
model = spam['model']
vectorizer = spam['vectorizer']

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp {
        background-color: transparent;
    }
    
    .header {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 600;
    }
    
    
    
    .stTextArea textarea {
        min-height: 200px !important;
        border-radius: 10px !important;
        border: 1px solid #dfe6e9 !important;
        padding: 15px !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    
    .result-box {
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .ham {
        background: linear-gradient(135deg, #a1ffce 0%, #faffd1 100%);
        color: #27ae60;
    }
    
    .spam {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: #e74c3c;
    }
    
    .warning {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #d35400;
    }
    
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #7f8c8d;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# App layout
st.markdown('<h1 class="header">📧 Smart Email Analyzer</h1>', unsafe_allow_html=True)

with st.container():
    
    text = st.text_area("Enter the email content below:", placeholder="Please enter the email content here...", height=200)
    send = st.button("Analyze Email")
    st.markdown('</div>', unsafe_allow_html=True)

if send and text:
    new_inp = {'Masseges': text}
    inp_df = pd.DataFrame([new_inp])
    vec_inp = vectorizer.transform(inp_df['Masseges'])
    prediction = model.predict(vec_inp)
    
    if prediction[0] == 'ham':
        st.markdown('<div class="result-box ham">✅ This email is NOT SPAM.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box spam">⚠️ This email is SPAM. Be cautious with this message!</div>', unsafe_allow_html=True)

elif text == '' and send:
    st.markdown('<div class="result-box warning">Please enter the email content before analyzing.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Smart Email Analyzer | Powered by Machine Learning</div>', unsafe_allow_html=True)