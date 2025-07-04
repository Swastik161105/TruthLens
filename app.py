import streamlit as st
import joblib
import re
import os
import gdown

os.makedirs("models", exist_ok=True)

if not os.path.exists("models/combined_model.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1NSEQkymAz7HnSmXUpN5LRrCH3S8R4sE8",
        "models/combined_model.pkl",
        quiet=False
    )



# Load all models and vectorizers
title_model = joblib.load("models/title_model.pkl")
full_model = joblib.load("models/combined_model.pkl")
title_vectorizer = joblib.load("vectorizers/title_vectorizer.pkl")
full_vectorizer = joblib.load("vectorizers/combined_vectorizer.pkl")



def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)  
    text = re.sub(r"[^a-z\s]", "", text)         
    text = re.sub(r"\s+", " ", text).strip()     
    return text

st.markdown("""
            <style>
            .st-emotion-cache-1j22a0y.e4x2yc34{
                visibility:hidden;
            }
            
            .stColumn.st-emotion-cache-k7yhar.eertqu01{
                border:2px solid black;
            }
            
            .st-emotion-cache-ajtf3x{
                background-color:black;
            }
            
            .stMain.st-emotion-cache-z4kicb.elbt1zu1{
                background-color:#191970;
            }
            
            
            .stRadio{
            
                padding: 20px 45px 20px 10px;
            }
            
            
            
            .stMainBlockContainer.block-container.st-emotion-cache-1w723zb.elbt1zu4{
                border: 2px solid;
                margin-top: 50px;
                padding-top: 10px;
                padding-left:20px;
                padding-right:20px;
                margin-bottom: 50px;
                padding-bottom: 20px;
                border-radius: 30px;
                background-color:white;
                display:flex;
                justify-content:center;
                align-items:center;
            }
            </style>""",unsafe_allow_html=True)



col1, col2 = st.columns([1, 5])
with col1:
    st.image("truthlens.png", width=100)
with col2:
    st.markdown("""    
            <div style="color:white;
                background-color:black;
                padding: 10px 10px 10px 10px;
                border: 2px solid black;
                text-align:center;
                margin:0;
                padding:0;">
                <h2 style="margin=0;padding=0;">TruthLens : Fake News Detector</h2>
            </div>""",unsafe_allow_html=True)





st.markdown("""<div style="color:black;
                background-color:lightblue;
                padding: 10px 10px 10px 10px;
                border: 2px solid black;
                text-align:center;">
                <h3>Check if your news is real or fake →</h3></div>""",unsafe_allow_html=True)


st.markdown("""<div style="background-color:lightyellow;border:2px solid black;text-size:large;text-align:center;border-top:none;border-bottom:none;"><h3>How it works:</h3></div>""",unsafe_allow_html=True)


st.markdown("""
            <div style="border:2px solid black;background-color:white;">
            <ol>
            <li>Pick any one basis of fake news detection as your choice.</li>
            <li>If you give <b>"News Title"</b> as your choice, we will analyze your News Title to give you a prediction.</li>
            <li>If the above prediction seems uncertain or you want stronger confirmation, you can choose <b>"Full News (Title + Text)"</b> in the second option.</li>
            </ol>
            </div>""",unsafe_allow_html=True)


st.markdown("""<div style="background-color: lightyellow; padding: 10px;border: 2px solid black;border-top:none;"><b>This tool uses AI to detect potential fake news. For best accuracy, use full news content. This is not a replacement for official fact-checking.</b></div>""",unsafe_allow_html=True)


option=st.radio("**Pick one basis of detection :** ",
                ["News Title","Full News (Title + Text)"])


if option == "News Title":
    title=st.text_input("**News Title:**")
    if st.button("**Analyze**"):
        if not title.strip():
            st.write("Please paste the news title")
            
        else:
            cleaned=clean_text(title)
            cleaned_vect=title_vectorizer.transform([cleaned])
            pred=title_model.predict(cleaned_vect)[0]
            prob=title_model.predict_proba(cleaned_vect)[0]
            if(pred==0):
                st.error(f"FAKE NEWS, Confidence: {prob[0]*100:.2f}%")
                st.info("This prediction is based on the title only.\n"
            "For better accuracy, please analyze the full article below.")
            else:
                st.success(f"REAL NEWS, Confidence: {prob[1]*100:.2f}%")
                st.info("The title seems trustworthy.\n"
            "You may still check the full article to be sure.")
else:
    full_news=st.text_input("**Full News:**")
    if st.button("**Analyze**"):
        if not full_news.strip():
            st.write("Please paste the full news article")
        else:
            cleaned=clean_text(full_news)
            cleaned_vect=full_vectorizer.transform([cleaned])
            pred=full_model.predict(cleaned_vect)[0]
            prob=full_model.predict_proba(cleaned_vect)[0]
            if(pred==0):
                st.error(f"FAKE NEWS, Confidence: {prob[0]*100:.2f}%")
                st.caption("This is the final and more reliable prediction.")
            else:
                st.success(f"REAL NEWS, Confidence: {prob[1]*100:.2f}%")
                st.caption("Full article confirms the reliability of the title.")

