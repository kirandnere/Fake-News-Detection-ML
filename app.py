import streamlit as st
import pickle
import re
import string

# 1. Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# 2. Text Cleaning Function (Same as before)
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    if n == 0:
        return "Fake News ⚠️"
    elif n == 1:
        return "Not A Fake News ✅"

# 3. Streamlit Website UI
st.title("🕵️‍♂️ Fake News Detector")
st.write("Enter a news article below to check if it is Real or Fake.")

# Input Box
news_text = st.text_area("Paste News Text Here:", height=200)

# Button
if st.button("Check News"):
    if news_text:
        # Process the input
        clean_text = wordopt(news_text)
        new_x_test = [clean_text]
        new_xv_test = vectorizer.transform(new_x_test)
        pred_LR = model.predict(new_xv_test)

        # Show Result
        result = output_label(pred_LR[0])

        if pred_LR[0] == 0:
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text first.")
