import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from nltk.stem import PorterStemmer

def preprocess_text(text):
    
    """Preprocess text"""
    
    
    text = text.lower()
    
    # replace all links with url
    text = re.sub(r"http[s]?://[\w\/\-\.\?]+", 'url', text)
    
    # replace hastags with the word after the hastag
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # replace digits with numbers
    text = re.sub(r'\d+', 'num', text) 
    
    # replace punctuations
    punctuation_pattern = r'[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    
    text = re.sub(punctuation_pattern, '', text)
    
    # replace @user with username
    text = re.sub(r'@\w+', 'username', text)
    

    # return words
    text = ' '.join([x for x in re.findall(r'\w+\s+', text)])
        
    stemmer = PorterStemmer()
    
    value =  ' '.join([stemmer.stem(word) for word in text.split()])
    
    df = pd.DataFrame(value, index = [0], columns = ['tweet'])
    
    return df

filename = 'Hate_and_Offensive_speech_model.pkl'


#load model
model = joblib.load(filename)

def predict_hate_speech(text):
            
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
        
    # Make predictions using the trained model
    prediction = model.predict(preprocessed_text['tweet'])
                
    if prediction == 0:
            return f'{prediction}: Speech does not contain hateful or offensive speech'
            
    return f'{prediction}: Speech contains hateful or offensive speech.'

# Streamlit app
def main():
    
    # title
    st.title("Hate speech and Offensive speech Detection")

    # description
    st.markdown("""This model was trained on 24783 tweets to detect whether a model contains hate or offensive speech.      
                 --- > Train Score: 99%
                 ---> Test auc: 70%
                 ---> Validation auc: 77%""")

    st.markdown("""Note: Model is overfitting and may not generalize well ☹.\n
                To do: Model generalization will be improved""")

    
    # Input text from the user
    user_input = st.text_area("Enter Text:", "")
    
    if st.button("Detect Hate Speech"):
                
        if user_input:
                
            # Make predictions and display the result
            prediction = predict_hate_speech(user_input)
            st.write(f"Prediction: {prediction}")
                
        else:
            st.warning("Please enter a tweet.")

if __name__ == "__main__":
    main()
