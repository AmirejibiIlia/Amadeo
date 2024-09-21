import pandas as pd
import spacy
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Streamlit App Title
st.title("AmadeoChat: Financial Question Answering")

# File uploader for the Excel file
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    st.write("Data loaded successfully!")

    # Display the first few rows of the data
    st.write(df.head())

    # Load the pre-trained BART model and tokenizer from Huggingface
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Load the Spacy NLP model (en_core_web_sm)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.write("Downloading Spacy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Text area for user to input a financial question
    user_input = st.text_area("Enter your financial question:")

    # Process and generate an answer when button is clicked
    if st.button("Get Answer"):
        if user_input:
            # Apply Spacy NLP to user input for entity recognition
            doc = nlp(user_input)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.write("Entities detected in the input:", entities)

            # Tokenize the user input and generate a BART summary/answer
            inputs = tokenizer([user_input], max_length=1024, return_tensors='pt')
            summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            # Display the BART-generated answer
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question to proceed.")
else:
    st.write("Please upload an Excel file to proceed.")
