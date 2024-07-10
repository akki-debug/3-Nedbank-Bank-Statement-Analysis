import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import re

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ''
    for page_num in range(len(pdf.pages)):
        text += pdf.pages[page_num].extract_text()
    return text

# Function to parse extracted text into a DataFrame
def parse_text_to_df(text):
    rows = []
    lines = text.split('\n')
    transaction_pattern = re.compile(r'(\d{6})\s+(\d{2}/\d{2}/\d{4})\s+(.+?)\s+((?:\d{1,3},?)+\.\d{2})?\s+((?:\d{1,3},?)+\.\d{2})?\s+((?:\d{1,3},?)+\.\d{2})?\s+((?:\d{1,3},?)+\.\d{2})')

    for line in lines:
        match = transaction_pattern.match(line)
        if match:
            rows.append(match.groups())

    columns = ["Tran list no", "Date", "Description", "Fees (R)", "Debits (R)", "Credits (R)", "Balance (R)"]
    df = pd.DataFrame(rows, columns=columns)
    
    # Convert numeric columns from strings to floats
    for col in ["Fees (R)", "Debits (R)", "Credits (R)", "Balance (R)"]:
        df[col] = df[col].str.replace(',', '').astype(float)
        
    return df

st.title("Bank Statement Data Extractor")

# Function to display DataFrame with full content of each cell
def display_full_dataframe(df):
    # Display column names and values
    for col in df.columns:
        st.write(f"**{col}**")
        for value in df[col]:
            st.write(value)
        st.write("---")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    df = parse_text_to_df(text)
    
    st.write("Extracted Bank Statement Data:")
    display_full_dataframe(df)
