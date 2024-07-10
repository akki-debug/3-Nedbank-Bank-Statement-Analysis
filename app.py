import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import re
from sklearn.tree import DecisionTreeClassifier

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

# Function to predict loan eligibility (dummy model for demonstration)
def predict_eligibility(df):
    # Dummy feature extraction: use balance and credits
    X = df[["Balance (R)", "Credits (R)"]].values
    y = (X[:, 0] > 5000).astype(int)  # Dummy target: eligible if balance > 5000
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    predictions = model.predict(X)
    df["Eligibility"] = predictions
    return df

st.title("Bank Statement Loan Eligibility Checker")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    df = parse_text_to_df(text)
    
    # Combine data extraction and eligibility prediction into a single display
    df = predict_eligibility(df)
    
    st.write("Bank Statement Data and Eligibility Predictions:")
    st.dataframe(df, height=800)  # Display all rows, adjust height as needed
    
    # Display eligibility as one-liner with color
    st.write("Eligibility Status:")
    for index, row in df.iterrows():
        if row['Eligibility'] == 1:
            st.write(f"{row['Description']} - <span style='color:green'>Eligible</span>")
        else:
            st.write(f"{row['Description']} - <span style='color:red'>Not Eligible</span>")
