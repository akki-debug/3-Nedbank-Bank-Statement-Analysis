import streamlit as st
import PyPDF2
import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# Function to parse financial summary from text
def parse_financial_summary(text):
    summary = {
        "Cash withdrawals": 0,
        "Electronic payments received": 0,
        "Debit card purchase": 0,
        "Investment repayments": 0,
        "Electronic transfers": 0,
        "Transfers in": 0,
        "Transfers out": 0,
        "Total charges and fees": 0,
        "Other credits": 0,
        "Other debits": 0
    }
    
    pattern = re.compile(r'(Cash withdrawals|Electronic payments received|Debit card purchase|Investment repayments|Electronic transfers|Transfers in|Transfers out|Total charges and fees|Other credits|Other debits) R([\d,]+.\d{2})')
    matches = pattern.findall(text)
    
    for match in matches:
        category, amount = match
        summary[category] = float(amount.replace(',', ''))
    
    return summary

# Function to determine loan eligibility
def determine_eligibility(summary):
    features = np.array(list(summary.values())).reshape(1, -1)
    prediction = model.predict(features)
    return "Eligible" if prediction[0] == 1 else "Not Eligible"

# Load the decision tree model (replace with your actual model file)
model = DecisionTreeClassifier()
# Dummy training for illustration (replace with actual model loading)
X_dummy = np.random.rand(100, 10)
y_dummy = np.random.randint(2, size=100)
model.fit(X_dummy, y_dummy)

# Streamlit UI
st.title("Loan Eligibility Determination")

uploaded_file = st.file_uploader("Upload Bank Statement PDF", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    
    # Display the extracted text
    st.subheader("Extracted Text")
    st.write(text)
    
    # Parse financial summary
    summary = parse_financial_summary(text)
    
    # Display the financial summary
    st.subheader("Financial Summary")
    st.write(summary)
    
    # Determine loan eligibility
    eligibility = determine_eligibility(summary)
    
    # Display loan eligibility
    st.subheader("Loan Eligibility")
    st.write(eligibility)

# Example of model evaluation (remove in production)
st.subheader("Model Evaluation (Example)")

# Split the dummy data
X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display the evaluation metrics
st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.write("Classification Report")
st.write(classification_report(y_test, y_pred))
