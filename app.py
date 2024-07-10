import streamlit as st
import pdfplumber
import re
import pandas as pd
import numpy as np

st.set_page_config(page_title='Bank Statement Data Extraction & Loan Eligibility', page_icon=':moneybag:')

st.title('Extract Data from Bank Statement PDF')
st.write('Upload your bank statement PDF to extract transaction data and check loan eligibility')

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Open the uploaded PDF file with pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Extract transactions from the text using regex
        transactions = []
        # Regular expression to match transactions
        transaction_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+([^\d]+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})')

        for line in text.split('\n'):
            match = transaction_pattern.search(line)
            if match:
                date_str, description, fees_str, debits_str, credits_str = match.groups()
                # Removing unwanted characters and converting to float
                fees = float(fees_str.replace(',', '').replace('R', '').replace(' ', ''))
                debits = float(debits_str.replace(',', '').replace('R', '').replace(' ', ''))
                credits = float(credits_str.replace(',', '').replace('R', '').replace(' ', ''))
                transactions.append([date_str, description.strip(), fees, debits, credits])

        # Convert transactions to a DataFrame
        df = pd.DataFrame(transactions, columns=['Date', 'Description', 'Fees', 'Debits', 'Credits'])
        
        # Convert 'Date' to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Display the extracted data
        st.subheader('Extracted Transactions')
        st.dataframe(df, use_container_width=True)

        # Feature extraction
        total_credits = df['Credits'].sum()
        total_debits = df['Debits'].sum()

        # Loan eligibility based on specified conditions
        eligible = total_credits > abs(total_debits) and total_credits > 1.25 * abs(total_debits)

        # Display result with color
        result = 'Eligible for Loan' if eligible else 'Not Eligible for Loan'
        color = 'green' if eligible else 'red'
        st.markdown(f'<p style="color:{color};font-size:24px;">{result}</p>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

hide_streamlit_style = """
                    <style>
                    # MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    footer:after {
                    content:'Made with ❤️ by Akshat'; 
                    visibility: visible;
                    display: block;
                    position: relative;
                    padding: 15px;
                    top: 2px;
                    }
                    </style>
                    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
