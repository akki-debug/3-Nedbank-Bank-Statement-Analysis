import streamlit as st
import pdfplumber
import re
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Function to parse PDF and extract data
def parse_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to process parsed text into a DataFrame
def process_text_to_df(text):
    transactions = []
    transaction_pattern = re.compile(r'(\d{2}/\d{2}/\d{4})\s+(.+?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+(-?R?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')
    
    for line in text.split('\n'):
        match = transaction_pattern.search(line)
        if match:
            date_str, description, amount_str, balance_str = match.groups()
            amount = float(amount_str.replace(',', '').replace('R', '').replace(' ', ''))
            balance = float(balance_str.replace(',', '').replace('R', '').replace(' ', ''))
            transactions.append([date_str, description.strip(), amount, balance])
    
    return pd.DataFrame(transactions, columns=['Date', 'Description', 'Amount', 'Balance'])

# Function to categorize expenses based on descriptions
def categorize_expense(description):
    description_lower = description.lower()
    if 'cashsend mobile' in description_lower:
        return 'POS Purchases'
    elif 'immediate trf' in description_lower or 'digital payment' in description_lower:
        return 'Payments'
    elif 'acb credit' in description_lower or 'immediate trf cr' in description_lower:
        return 'Credits'
    elif 'fees' in description_lower or 'charge' in description_lower:
        return 'Bank Charges'
    elif 'atm' in description_lower or 'cash deposit' in description_lower:
        return 'Cash Deposits/Withdrawals'
    elif 'airtime' in description_lower:
        return 'Cellular Expenses'
    elif 'electricity' in description_lower:
        return 'Electricity Charges'
    elif 'interest' in description_lower:
        return 'Interest and Fees'
    elif 'unsuccessful' in description_lower:
        return 'Unsuccessful Transactions'
    elif 'realtime credit' in description_lower:
        return 'Real-time Credits'
    else:
        return 'Others'

# Function to compute key metrics
def compute_metrics(df):
    avg_daily_expense = df['Amount'].mean()
    total_expense = df['Amount'].sum()
    max_expense = df['Amount'].max()
    min_expense = df['Amount'].min()
    num_transactions = len(df)
    return avg_daily_expense, total_expense, max_expense, min_expense, num_transactions

# Function to train the decision tree model
def train_decision_tree_model(data):
    X = data[['Closing Balance', 'Credit Transactions', 'Total Credit']]
    y = data['Eligibility']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Function to predict eligibility using the decision tree model
def predict_eligibility(model, df):
    total_credits = df[df['Amount'] > 0]['Amount'].sum()
    total_debits = df[df['Amount'] < 0]['Amount'].sum()
    closing_balance = df['Balance'].iloc[-1]
    credit_transactions = df[df['Amount'] > 0].shape[0]
    
    input_data = pd.DataFrame([[closing_balance, credit_transactions, total_credits]],
                              columns=['Closing Balance', 'Credit Transactions', 'Total Credit'])
    
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit application
def main():
    st.title('Upload Your Bank Statement')
    
    st.write(
        """
        <style>
        .success-message {
            font-size: 1.25rem;
            color: green;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="success-message">Congratulations! You\'ve successfully passed the initial eligibility check. Please submit your bank statement so we can complete the final eligibility review.</p>', unsafe_allow_html=True)

    # File upload for PDF statement
    uploaded_file = st.file_uploader('Please upload your bank statement', type='pdf')
    
    if uploaded_file is not None:
        st.write(f'Filename: {uploaded_file.name}')

        # Parse PDF
        text = parse_pdf(uploaded_file)
        
        # Process parsed text into DataFrame
        df = process_text_to_df(text)
        
        # Convert 'Date' to datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        if not df.empty:
            df['Category'] = df['Description'].apply(categorize_expense)
            
            # Train the model using a provided dataset
            nedbank_data = pd.read_csv('nedbank.csv')  # Using 'nedbank.csv' dataset
            model, accuracy = train_decision_tree_model(nedbank_data)
            st.write(f'Model accuracy: {accuracy * 100:.2f}%')

            # Predict eligibility using the trained model
            eligibility = predict_eligibility(model, df)
            if eligibility == 1:
                st.markdown('<p style="color:green;">The user is eligible for a loan.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red;">The user is not eligible for a loan.</p>', unsafe_allow_html=True)

            st.subheader('Parsed Data')
            st.write(df)

            avg_daily_expense, total_expense, max_expense, min_expense, num_transactions = compute_metrics(df)

            st.subheader('Key Metrics')
            st.write(f'Average Daily Expense: R{avg_daily_expense:.2f}')
            st.write(f'Total Expense: R{total_expense:.2f}')
            st.write(f'Maximum Expense: R{max_expense:.2f}')
            st.write(f'Minimum Expense: R{min_expense:.2f}')
            st.write(f'Number of Transactions: {num_transactions}')

            st.subheader('Expense Overview')

            fig_bar = px.bar(df, x='Date', y='Amount', color='Category', title='Total Expenses per Date')
            st.plotly_chart(fig_bar)

            fig_pie_category = px.pie(df, values='Amount', names='Category', title='Expense Distribution by Category')
            st.plotly_chart(fig_pie_category)

            fig_pie_description = px.pie(df, values='Amount', names='Description', title='Expense Distribution by Description')
            st.plotly_chart(fig_pie_description)

            fig_line = px.line(df, x='Date', y='Amount', title='Daily Expense Trend')
            st.plotly_chart(fig_line)
        else:
            st.write("No transactions found in the uploaded statement.")
    else:
        st.write("Please upload a PDF file.")

# Entry point
if __name__ == '__main__':
    main()
