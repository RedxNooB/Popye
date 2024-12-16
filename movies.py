# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:52:09 2024

@author: d
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
st.title("Movie Gross Revenue Predictor")
st.text("Predict how much your favorite movie could earn!")
st.image("movies.png", caption="Explore Box Office Predictions!", use_column_width=True)

# Load the dataset
df = pd.read_csv('movies.csv')

# Show a preview of the dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Select features and target
st.header("Select Features for Prediction")
multi_var = st.multiselect(
    "Select features to include in the prediction:",
    ['budget', 'runtime', 'score', 'votes'],
    default=['budget', 'runtime']
)

# Select regression model
sel_box_var = st.selectbox("Select Regression Method", ['Linear', 'Ridge', 'Lasso'], index=0)

# Prepare data for modeling
X = df[multi_var].fillna(0)  # Fill NaN with zeros for simplicity
Y = df['gross'].fillna(0)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the selected model
if sel_box_var == 'Linear':
    model = LinearRegression()
elif sel_box_var == 'Lasso':
    model = Lasso()
else:
    model = Ridge()

reg = model.fit(X_train, Y_train)
Y_pred = reg.predict(X_test)

# Display model results
st.text(f"Intercept: {reg.intercept_}")
st.text(f"Coefficients: {reg.coef_}")
st.text(f"R² Score: {r2_score(Y_test, Y_pred)}")

# Prediction section
st.header("Make a Prediction")
input_data = {}
for feature in multi_var:
    input_data[feature] = st.number_input(f"Enter {feature}", min_value=0.0)

# Convert input to DataFrame and predict
input_df = pd.DataFrame([input_data])
predicted_gross = reg.predict(input_df)[0]
st.text(f"Predicted Gross Revenue: ${predicted_gross:,.2f}")

# Footer
st.text("Thank you for using the Movie Gross Revenue Predictor!")
