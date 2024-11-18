import streamlit as st
import pandas as pd
import numpy as np
from utils import prepare_input, make_predictions, explain_prediction, create_gauge_chart, create_model_probability_chart, load_models
import os

st.title("Credit Card Fraud Detection")

# Load models once at the beginning
models = load_models()

# Form for user inputs
st.header("Transaction Details")

# Transaction Amount
amt = st.number_input("Transaction Amount (amt)", min_value=0.0)

# Transaction Category
categories = [
    'gas_transport', 'grocery_net', 'grocery_pos', 'health_fitness',
    'home', 'kids_pets', 'misc_net', 'misc_pos', 'personal_care',
    'shopping_net', 'shopping_pos', 'travel'
]
category = st.selectbox("Transaction Category", categories)

# State
states = [
    'NY', 'CA', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI',
    # Add more states as per your dataset
]
state = st.selectbox("State", states)

# Optionally, add input fields for 'job', 'street', 'first', 'last' if feasible
# For simplicity, we'll use default values for these features

if st.button("Predict"):
    # Prepare input data
    input_df, input_dict = prepare_input(
        amt=amt,
        category=category,
        state=state,
        # You can add default values for other features here
    )

    # Make predictions, passing models as an argument
    avg_probability, probabilities = make_predictions(input_df, models)

    # Generate explanation
    explanation = explain_prediction(input_dict)

    # Display Gauge and Bar Chart
    st.subheader("Fraud Probability")
    fig = create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Probabilities")
    fig_probs = create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

    # Display explanation
    st.subheader("Explanation")
    st.write(explanation)
