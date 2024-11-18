import pandas as pd
import numpy as np
import pickle
import os
import requests
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = 'https://api.gemini.com/v1/predict'

# Load models globally
models = {}

def load_models():
    global models
    if models:
        return models  # Models already loaded

    model_files = {
        'XGBoost': 'xgb_model.pkl',
        'Logistic Regression': 'lr_balanced_model.pkl',
        'Easy Ensemble': 'eec_model.pkl',
        'Decision Tree': 'dt_balanced_model.pkl',
        'CatBoost': 'cat_model.pkl',
        'Balanced Bagging': 'bbc_model.pkl'
    }

    for model_name, file_name in model_files.items():
        try:
            with open(f'models/{file_name}', 'rb') as file:
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            print(f"Model file {file_name} not found.")
        except Exception as e:
            print(f"Error loading model {file_name}: {e}")
    return models

def prepare_input(amt, category, state):
    # Default values
    default_first = 'John'
    default_last = 'Doe'
    default_street = '123 Main St'
    default_job = 'Professional'

    input_dict = {
        'amt': amt,
        'category': category,
        'state': state,
        'first': default_first,
        'last': default_last,
        'street': default_street,
        'job': default_job
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical variables
    categorical_cols = ['category', 'state', 'first', 'last', 'street', 'job']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)

    # Ensure all expected columns are present
    expected_cols = models_feature_columns()

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]

    return input_df, input_dict

def models_feature_columns():
    # Return the list of feature columns expected by the models
    xgb_model = models.get('XGBoost')
    if xgb_model:
        try:
            feature_names = xgb_model.get_booster().feature_names
            return feature_names
        except AttributeError:
            pass
    # Fallback: Hardcoded feature names
    return ['amt', 'category_gas_transport', 'category_grocery_net', 'state_NY', 'first_John', 'last_Doe', 'job_Professional', 'street_123 Main St']

def make_predictions(input_df, models):
    probabilities = {}
    for model_name, model in models.items():
        try:
            prob = model.predict_proba(input_df)[0][1]
            probabilities[model_name] = prob
        except Exception as e:
            print(f"Error in {model_name} model: {e}")

    # Get Gemini API prediction (optional)
    # gemini_prob = make_gemini_prediction(input_df)
    # if gemini_prob is not None:
    #     probabilities['Gemini API'] = gemini_prob

    if probabilities:
        avg_probability = np.mean(list(probabilities.values())) * 100  # Convert to percentage
    else:
        avg_probability = 0
        print("No valid model predictions available.")

    return avg_probability, probabilities

def explain_prediction(input_dict):
    explanation = "Based on the transaction amount and category, along with the state information, the models have evaluated the risk of fraud."
    return explanation

def create_gauge_chart(probability):
    # Determine color based on fraud probability
    if probability < 30:
        color = "green"
    elif probability < 60:
        color = "yellow"
    else:
        color = "red"

    probability_percentage = probability

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percentage,
            title={"text": "Fraud Probability", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "red"},
                ],
            },
        )
    )

    fig.update_layout(height=300)
    return fig

def create_model_probability_chart(probabilities):
    if not probabilities:
        return go.Figure()

    models = list(probabilities.keys())
    probs = [p * 100 if p <= 1 else p for p in probabilities.values()]  # Ensure percentages

    fig = go.Figure(
        data=[
            go.Bar(
                y=models,
                x=probs,
                orientation="h",
                text=[f"{p:.2f}%" for p in probs],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Fraud Probability by Model",
        yaxis_title="Models",
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 100]),
        height=400,
    )

    return fig
