# import json
# from datetime import datetime

# import pandas as pd
# import plotly.express as px
# import requests
# import streamlit as st
# from openai import AzureOpenAI
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from statsmodels.tsa.arima.model import ARIMA
# from transformers import pipeline

# API_KEY = "" #openai api key
# SLACK_WEBHOOK = "" #Slack webhook url


# def truncate_text(text, max_length=512):
#     return text[:max_length]


# def load_competitor_data():
#     """Load competitor data from a CSV file."""
#     data = pd.read_csv("competitor_data.csv")
#     print(data.head())
#     return data


# def load_reviews_data():
#     """Load reviews data from a CSV file."""
#     reviews = pd.read_csv("reviews.csv")
#     return reviews


# def analyze_sentiment(reviews):
#     """Analyze customer sentiment for reviews."""
#     sentiment_pipeline = pipeline("sentiment-analysis")
#     return sentiment_pipeline(reviews)


# def train_predictive_model(data):
#     """Train a predictive model for competitor pricing strategy."""
#     data["Discount"] = data["Discount"].str.replace("%", "").astype(float)
#     data["Price"] = data["Price"].astype(int)
#     data["Predicted_Discount"] = data["Discount"] + (data["Price"] * 0.05).round(2)

#     X = data[["Price", "Discount"]]
#     y = data["Predicted_Discount"]
#     print(X)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, train_size=0.8
#     )

#     model = RandomForestRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     return model


# import numpy as np
# import pandas as pd


# def forecast_discounts_arima(data, future_days=5):
#     """
#     Forecast future discounts using ARIMA.
#     :param data: DataFrame containing historical discount data (with a datetime index).
#     :param future_days: Number of days to forecast.
#     :return: DataFrame with historical and forecasted discounts.
#     """

#     data = data.sort_index()
#     print(product_data.index)

#     data["Discount"] = pd.to_numeric(data["Discount"], errors="coerce")
#     data = data.dropna(subset=["Discount"])

#     discount_series = data["Discount"]
#     if not isinstance(data.index, pd.DatetimeIndex):
#         try:
#             data.index = pd.to_datetime(data.index)
#         except Exception as e:
#             raise ValueError(
#                 "Index must be datetime or convertible to datetime."
#             ) from e

#     model = ARIMA(discount_series, order=(5, 1, 0))
#     model_fit = model.fit()

#     forecast = model_fit.forecast(steps=future_days)
#     future_dates = pd.date_range(
#         start=discount_series.index[-1] + pd.Timedelta(days=1), periods=future_days
#     )

#     forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Discount": forecast})
#     forecast_df.set_index("Date", inplace=True)

#     return forecast_df


# def send_to_slack(data):
#     """ """
#     payload = {"text": data}
#     response = requests.post(
#         SLACK_WEBHOOK,
#         data=json.dumps(payload),
#         headers={"Content-Type": "application/json"},
#     )


# def generate_strategy_recommendation(product_name, competitor_data, sentiment):
#     """Generate strategic recommendations using an LLM."""
#     date = datetime.now()
#     prompt = f"""
#     You are a highly skilled business strategist specializing in e-commerce. Based on the following details, suggest actionable strategies to optimize pricing, promotions, and customer satisfaction for the selected product:

# 1. **Product Name**: {product_name}

# 2. **Competitor Data** (including current prices, discounts, and predicted discounts):
# {competitor_data}

# 3. **Sentiment Analysis**:
# {sentiment}


# 5. **Today's Date**: {str(date)}

# ### Task:
# - Analyze the competitor data and identify key pricing trends.
# - Leverage sentiment analysis insights to highlight areas where customer satisfaction can be improved.
# - Use the discount predictions to suggest how pricing strategies can be optimized over the next 5 days.
# - Recommend promotional campaigns or marketing strategies that align with customer sentiments and competitive trends.
# - Ensure the strategies are actionable, realistic, and geared toward increasing customer satisfaction, driving sales, and outperforming competitors.

# Provide your recommendations in a structured format:
# 1. **Pricing Strategy**
# 2. **Promotional Campaign Ideas**
# 3. **Customer Satisfaction Recommendations**
#     """

#     messages = [{"role": "user", "content": prompt}]

#     data = {
#         "messages": [{"role": "user", "content": prompt}],
#         "model": "llama3-8b-8192",
#         "temperature": 0,
#     }

#     headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

#     res = requests.post(
#         "https://api.groq.com/openai/v1/chat/completions",
#         data=json.dumps(data),
#         headers=headers,
#     )
#     res = res.json()
#     response = res["choices"][0]["message"]["content"]
#     return response


# ####--------------------------------------------------##########

# st.set_page_config(page_title="E-Commerce Competitor Strategy Dashboard", layout="wide")


# st.title("E-Commerce Competitor Strategy Dashboard")
# st.sidebar.header("Select a Product")

# products = [
#     "Apple iPhone 15",
#     "Apple 2023 MacBook Pro (16-inch, Apple M3 Pro chip with 12‑core CPU and 18‑core GPU, 36GB Unified Memory, 512GB) - Silver",
#     "OnePlus Nord 4 5G (Mercurial Silver, 8GB RAM, 256GB Storage)",
#     "Sony WH-1000XM5 Best Active Noise Cancelling Wireless Bluetooth Over Ear Headphones with Mic for Clear Calling, up to 40 Hours Battery -Black",
# ]
# selected_product = st.sidebar.selectbox("Choose a product to analyze:", products)


# competitor_data = load_competitor_data()
# reviews_data = load_reviews_data()

# product_data = competitor_data[competitor_data["product_name"] == selected_product]
# product_reviews = reviews_data[reviews_data["product_name"] == selected_product]

# st.header(f"Competitor Analysis for {selected_product}")
# st.subheader("Competitor Data")
# st.table(product_data.tail(5))

# if not product_reviews.empty:
#     product_reviews["reviews"] = product_reviews["reviews"].apply(
#         lambda x: truncate_text(x, 512)
#     )
#     reviews = product_reviews["reviews"].tolist()
#     sentiments = analyze_sentiment(reviews)

#     st.subheader("Customer Sentiment Analysis")
#     sentiment_df = pd.DataFrame(sentiments)
#     fig = px.bar(sentiment_df, x="label", title="Sentiment Analysis Results")
#     st.plotly_chart(fig)
# else:
#     st.write("No reviews available for this product.")


# # Preprocessing

# product_data["Date"] = pd.to_datetime(product_data["Date"], errors="coerce")
# product_data = product_data.dropna(subset=["Date"])
# product_data.set_index("Date", inplace=True)
# product_data = product_data.sort_index()

# product_data["Discount"] = pd.to_numeric(product_data["Discount"], errors="coerce")
# product_data = product_data.dropna(subset=["Discount"])

# # Forecasting Model
# product_data_with_predictions = forecast_discounts_arima(product_data)


# st.subheader("Competitor Current and Predicted Discounts")
# st.table(product_data_with_predictions.tail(10))

# recommendations = generate_strategy_recommendation(
#     selected_product,
#     product_data_with_predictions,
#     sentiments if not product_reviews.empty else "No reviews available",
# )
# st.subheader("Strategic Recommendations")
# st.write(recommendations)
# send_to_slack(recommendations)
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
import streamlit as st
import json
import requests

# Load and preprocess data
def load_competitor_data():
    """Load competitor data from CSV file."""
    data = pd.read_csv("competitor_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = pd.to_numeric(data['Price'])
    data['Discount'] = pd.to_numeric(data['Discount'])
    return data

def load_reviews_data():
    """Load reviews data from CSV file."""
    reviews = pd.read_csv("reviews.csv")
    reviews['date'] = pd.to_datetime(reviews['date'])
    reviews['rating'] = pd.to_numeric(reviews['rating'])
    return reviews

def analyze_sentiment(reviews):
    """Analyze customer sentiment from reviews."""
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = sentiment_pipeline(reviews)
    return pd.DataFrame(sentiments)

def train_predictive_model(data):
    """Train a Random Forest model for price prediction."""
    X = data[['Price', 'Discount']]
    y = data['Price'].shift(-1)  # Predict next day's price
    y = y.fillna(method='ffill')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def forecast_discounts_arima(data, future_days=5):
    """Forecast future discounts using ARIMA model."""
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('Date')
    
    model = ARIMA(data['Discount'], order=(5,1,0))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1),
        periods=future_days
    )
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Discount': forecast
    })
    forecast_df.set_index('Date', inplace=True)
    
    return forecast_df

def generate_insights(data, reviews_data, forecast_data):
    """Generate market insights from the data."""
    insights = {
        'price_trends': {
            'avg_price': data['Price'].mean(),
            'price_volatility': data['Price'].std(),
            'max_discount': data['Discount'].max()
        },
        'customer_sentiment': {
            'avg_rating': reviews_data['rating'].mean() if 'rating' in reviews_data.columns else None,
            'review_count': len(reviews_data)
        },
        'forecast': {
            'predicted_discounts': forecast_data['Predicted_Discount'].tolist(),
            'forecast_dates': forecast_data.index.strftime('%Y-%m-%d').tolist()
        }
    }
    return insights

# Streamlit UI
st.set_page_config(page_title="E-Commerce Analytics Dashboard", layout="wide")
st.title("E-Commerce Analytics Dashboard")

# Load data
competitor_data = load_competitor_data()
reviews_data = load_reviews_data()

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    [competitor_data['Date'].min(), competitor_data['Date'].max()]
)

# Filter data based on selection
filtered_data = competitor_data[
    (competitor_data['Date'] >= pd.Timestamp(date_range[0])) &
    (competitor_data['Date'] <= pd.Timestamp(date_range[1]))
]

# Price Analysis
st.header("Price Analysis")
fig_price = px.line(filtered_data, x='Date', y='Price', title='Price Trends')
st.plotly_chart(fig_price)

# Discount Analysis
st.header("Discount Analysis")
fig_discount = px.bar(filtered_data, x='Date', y='Discount', title='Discount Distribution')
st.plotly_chart(fig_discount)

# Sentiment Analysis
if not reviews_data.empty:
    st.header("Customer Sentiment Analysis")
    reviews = reviews_data['reviews'].tolist()
    sentiments = analyze_sentiment(reviews)
    fig_sentiment = px.pie(sentiments, names='label', title='Sentiment Distribution')
    st.plotly_chart(fig_sentiment)

# Price Prediction
st.header("Price Predictions")
model = train_predictive_model(filtered_data)
forecast = forecast_discounts_arima(filtered_data)
st.write("Predicted Discounts for Next 5 Days:")
st.dataframe(forecast)

# Generate and display insights
insights = generate_insights(filtered_data, reviews_data, forecast)
st.header("Market Insights")
st.json(insights)