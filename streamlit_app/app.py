import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os 

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices using machine learning based on various features of California housing data.")

# Sidebar for inputs
st.sidebar.header("House Features")

# Input fields
med_inc = st.sidebar.slider("Median Income (in 10k$)", 0.5, 15.0, 5.0, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1.0, 50.0, 10.0, 1.0)
ave_rooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 6.0, 0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 3.0, 1.2, 0.1)
population = st.sidebar.slider("Population", 100.0, 10000.0, 3000.0, 100.0)
ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, 0.1)
longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -118.0, 0.1)

# API endpoint - READ FROM ENVIRONMENT VARIABLE
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Show which API URL is being used (for debugging)
st.sidebar.markdown(f"**API URL:** {API_URL}")

# Prediction button
if st.sidebar.button("Predict Price", type="primary"):
    # Prepare data
    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    try:
        # Make API call
        response = requests.post(f"{API_URL}/predict", json=features)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result["predicted_price"]
            
            # Display result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Price",
                    value=f"${predicted_price:,.2f}",
                    delta=None
                )
            
            with col2:
                price_per_room = predicted_price / ave_rooms
                st.metric(
                    label="Price per Room",
                    value=f"${price_per_room:,.2f}",
                    delta=None
                )
            
            with col3:
                affordability = "High" if predicted_price < 200000 else "Medium" if predicted_price < 400000 else "Low"
                st.metric(
                    label="Affordability",
                    value=affordability,
                    delta=None
                )
            
            # Feature summary visualization - FIXED VERSION
            st.subheader("Input Features Summary")
            feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
            
            # Create bar chart with proper method
            fig = px.bar(feature_df, x='Feature', y='Value', title="Input Feature Values")
            fig.update_layout(xaxis_tickangle=45)  # Correct method
            st.plotly_chart(fig, use_container_width=True)
            
            # Location visualization
            st.subheader("Property Location")
            map_data = pd.DataFrame({
                'lat': [latitude],
                'lon': [longitude]
            })
            st.map(map_data, zoom=6)
            
        else:
            st.error(f"API Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API server is running on http://localhost:8000")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app predicts California house prices using a Random Forest model trained on the California Housing dataset.")

# Main content when no prediction is made
if "predicted_price" not in locals():
    st.markdown("### How to use this app:")
    st.markdown("1. Adjust the feature values in the sidebar")
    st.markdown("2. Click 'Predict Price' to get a prediction")
    st.markdown("3. View the results and visualizations")
    
    # Sample data visualization
    st.subheader("California Housing Dataset Overview")
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Latitude': np.random.uniform(32, 42, 1000),
        'Longitude': np.random.uniform(-125, -114, 1000),
        'Price': np.random.uniform(50000, 800000, 1000)
    })
    
    fig = px.scatter(sample_data, x='Longitude', y='Latitude', color='Price',
                    title="Sample California Housing Prices by Location",
                    color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)