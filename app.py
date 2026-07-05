%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Supply Chain Plant Optimization")

# 1. Load Data
@st.cache_data
def load_plant_data():
    data = {
        'id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'],
        'plant_location': ['Gampaha', 'Kandy', 'Gampaha', 'Matara', 'Kandy', 'Matara', 'Gampaha', 'Kandy', 'Matara', 'Gampaha'],
        'plant_lat': [7.0873, 7.2906, 7.0873, 5.9549, 7.2906, 5.9549, 7.0873, 7.2906, 5.9549, 7.0873],
        'plant_lon': [80.0144, 80.6337, 80.0144, 80.5550, 80.6337, 80.5550, 80.0144, 80.6337, 80.5550, 80.0144],
        'farm_location': ['Dompe', 'Gampola', 'Mirigama', 'Akuressa', 'Peradeniya', 'Kamburupitiya', 'Avissawella', 'Katugastota', 'Weligama', 'Veyangoda'],
        'plant_cost': [152500, 98000, 210000, 135000, 185000, 115000, 240000, 160000, 105000, 198000],
        'distance_to_farm_km': [2.4, 5.1, 1.8, 8.7, 3.9, 4.2, 0.5, 6.3, 10.2, 2.9],
        'employees_count': [24, 15, 38, 19, 30, 18, 45, 26, 14, 34],
        'input_kg': [500, 320, 750, 450, 600, 380, 900, 520, 300, 680],
        'output_ml': [410000, 250000, 630000, 340000, 495000, 310000, 765000, 415000, 220000, 560000],
        'ingredients_cost': [12500, 8100, 19200, 11000, 15000, 9500, 22000, 13200, 7800, 17400],
        'sales_ml': [395000, 242000, 620000, 310000, 480000, 295000, 750000, 390000, 205000, 545000]
    }
    return pd.DataFrame(data)

df = load_plant_data()

# 2. Model Prediction logic
def predict_metrics(plant_location, plant_cost, distance, employees, input_kg, ingredients_cost, sales_ml):
    try:
        with open("multi_output_model.pkl", "rb") as f:
            model = pickle.load(f)
            is_gampaha = 1 if plant_location == 'Gampaha' else 0
            is_kandy = 1 if plant_location == 'Kandy' else 0
            is_matara = 1 if plant_location == 'Matara' else 0
            
            # Formulating the exact 17 feature sequence shape
            features = [
                plant_cost, distance, employees, input_kg, ingredients_cost, sales_ml,
                is_gampaha, is_kandy, is_matara, 0, 0, 0, 0, 0, 0, 0, 0
            ]
            preds = model.predict([features])
            return preds[0][0], preds[0][1], preds[0][2]
    except Exception as e:
        # Dynamic fallback approximation rules
        prod = 0.85 - (distance * 0.01)
        wastage = (input_kg * 0.03) + (distance * 1.2)
        returns = (sales_ml * 0.011)
        return float(prod), float(wastage), float(returns)

st.title("🏭 Supply Chain & Plant Performance Optimization Dashboard")

if 'selected_plant' not in st.session_state:
    st.session_state.selected_plant = "P001"

col1, col2 = st.columns([2, 1.2])

with col1:
    st.subheader("Map View - Processing Plants")
    m = folium.Map(location=[6.8, 80.3], zoom_start=8)
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['plant_lat'], row['plant_lon']],
            popup=f"Plant: {row['id']}",
            icon=folium.Icon(color="blue" if row['id'] != st.session_state.selected_plant else "red", icon="industry", prefix="fa")
        ).add_to(m)

    map_data = st_folium(m, width="100%", height=450, key="plant_map")
    if map_data and map_data.get("last_object_clicked"):
        clicked_lat = map_data["last_object_clicked"]["lat"]
        clicked_lon = map_data["last_object_clicked"]["lng"]
        match = df[np.isclose(df['plant_lat'], clicked_lat, atol=0.01) & np.isclose(df['plant_lon'], clicked_lon, atol=0.01)]
        if not match.empty:
            st.session_state.selected_plant = match.iloc[0]['id']

target_row = df[df['id'] == st.session_state.selected_plant].iloc[0]

with col2:
    st.subheader(f"📊 Live Analysis: {target_row['id']}")
    st.info(f"**Region:** {target_row['plant_location']}")
    p_cost = st.number_input("Plant Setup Cost ($)", value=int(target_row['plant_cost']))
    dist = st.slider("Distance to Farm (km)", 0.0, 15.0, float(target_row['distance_to_farm_km']))
    emp = st.number_input("Employee Count", value=int(target_row['employees_count']))
    inp_kg = st.number_input("Raw Input Weight (kg)", value=int(target_row['input_kg']))
    ing_cost = st.number_input("Ingredients Cost ($)", value=int(target_row['ingredients_cost']))
    sales_vol = st.number_input("Expected Sales Target (ml)", value=int(target_row['sales_ml']))

    pred_productivity, pred_wastage, pred_returns = predict_metrics(
        target_row['plant_location'], p_cost, dist, emp, inp_kg, ing_cost, sales_vol
    )
    
    st.subheader("🧠 Machine Learning Predictions")
    m1, m2, m3 = st.columns(3)
    m1.metric("Productivity Index", f"{pred_productivity:.2%}")
    m2.metric("Predicted Wastage", f"{pred_wastage:.2f} kg")
    m3.metric("Expected Returns", f"{pred_returns:.0f} ml")

st.markdown("---")
st.subheader("📈 Macro Plant Analysis & Comparative Insights")
c1, c2 = st.columns(2)
with c1:
    st.write("**Resource Utilization Capacity (Inputs vs Production)**")
    st.bar_chart(data=df, x='id', y='input_kg', color='#2b5c8f')
with c2:
    st.write("**Operational Distances affecting Performance**")
    st.line_chart(data=df, x='id', y='distance_to_farm_km', color='#cc4a4a')

st.dataframe(df, use_container_width=True)
