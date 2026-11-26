import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Crop Recommendation & Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# ==================================================
# LOAD DATA AND TRAIN MODELS
# ==================================================
# ==================================================
# LOAD DATA AND TRAIN MODELS — FIXED VERSION
# ==================================================
@st.cache_resource
def init_models():
    csv_file = "Crop_Yield_Data.csv"
    
    if not os.path.exists(csv_file):
        return None, "CSV_NOT_FOUND"  # ✅ Returns 2 values only
    
    df = pd.read_csv(csv_file)
    
    # Auto-find column names
    def find_col(df, keywords):
        for col in df.columns:
            if all(k in col.lower() for k in keywords):
                return col
        return None
    
    col_rainfall = find_col(df, ['rainfall'])
    col_temp = find_col(df, ['temp'])
    col_humidity = find_col(df, ['humid'])
    col_ph = find_col(df, ['ph'])
    col_n = find_col(df, ['nitrogen'])
    col_p = find_col(df, ['phosphorus'])
    col_k = find_col(df, ['potassium'])
    col_crop = find_col(df, ['crop'])
    col_irrigation = find_col(df, ['irrigation'])
    col_season = find_col(df, ['season'])
    col_yield = find_col(df, ['yield'])
    
    # Encode categorical columns
    le_crop = LabelEncoder()
    le_irrigation = LabelEncoder()
    le_season = LabelEncoder()
    
    df['crop_encoded'] = le_crop.fit_transform(df[col_crop])
    df['irrigation_encoded'] = le_irrigation.fit_transform(df[col_irrigation])
    df['season_encoded'] = le_season.fit_transform(df[col_season])
    
    FEATURES = [
        col_rainfall, col_temp, col_humidity, col_ph,
        col_n, col_p, col_k, 'irrigation_encoded', 'season_encoded'
    ]
    
    # Model 1: Crop Recommendation
    X_rec = df[FEATURES].copy()
    y_rec = df['crop_encoded'].copy()
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_rec, y_rec, test_size=0.3, random_state=42, stratify=y_rec
    )
    rec_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rec_model.fit(Xr_train, yr_train)
    
    # Model 2: Yield Prediction
    X_yield = df[FEATURES + ['crop_encoded']].copy()
    y_yield = df[col_yield].copy()
    Xy_train, Xy_test, yy_train, yy_test = train_test_split(
        X_yield, y_yield, test_size=0.3, random_state=42
    )
    yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
    yield_model.fit(Xy_train, yy_train)
    
    # Performance metrics
    rec_acc = accuracy_score(yr_test, rec_model.predict(Xr_test))
    y_pred = yield_model.predict(Xy_test)
    rmse = np.sqrt(mean_squared_error(yy_test, y_pred))
    r2 = r2_score(yy_test, y_pred)
    
    result_dict = {
        'rec_model': rec_model,
        'yield_model': yield_model,
        'le_crop': le_crop,
        'le_irrigation': le_irrigation,
        'le_season': le_season,
        'df': df,
        'cols': {
            'rainfall': col_rainfall, 'temp': col_temp, 'humidity': col_humidity,
            'ph': col_ph, 'n': col_n, 'p': col_p, 'k': col_k,
            'crop': col_crop, 'irrigation': col_irrigation,
            'season': col_season, 'yield': col_yield
        },
        'metrics': {'rec_acc': rec_acc, 'rmse': rmse, 'r2': r2},
        'features': FEATURES
    }
    
    return result_dict, None  # ✅ Returns 2 values only


# ==================================================
# MAIN APP — This part stays the same
# ==================================================
with st.spinner("Loading data and training models..."):
    data, error = init_models()

if error == "CSV_NOT_FOUND":
    st.error("❌ Error: `Crop_Yiedl_Data.csv` not found! Put it in the same folder as this script.")
    st.stop()

# Shortcuts
rec_model = data['rec_model']
yield_model = data['yield_model']
le_crop = data['le_crop']
le_irrigation = data['le_irrigation']
le_season = data['le_season']
df = data['df']
cols = data['cols']
metrics = data['metrics']


st.title("🌾 Crop Recommendation & Yield Prediction")
st.markdown("---")

# Key Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Records", len(df))
with c2:
    st.metric("Crop Types", len(df[cols['crop']].unique()))
with c3:
    st.metric("Recommendation Accuracy", f"{metrics['rec_acc']*100:.1f}%")
with c4:
    st.metric("Yield RMSE", f"{metrics['rmse']:.3f} mt/ha")

st.markdown("---")

# Prediction Functions
def get_recommendation(rf, temp, hum, ph, n, p, k, irri, season):
    ie = le_irrigation.transform([irri])[0]
    se = le_season.transform([season])[0]
    inp = np.array([[rf, temp, hum, ph, n, p, k, ie, se]])
    pred = rec_model.predict(inp)[0]
    name = le_crop.inverse_transform([pred])[0]
    conf = np.max(rec_model.predict_proba(inp)) * 100
    return name, round(conf, 1)

def get_yield(crop, rf, temp, hum, ph, n, p, k, irri, season):
    ce = le_crop.transform([crop])[0]
    ie = le_irrigation.transform([irri])[0]
    se = le_season.transform([season])[0]
    inp = np.array([[rf, temp, hum, ph, n, p, k, ie, se, ce]])
    return round(yield_model.predict(inp)[0], 4)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📋 Dataset", "📊 Model Info"])

with tab1:
    st.subheader("Enter Field Conditions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rainfall = st.number_input("Rainfall (mm)", 500.0, 3000.0, 1400.0, 10.0)
        temp = st.number_input("Temperature (°C)", 20.0, 40.0, 29.0, 0.5)
        humidity = st.number_input("Humidity (%)", 50.0, 100.0, 78.0, 1.0)
        ph = st.number_input("Soil pH", 4.0, 9.0, 6.3, 0.1)
    
    with col2:
        nitrogen = st.number_input("Nitrogen (kg/ha)", 50.0, 200.0, 110.0, 5.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", 30.0, 100.0, 52.0, 2.0)
        potassium = st.number_input("Potassium (kg/ha)", 50.0, 150.0, 80.0, 2.0)
    
    with col3:
        irrigation = st.selectbox("Irrigation", df[cols['irrigation']].unique())
        season = st.selectbox("Season", df[cols['season']].unique())
        crop_sel = st.selectbox("Crop for Yield", df[cols['crop']].unique())
    
    st.markdown("---")
    b1, b2 = st.columns(2)
    
    with b1:
        if st.button("🌾 Recommend Crop", type="primary", use_container_width=True):
            crop, conf = get_recommendation(rainfall, temp, humidity, ph, nitrogen, phosphorus, potassium, irrigation, season)
            st.success(f"### Recommended: **{crop}**")
            st.info(f"Confidence: {conf}%")
    
    with b2:
        if st.button("📈 Predict Yield", type="secondary", use_container_width=True):
            yld = get_yield(crop_sel, rainfall, temp, humidity, ph, nitrogen, phosphorus, potassium, irrigation, season)
            st.success(f"### Predicted Yield: **{yld} mt/ha**")
            avg = df[df[cols['crop']]==crop_sel][cols['yield']].mean()
            diff = yld - avg
            if diff > 0:
                st.info(f"{diff:.2f} mt/ha above average for {crop_sel}")
            elif diff < 0:
                st.warning(f"{abs(diff):.2f} mt/ha below average for {crop_sel}")
            else:
                st.info(f"Matches average for {crop_sel}")

with tab2:
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True)

with tab3:
    st.subheader("Model Performance")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Crop Recommendation**")
        st.write(f"Algorithm: Random Forest Classifier")
        st.write(f"Accuracy: **{metrics['rec_acc']*100:.1f}%**")
    with col_b:
        st.info("**Yield Prediction**")
        st.write(f"Algorithm: Random Forest Regressor")
        st.write(f"RMSE: **{metrics['rmse']:.4f}** mt/ha")
        st.write(f"R² Score: **{metrics['r2']:.4f}**")

st.markdown("---")
st.caption("🌾 Crop Recommendation & Yield Prediction System")
