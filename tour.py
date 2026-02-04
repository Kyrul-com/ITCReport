import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ==========================================
# 1. CONFIGURATION & MAPPINGS
# ==========================================
st.set_page_config(page_title="MY Tourism AI Forecast 2026", layout="wide")

DATA_URL = "https://storage.data.gov.my/demography/arrivals_soe.parquet"

ISO_MAP = {
    "SGP": "Singapore", "IDN": "Indonesia", "CHN": "China", "THA": "Thailand",
    "BRN": "Brunei", "IND": "India", "KOR": "South Korea", "VNM": "Vietnam",
    "AUS": "Australia", "PHL": "Philippines", "GBR": "United Kingdom",
    "JPN": "Japan", "USA": "United States", "TWN": "Taiwan", "DEU": "Germany",
    "SAU": "Saudi Arabia", "NLD": "Netherlands", "FRA": "France", "RUS": "Russia"
}

RELIGION_PROXY = {
    "Indonesia": 0.87, "Brunei": 0.81, "Saudi Arabia": 1.0, "Turkey": 0.99,
    "Bangladesh": 0.90, "Pakistan": 0.96, "Egypt": 0.90, "Iran": 0.99,
    "India": 0.14, "Singapore": 0.15, "China": 0.02, "Thailand": 0.05,
    "United Kingdom": 0.06, "Australia": 0.03, "Philippines": 0.06
}

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def load_gov_data():
    try:
        df = pd.read_parquet(DATA_URL)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df = df[df['year'] >= 2019] 
        df['Country_Name'] = df['country'].map(ISO_MAP).fillna(df['country'])
        return df
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. AI FORECASTING ENGINE
# ==========================================
def generate_forecast(df):
    monthly_data = df.groupby("date")['arrivals'].sum().asfreq('MS')
    
    # Train Model
    model = ExponentialSmoothing(
        monthly_data, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    ).fit()
    
    # Predict until Dec 2026
    last_date = monthly_data.index[-1]
    target_date = pd.Timestamp("2026-12-01")
    months_needed = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    
    if months_needed <= 0:
        return monthly_data.reset_index(), pd.DataFrame()
        
    forecast_series = model.forecast(months_needed)
    
    forecast_df = pd.DataFrame({
        'date': forecast_series.index,
        'arrivals': forecast_series.values,
        'Type': 'AI Forecast'
    })
    
    history_df = monthly_data.reset_index()
    history_df['Type'] = 'Actual'
    
    return history_df, forecast_df

# ==========================================
# 4. MAIN APP LOGIC
# ==========================================
raw_df = load_gov_data()

if not raw_df.empty:
    main_df = raw_df[raw_df['year'] >= 2022].copy()
    
    # Apply Religion Logic
    main_df['Muslim_Ratio'] = main_df['Country_Name'].map(RELIGION_PROXY).fillna(0.05)
    main_df['Muslim_Est'] = (main_df['arrivals'] * main_df['Muslim_Ratio']).astype(int)

    # --- GLOBAL CALCULATION (Runs for ALL pages) ---
    # We calculate the forecast here so the variables exist everywhere
    with st.spinner("Analyzing AI trends..."):
        hist_monthly, pred_monthly = generate_forecast(raw_df)
        combined_trend = pd.concat([hist_monthly, pred_monthly])
        combined_trend = combined_trend[combined_trend['date'] >= '2023-01-01']

        # Calculate the 2026 Total NOW so it is saved in memory
        act_2026 = hist_monthly[hist_monthly['date'].dt.year == 2026]['arrivals'].sum()
        fc_2026 = pred_monthly[pred_monthly['date'].dt.year == 2026]['arrivals'].sum()
        total_2026 = act_2026 + fc_2026

    # --- UI NAVIGATION ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🤖 AI Forecast 2026", "📊 Demographics", "☪️ Muslim Tourism"])

    if page == "🤖 AI Forecast 2026":
        st.title("🇲🇾 Tourism Forecast (2026)")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Confirmed Arrivals (YTD)", f"{act_2026/1e6:.2f} M")
        col2.metric("Predicted Rest of Year", f"{fc_2026/1e6:.2f} M")
        col3.metric("TOTAL 2026 Projection", f"{total_2026/1e6:.2f} M")

        fig = px.line(combined_trend, x="date", y="arrivals", color="Type", 
                      markers=True, title="Projected Arrival Trend",
                      color_discrete_map={"Actual": "#36a2eb", "AI Forecast": "#ff6384"})
        st.plotly_chart(fig, use_container_width=True)

    elif page == "📊 Demographics":
        st.title("Nationality & Gender Breakdown")
        latest_year = main_df['year'].max()
        st.subheader(f"Data for {latest_year}")
        
        top10 = main_df[main_df['year'] == latest_year].groupby('Country_Name')['arrivals'].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top10, x="arrivals", y="Country_Name", orientation='h')
        st.plotly_chart(fig_bar, use_container_width=True)

    elif page == "☪️ Muslim Tourism":
        st.title("Muslim Friendly Tourism (MFTH) Estimates")
        
        # Now this works because total_2026 was calculated at the top!
        avg_muslim_ratio = main_df['Muslim_Est'].sum() / main_df['arrivals'].sum()
        forecasted_muslim_2026 = total_2026 * avg_muslim_ratio
        
        st.metric("Projected Muslim Arrivals 2026", f"{forecasted_muslim_2026/1e6:.2f} M")
        
        mfth_df = main_df.groupby('year')[['Muslim_Est']].sum().reset_index()
        fig_mfth = px.bar(mfth_df, x="year", y="Muslim_Est", title="Muslim Arrivals Growth")
        st.plotly_chart(fig_mfth, use_container_width=True)

else:
    st.error("Could not load data.")