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
def generate_forecast(df, value_column='arrivals'):
    """
    Generic AI function. You can pass it 'arrivals' OR 'Muslim_Est'
    """
    # Group by month using the specific column we want to predict
    monthly_data = df.groupby("date")[value_column].sum().asfreq('MS')
    
    # Train Model (Holt-Winters)
    model = ExponentialSmoothing(
        monthly_data, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    ).fit()
    
    # Predict until Dec 2026
    last_date = monthly_data.index[-1]
    
    # Hardcode timezone to avoid "works on my machine" issues
    target_date = pd.Timestamp("2026-12-01")
    months_needed = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
    
    if months_needed <= 0:
        return monthly_data.reset_index(), pd.DataFrame()
        
    forecast_series = model.forecast(months_needed)
    
    forecast_df = pd.DataFrame({
        'date': forecast_series.index,
        value_column: forecast_series.values,
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

    # --- GLOBAL AI CALCULATION (Runs for ALL pages) ---
    with st.spinner("Running AI Models (Total & Muslim Markets)..."):
        
        # 1. Forecast TOTAL Arrivals
        hist_total, pred_total = generate_forecast(raw_df, value_column='arrivals')
        
        # 2. Forecast MUSLIM Arrivals (The New Part)
        # We pass the raw_df but tell the AI to look at 'Muslim_Est' column
        raw_df_muslim = raw_df.copy()
        raw_df_muslim['Muslim_Ratio'] = raw_df_muslim['country'].map(ISO_MAP).map(RELIGION_PROXY).fillna(0.05)
        raw_df_muslim['Muslim_Est'] = (raw_df_muslim['arrivals'] * raw_df_muslim['Muslim_Ratio']).astype(int)
        
        hist_muslim, pred_muslim = generate_forecast(raw_df_muslim, value_column='Muslim_Est')

        # Combine Total Data
        combined_trend = pd.concat([hist_total, pred_total])
        combined_trend = combined_trend[combined_trend['date'] >= '2023-01-01'].copy()
        
        # Combine Muslim Data
        combined_muslim_trend = pd.concat([hist_muslim, pred_muslim])
        combined_muslim_trend = combined_muslim_trend[combined_muslim_trend['date'] >= '2022-01-01'].copy()

        # Calculate 2026 Total for Top Cards
        act_2026 = hist_total[hist_total['date'].dt.year == 2026]['arrivals'].sum()
        fc_2026 = pred_total[pred_total['date'].dt.year == 2026]['arrivals'].sum()
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

        st.divider()

        view_mode = st.radio("Graph View:", ["Monthly Rate (Speed)", "Cumulative Total (Sum)"], horizontal=True)

        if view_mode == "Cumulative Total (Sum)":
            data_2026 = combined_trend[combined_trend['date'].dt.year == 2026].copy()
            data_2026['Cumulative_Arrivals'] = data_2026['arrivals'].cumsum()
            fig = px.line(data_2026, x="date", y="Cumulative_Arrivals", color="Type", markers=True)
            fig.update_layout(title="Cumulative Arrivals 2026", yaxis_title="Total Visitors")
        else:
            fig = px.line(combined_trend, x="date", y="arrivals", color="Type", markers=True)
            fig.update_layout(title="Monthly Visitor Rate", yaxis_title="Visitors per Month")

        st.plotly_chart(fig, use_container_width=True)

    elif page == "📊 Demographics":
        st.title("Nationality & Gender Breakdown")
        latest_year = main_df['year'].max()
        st.subheader(f"Data for {latest_year}")
        top10 = main_df[main_df['year'] == latest_year].groupby('Country_Name')['arrivals'].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top10, x="arrivals", y="Country_Name", orientation='h')
        st.plotly_chart(fig_bar, use_container_width=True)

    elif page == "☪️ Muslim Tourism":
        st.title("Muslim Friendly Tourism (MFTH) Forecast")
        st.markdown("This chart uses AI to predict **Muslim-specific arrivals** based on historical seasonality from OIC countries.")

        # Calculate Totals for Metric Cards
        muslim_2026_actual = hist_muslim[hist_muslim['date'].dt.year == 2026]['Muslim_Est'].sum()
        muslim_2026_forecast = pred_muslim[pred_muslim['date'].dt.year == 2026]['Muslim_Est'].sum()
        muslim_total_proj = muslim_2026_actual + muslim_2026_forecast

        col1, col2 = st.columns(2)
        col1.metric("Projected Muslim Arrivals (2026)", f"{muslim_total_proj/1e6:.2f} M")
        col2.metric("Market Share", f"{(muslim_total_proj/total_2026)*100:.1f}%", "of Total Tourists")

        # Plot the Combined Data (Actual + AI)
        # We aggregate by Year for a clean Bar Chart
        yearly_muslim = combined_muslim_trend.groupby([combined_muslim_trend['date'].dt.year, 'Type'])['Muslim_Est'].sum().reset_index()
        yearly_muslim.columns = ['Year', 'Data Source', 'Visitors']
        
        fig_mfth = px.bar(yearly_muslim, x="Year", y="Visitors", color="Data Source", 
                          title="Muslim Arrivals: History vs AI Prediction",
                          text_auto='.2s',
                          color_discrete_map={"Actual": "#00CC96", "AI Forecast": "#FFA500"})
        
        st.plotly_chart(fig_mfth, use_container_width=True)

        with st.expander("View Monthly AI Prediction Data"):
            st.dataframe(pred_muslim)

else:
    st.error("Could not load data.")
