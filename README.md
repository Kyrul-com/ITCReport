# 🇲🇾 ITCReport: Malaysia Tourism Analytics & AI Forecast

ITCReport is a Python-based data engineering and visualization dashboard that tracks tourist arrivals to Malaysia. It connects directly to the official DOSM Open Data storage to fetch real-time immigration statistics.

The application features an AI Forecasting Engine to predict tourist numbers for the remainder of 2026 and a Demographic Proxy Model to estimate Muslim-Friendly Tourism (MFTH) market share.

## 🚀 Key Features

*   **Live Data Pipeline**: Fetches `.parquet` data directly from the Malaysian Government's open data storage (MyIMMs data).
*   **🤖 AI Forecasting (2026)**: Uses Holt-Winters Exponential Smoothing (Machine Learning) to analyze seasonality from 2019-2025 and predict monthly arrivals for the rest of 2026.
*   **☪️ Muslim-Friendly Tourism (MFTH) Estimator**: A logic layer that calculates estimated Muslim tourist arrivals based on country-of-origin demographic proxies.
*   **Interactive Visualization**:
    *   Switch between "Monthly Speed" and "Cumulative Totals" (Target: 30M).
    *   Drill-down analytics for Nationality and Gender.
    *   Time-series forecasting with trend analysis.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vaxei/ITCReport.git
    cd ITCReport
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Windows
    # source .venv/bin/activate # macOS/Linux
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

Run the dashboard locally using Streamlit:

```bash
python -m streamlit run app.py
```

## 📊 Methodology

### 1. The AI Forecast Model
We use the **Holt-Winters Exponential Smoothing** algorithm to predict future arrivals. It analyzes the **Trend** (general growth) and **Seasonality** (12-month cycles) using historical data from 2019 to the present to learn seasonal patterns like holiday spikes and off-peak dips.

### 2. Muslim-Friendly Tourism (MFTH) Proxy
Since official immigration data does not record the religion of tourists, this dashboard estimates MFTH numbers using a **Country-of-Origin Proxy Model**:

> **Formula**: `Arrivals * Muslim Population % of Source Country`

**Data Sources**: Pew Research Center & CIA World Factbook.

## 📁 Project Structure

*   `app.py`: Main application entry point.
*   `requirements.txt`: Python dependencies.
*   `README.md`: Documentation.

## 🔗 Data Sources

*   **Raw Data**: Department of Statistics Malaysia (DOSM)
*   **Dataset**: `arrivals_soe.parquet` (Immigration Department of Malaysia)

---
Created by vaxei.py
