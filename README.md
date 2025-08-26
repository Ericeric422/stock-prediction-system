# Stock Market Prediction System

**Author**: AI Assistant
**License**: Educational Use Only

---

### ⚠️ IMPORTANT DISCLAIMER

This system is for **educational and research purposes only**. Stock market prediction is extremely challenging, volatile, and risky. This project is a demonstration of a data science workflow, not a financial advising tool.

- **Past performance does not guarantee future results.**
- **Always use paper trading before risking real money.**
- **Never invest more than you can afford to lose.**

---

## 1. Overview

This project is a complete, end-to-end system for predicting stock price movements using machine learning and deep learning. It fetches historical stock data, engineers a wide array of technical features, trains multiple models (including RandomForest, XGBoost, SVM, and LSTM), and evaluates them using a robust backtesting framework.

Finally, it saves the best-performing models and serves predictions through a simple REST API built with Flask.

## 2. Features

- **Data Collection**: Fetches historical stock data from Yahoo Finance (`yfinance`).
- **Feature Engineering**: Generates over 50 technical indicators (RSI, MACD, Bollinger Bands, moving averages, volatility, momentum, etc.).
- **Multiple Model Training**:
  - **Traditional ML**: `RandomForest`, `XGBoost`, `LogisticRegression`, `SVM`.
  - **Deep Learning**: `LSTM` model for sequence prediction (if TensorFlow is installed).
  - **Ensemble Model**: A `VotingClassifier` that combines predictions for improved robustness.
- **Time-Series Aware Evaluation**: Uses `TimeSeriesSplit` for cross-validation and a time-based train/test split to prevent data leakage.
- **Financial Backtesting**: Evaluates models not just on accuracy, but on financial metrics like **Sharpe Ratio** and **Max Drawdown**.
- **Production-Ready API**: A Flask API to serve predictions for any given stock symbol.
- **Model Persistence**: Saves trained models, scalers, and feature lists for immediate use without retraining.
- **Monitoring**: A basic system to track prediction performance over time.
- **Visualization**: Automatically generates plots for model comparison and feature importance.

## 3. Project Structure

```
market/
├── models/                  # Saved models, scalers, and feature lists
├── data_cache/              # (Optional) Caches downloaded data
├── stock_prediction_system.py # The main script containing all logic
├── requirements.txt         # Project dependencies
├── stock_prediction.log     # Log file for debugging
├── model_comparison.png     # Output plot comparing model performance
├── feature_importance.png   # Output plot of top features
└── README.md                # This file
```

## 4. Setup and Installation

**Prerequisites**:
- Python 3.8+
- `pip` package manager

**Installation Steps**:

1.  **Clone the repository or download the files.**

2.  **Navigate to the project directory:**
    ```bash
    cd path/to/market
    ```

3.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 5. How to Run the System

Execute the main script from your terminal:

```bash
# For training models and starting the API:
python stock_prediction_system.py --start-api

# For training models only (without starting the API):
python stock_prediction_system.py
```

The script will perform all steps automatically:
1.  Collect data for the stocks defined in the `SYMBOLS` list.
2.  Engineer features.
3.  Train and evaluate all models.
4.  Print a summary of the results to the console.
5.  Save the trained models to the `models/` directory.
6.  Generate `model_comparison.png` and `feature_importance.png`.

At the end, it will ask if you want to start the API server.

## 6. Using the API

If you choose to run the API server, it will be available at `http://localhost:5000`.

### Endpoints

- **Get a Prediction**
  - `GET /predict/<SYMBOL>`
  - **Example**: `http://localhost:5000/predict/AAPL`
  - **Success Response**:
    ```json
    {
      "symbol": "AAPL",
      "prediction": 1,
      "probability": 0.62,
      "direction": "UP",
      "confidence": 0.24,
      "timestamp": "2023-10-27T10:30:00Z",
      "model_used": "Ensemble"
    }
    ```

- **Health Check**
  - `GET /health`
  - Returns the status of the API and loaded models.

- **Performance Summary**
  - `GET /performance`
  - Returns a summary from the monitoring system.
