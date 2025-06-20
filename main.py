from flask import Flask, render_template_string, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import os, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go
from datetime import datetime

app = Flask(__name__)
model_path = "lstm_model.h5"
metrics_path = "metrics.json"

# ---------- HTML Template ----------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Stock Prediction with Candlestick Chart</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { padding: 20px; background-color: #f9f9f9; }
        .price-box, .accuracy-box { font-size: 1.2rem; margin-top: 10px; }
        #candlestick-chart { width: 100%; height: 500px; }
    </style>
</head>
<body>
<div class="container">
    <h2 class="text-center mb-4">📈 Live Stock Prediction with Candlestick Chart</h2>
    <form id="stockForm">
        <div class="mb-3">
            <input type="text" class="form-control" id="ticker" placeholder="Enter Stock Ticker (e.g. AAPL)" required>
        </div>
        <div class="mb-3">
            <input type="date" class="form-control" id="targetDate" required>
        </div>
        <button type="submit" class="btn btn-primary">Predict</button>
    </form>
    <hr>
    <div id="result" class="mt-3"></div>
    <div id="accuracy" class="accuracy-box mt-2"></div>
    <div class="mt-4" id="chart">
        <div id="candlestick-chart"></div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
<script>
    document.getElementById("stockForm").addEventListener("submit", function(e) {
        e.preventDefault();
        const ticker = document.getElementById("ticker").value;
        const targetDate = document.getElementById("targetDate").value;

        axios.post('/predict', new URLSearchParams({ ticker, targetDate }))
            .then(response => {
                document.getElementById("result").innerHTML = `
                    <div class="price-box">
                        <strong>📍 Current Price:</strong> $${response.data.current_price}<br>
                        <strong>🤖 Predicted Price on ${targetDate}:</strong> $${response.data.predicted_price}
                    </div>`;
                document.getElementById("accuracy").innerHTML = `
                    <div class="accuracy-box">
                        <strong>📊 Model Accuracy:</strong><br>
                        MAE: ${response.data.accuracy.mae}<br>
                        RMSE: ${response.data.accuracy.rmse}
                    </div>`;
            });

        axios.post('/chart', new URLSearchParams({ ticker }))
            .then(response => {
                const fig = response.data.chart;
                Plotly.react('candlestick-chart', fig.data, fig.layout);
            });
    });
</script>
</body>
</html>
"""

# ---------- Train & Save Model ----------
def train_and_save_model():
    print("Training model...")
    df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    model.save(model_path)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    metrics = {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4)
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    print("Model trained and accuracy saved.")

# ---------- Load Model and Accuracy ----------
def load_model_and_metrics():
    if not os.path.exists(model_path):
        train_and_save_model()
    model = load_model(model_path)

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {"mae": "N/A", "mse": "N/A", "rmse": "N/A"}
    return model, metrics

model, metrics = load_model_and_metrics()
scaler = MinMaxScaler()

# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    target_date_str = request.form['targetDate']
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()

    if target_date <= today:
        return jsonify({'current_price': '-', 'predicted_price': '❌ Choose a future date', 'accuracy': metrics})

    data = yf.download(ticker, period="90d", interval="1d")[['Close']]
    if len(data) < 60:
        return jsonify({'current_price': 'N/A', 'predicted_price': '❌ Not enough data', 'accuracy': metrics})

    current_price = float(data['Close'].iloc[-1])
    scaled_data = scaler.fit_transform(data)
    last_seq = scaled_data[-60:].tolist()

    n_days = (target_date - today).days
    for _ in range(n_days):
        input_seq = np.reshape(last_seq[-60:], (1, 60, 1))
        pred = model.predict(input_seq, verbose=0)
        last_seq.append(pred[0].tolist())

    predicted_price = scaler.inverse_transform([last_seq[-1]])[0][0]
    return jsonify({
        'current_price': round(current_price, 2),
        'predicted_price': round(float(predicted_price), 2),
        'accuracy': metrics
    })

@app.route('/chart', methods=['POST'])
def chart():
    ticker = request.form['ticker'].upper()
    df = yf.download(ticker, period="60d", interval="1d")

    if df.empty:
        return jsonify({'chart': {'data': [], 'layout': {'title': 'No data'}}})

    df.reset_index(inplace=True)

    candle = go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing=dict(line=dict(color='green')),
        decreasing=dict(line=dict(color='red'))
    )

    layout = go.Layout(
        title=f'{ticker} - Candlestick Chart (60 Days)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig = go.Figure(data=[candle], layout=layout)
    return jsonify({'chart': fig.to_plotly_json()})

# ---------- Main ----------
if __name__ == '__main__':
    app.run(debug=True)
