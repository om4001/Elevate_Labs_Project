# Elevate_Labs_Project

````markdown
# 📈 Live Stock Price Prediction Web App

This project is a Flask-based web application that allows users to **predict future stock prices** using an **LSTM (Long Short-Term Memory)** neural network and visualize recent price movements via an interactive **candlestick chart**. 

The model is trained using historical data from Yahoo Finance, and the app includes metrics such as **MAE** and **RMSE** to indicate model accuracy.

---

## 🚀 Features

- ✅ Predict stock prices for any ticker symbol (e.g., `AAPL`, `GOOG`, `MSFT`)
- 📆 Choose a **future date** to predict price
- 📉 Visualize the **last 60 days** using **candlestick chart** (Plotly)
- 📊 View model performance metrics (MAE & RMSE)
- 🔁 Automatically trains and saves the model if not already present

---

## 🛠️ Tech Stack

- **Backend**: Flask, Keras (TensorFlow), Scikit-learn
- **Frontend**: Bootstrap 5, Plotly.js, Axios
- **Data Source**: [Yahoo Finance via `yfinance`](https://pypi.org/project/yfinance/)
- **Model**: 2-layer LSTM with Dense output

---

## 📷 Demo UI

![App Screenshot](https://i.imgur.com/FakeScreenshot.png)
> *Note: Replace with actual screenshot from your running app.*

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/stock-predictor-flask.git
cd stock-predictor-flask
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install flask yfinance pandas numpy scikit-learn tensorflow plotly
```

---

## 🚦 Run the App

```bash
python app.py
```

Visit: `http://127.0.0.1:5000`

---

## 🧠 Model Details

* Uses 60 previous days of closing price data
* Forecasts price for the user-specified future date
* Trained on **AAPL** historical data from `2020-01-01` to `2024-01-01`
* Evaluation Metrics:

  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Squared Error)

---

## 📂 File Structure

```bash
├── app.py              # Main Flask app
├── lstm_model.h5       # Saved LSTM model
├── metrics.json        # MAE, MSE, RMSE values
├── README.md           # Project documentation
```

---

## 🔒 Notes

* App uses only **local model inference**; no external API calls for ML predictions.
* If no model exists, it will automatically train one using AAPL data.

---

## 🤝 Contributions

Feel free to fork, raise issues, or submit pull requests!

---

## 📄 License

This project is licensed under the MIT License.
