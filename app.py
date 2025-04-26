from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import time

# ←— force non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
from flask_cors import CORS
from matplotlib.ticker import StrMethodFormatter
# Tensorflow (Keras & LSTM) related packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import required storage package from Google Cloud Storage
from google.cloud import storage
from statsmodels.tsa.statespace.sarimax import SARIMAX




from flask import Flask, jsonify, request, make_response
import os
import pandas as pd
import numpy as np
import time
from dateutil import *
from datetime import timedelta
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from prophet import Prophet
from google.cloud import storage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Google Cloud Storage client initialization
gcs_client = storage.Client()

# Constants
GCS_BUCKET_NAME = 'forecasting-lstm-images'
BASE_IMAGE_URL = 'https://storage.googleapis.com/forecasting-lstm-images/'
LOCAL_IMAGE_DIR = 'static/images/'

# CORS handling
def prepare_cors_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

def add_cors_headers(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods", "PUT, GET, POST, DELETE, OPTIONS")
    return response

# Utility to create dataset from time series
def generate_dataset(sequence, look_back=30):
    X, y = [], []
    for i in range(len(sequence) - look_back):
        X.append(sequence[i:i+look_back, 0])
        y.append(sequence[i+look_back, 0])
    return np.array(X), np.array(y)

# Upload local image to Google Cloud Storage
def upload_to_gcs(file_name):
    bucket = gcs_client.get_bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file_name)
    blob.upload_from_filename(f"{LOCAL_IMAGE_DIR}{file_name}")

# Create and save loss plot
def save_loss_plot(history, file_name, title):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {title}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(LOCAL_IMAGE_DIR + file_name)
    plt.close()

# Create and save LSTM predictions plot
def save_lstm_forecast_plot(Y_train, Y_test, predictions, file_name, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(len(Y_train)), Y_train, 'g', label="Training Data")
    ax.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, 'b', marker='.', label="Actual Data")
    ax.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), predictions, 'r', label="Predicted Data")
    ax.set_title(f'LSTM Forecast - {title}')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Scaled Values')
    ax.legend()
    fig.savefig(LOCAL_IMAGE_DIR + file_name)
    plt.close(fig)

# Create and save time series actual data plot
def save_actual_data_plot(all_days, Ys, file_name, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(all_days, Ys, marker='.', label=title)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_title(f'All {title} Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    fig.savefig(LOCAL_IMAGE_DIR + file_name)
    plt.close(fig)

# Other functions (Prophet plot, SARIMAX plot, Forecast APIs, etc.) 
# will continue here...

@app.route('/api/forecast', methods=['POST'])
def predict_issue_forecast():
    payload = request.get_json()
    issues = payload["issues"]
    forecast_type = payload["type"]
    repository_name = payload["repo"]

    issues_df = pd.DataFrame(issues).groupby([forecast_type], as_index=False).count()[[forecast_type, 'issue_number']]
    issues_df = issues_df.rename(columns={forecast_type: 'ds', 'issue_number': 'y'})
    issues_df['ds'] = pd.to_datetime(issues_df['ds'])

    first_date = issues_df['ds'].min()
    last_date = issues_df['ds'].max()
    all_dates = pd.date_range(start=first_date, end=last_date, freq='D')

    issue_counts = np.zeros(len(all_dates), dtype=float)
    for dt, count in zip(issues_df['ds'], issues_df['y']):
        idx = (dt - first_date).days
        issue_counts[idx] = count

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_counts = scaler.fit_transform(issue_counts.reshape(-1, 1))

    train_size = int(len(scaled_counts) * 0.8)
    train_set, test_set = scaled_counts[:train_size], scaled_counts[train_size:]

    X_train, Y_train = generate_dataset(train_set, 30)
    X_test, Y_test = generate_dataset(test_set, 30)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(100, input_shape=(1, 30)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=70,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1,
        shuffle=False
    )

    # Save plots
    loss_plot_name = f"model_loss_{forecast_type}_{repository_name}.png"
    lstm_plot_name = f"lstm_generated_data_{forecast_type}_{repository_name}.png"
    actual_plot_name = f"all_issues_data_{forecast_type}_{repository_name}.png"

    save_loss_plot(history, loss_plot_name, forecast_type)
    predictions = model.predict(X_test)
    save_lstm_forecast_plot(Y_train, Y_test, predictions, lstm_plot_name, forecast_type)
    save_actual_data_plot(all_dates, issue_counts, actual_plot_name, 'Issues')

    # Upload images
    for img_name in [loss_plot_name, lstm_plot_name, actual_plot_name]:
        upload_to_gcs(img_name)

    # Prophet Forecast
    prophet_generated = False
    prophet_plot_name = f"prophet_forecast_{forecast_type}_{repository_name}.png"
    try:
        if len(issues_df) >= 10:
            prophet_model = Prophet()
            prophet_model.fit(issues_df)
            future = prophet_model.make_future_dataframe(periods=30)
            forecast = prophet_model.predict(future)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(issues_df['ds'], issues_df['y'], 'bo-', label='Actual Data')
            ax.plot(forecast['ds'], forecast['yhat'], 'orange', linestyle='--', label='Forecasted')
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
            ax.set_title(f"Prophet Forecast for {repository_name}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Issue Count')
            ax.legend()
            fig.savefig(LOCAL_IMAGE_DIR + prophet_plot_name)
            plt.close(fig)
            upload_to_gcs(prophet_plot_name)
            prophet_generated = True
    except Exception as e:
        print(f"Skipping Prophet: {str(e)}")

    # SARIMAX Forecast
    sarimax_plot_name = f"sarimax_forecast_{forecast_type}_{repository_name}.png"
    try:
        df_sarimax = issues_df.set_index('ds').resample('D').sum().fillna(0)
        sarimax_model = SARIMAX(df_sarimax['y'],
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 7),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        sarimax_result = sarimax_model.fit(disp=False)
        sarimax_forecast = sarimax_result.get_forecast(steps=30)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_sarimax.index, df_sarimax['y'], 'bo-', label='Actual')
        ax.plot(sarimax_forecast.predicted_mean.index, sarimax_forecast.predicted_mean, 'orange', linestyle='--', label='Forecasted')
        ax.fill_between(sarimax_forecast.predicted_mean.index,
                        sarimax_forecast.conf_int().iloc[:, 0],
                        sarimax_forecast.conf_int().iloc[:, 1],
                        color='orange', alpha=0.3)
        ax.set_title(f"SARIMAX Forecast for {repository_name}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Issue Count')
        ax.legend()
        fig.savefig(LOCAL_IMAGE_DIR + sarimax_plot_name)
        plt.close(fig)
        upload_to_gcs(sarimax_plot_name)
    except Exception as e:
        print(f"Skipping SARIMAX: {str(e)}")

    return jsonify({
        "model_loss_image_url": BASE_IMAGE_URL + loss_plot_name,
        "lstm_generated_image_url": BASE_IMAGE_URL + lstm_plot_name,
        "all_issues_data_image": BASE_IMAGE_URL + actual_plot_name,
        "prophet_forecast_image_url": BASE_IMAGE_URL + prophet_plot_name if prophet_generated else None,
        "sarimax_forecast_image_url": BASE_IMAGE_URL + sarimax_plot_name
    })

@app.route('/api/forecast/pulls', methods=['POST'])
def predict_pull_requests_forecast():
    payload = request.get_json()
    pulls = payload["pulls"]
    repository_name = payload["repo"]

    pulls_df = pd.DataFrame(pulls).groupby(['created_at'], as_index=False).count()[['created_at', 'pull_number']]
    pulls_df = pulls_df.rename(columns={'created_at': 'ds', 'pull_number': 'y'})
    pulls_df['ds'] = pd.to_datetime(pulls_df['ds'])

    if pulls_df.shape[0] < 2:
        return jsonify({
            "model_loss_image_url": None,
            "lstm_generated_image_url": None,
            "all_pulls_data_image": None,
            "prophet_forecast_image_url": None,
            "sarimax_forecast_image_url": None
        })

    first_date = pulls_df['ds'].min()
    last_date = pulls_df['ds'].max()
    all_dates = pd.date_range(start=first_date, end=last_date, freq='D')

    pull_counts = np.zeros(len(all_dates), dtype=float)
    for dt, count in zip(pulls_df['ds'], pulls_df['y']):
        idx = (dt - first_date).days
        pull_counts[idx] = count

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_counts = scaler.fit_transform(pull_counts.reshape(-1, 1))

    train_size = int(len(scaled_counts) * 0.8)
    train_set, test_set = scaled_counts[:train_size], scaled_counts[train_size:]

    X_train, Y_train = generate_dataset(train_set, 30)
    X_test, Y_test = generate_dataset(test_set, 30)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return jsonify({
            "model_loss_image_url": None,
            "lstm_generated_image_url": None,
            "all_pulls_data_image": None,
            "prophet_forecast_image_url": None,
            "sarimax_forecast_image_url": None
        })

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(100, input_shape=(1, 30)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=70,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1,
        shuffle=False
    )

    # Save plots
    loss_plot_name = f"model_loss_pulls_{repository_name}.png"
    lstm_plot_name = f"lstm_generated_data_pulls_{repository_name}.png"
    actual_plot_name = f"all_pulls_data_{repository_name}.png"

    save_loss_plot(history, loss_plot_name, "Pulls")
    predictions = model.predict(X_test)
    save_lstm_forecast_plot(Y_train, Y_test, predictions, lstm_plot_name, "Pulls")
    save_actual_data_plot(all_dates, pull_counts, actual_plot_name, "Pull Requests")

    for img_name in [loss_plot_name, lstm_plot_name, actual_plot_name]:
        upload_to_gcs(img_name)

    # Prophet and SARIMAX can be added same way if needed (optional)

    return jsonify({
        "model_loss_image_url": BASE_IMAGE_URL + loss_plot_name,
        "lstm_generated_image_url": BASE_IMAGE_URL + lstm_plot_name,
        "all_pulls_data_image": BASE_IMAGE_URL + actual_plot_name,
        "prophet_forecast_image_url": None,
        "sarimax_forecast_image_url": None
    })


@app.route('/api/forecast/branches', methods=['POST'])
def predict_branches_forecast():
    payload = request.get_json()
    branches = payload["branches"]
    repository_name = payload["repo"]

    branches_df = pd.DataFrame(branches)

    today = pd.Timestamp.today()
    if branches_df['created_at'].nunique() <= 1:
        branches_df = branches_df.sort_values('branch_name')
        branches_df['created_at'] = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(branches_df))]

    branches_df = branches_df.groupby(['created_at'], as_index=False).count()[['created_at', 'branch_name']]
    branches_df = branches_df.rename(columns={'created_at': 'ds', 'branch_name': 'y'})
    branches_df['ds'] = pd.to_datetime(branches_df['ds'])

    if branches_df.shape[0] < 2:
        return jsonify({
            "model_loss_image_url": None,
            "lstm_generated_image_url": None,
            "all_branches_data_image": None,
            "prophet_forecast_image_url": None,
            "sarimax_forecast_image_url": None
        })

    first_date = branches_df['ds'].min()
    last_date = branches_df['ds'].max()
    all_dates = pd.date_range(start=first_date, end=last_date, freq='D')

    branch_counts = np.zeros(len(all_dates), dtype=float)
    for dt, count in zip(branches_df['ds'], branches_df['y']):
        idx = (dt - first_date).days
        branch_counts[idx] = count

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_counts = scaler.fit_transform(branch_counts.reshape(-1, 1))

    train_size = int(len(scaled_counts) * 0.8)
    train_set, test_set = scaled_counts[:train_size], scaled_counts[train_size:]

    X_train, Y_train = generate_dataset(train_set, 30)
    X_test, Y_test = generate_dataset(test_set, 30)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return jsonify({
            "model_loss_image_url": None,
            "lstm_generated_image_url": None,
            "all_branches_data_image": None,
            "prophet_forecast_image_url": None,
            "sarimax_forecast_image_url": None
        })

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential([
        LSTM(100, input_shape=(1, 30)),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=20,
        batch_size=70,
        validation_data=(X_test, Y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1,
        shuffle=False
    )

    # Save plots
    loss_plot_name = f"model_loss_branches_{repository_name}.png"
    lstm_plot_name = f"lstm_generated_data_branches_{repository_name}.png"
    actual_plot_name = f"all_branches_data_{repository_name}.png"

    save_loss_plot(history, loss_plot_name, "Branches")
    predictions = model.predict(X_test)
    save_lstm_forecast_plot(Y_train, Y_test, predictions, lstm_plot_name, "Branches")
    save_actual_data_plot(all_dates, branch_counts, actual_plot_name, "Branches")

    for img_name in [loss_plot_name, lstm_plot_name, actual_plot_name]:
        upload_to_gcs(img_name)

    return jsonify({
        "model_loss_image_url": BASE_IMAGE_URL + loss_plot_name,
        "lstm_generated_image_url": BASE_IMAGE_URL + lstm_plot_name,
        "all_branches_data_image": BASE_IMAGE_URL + actual_plot_name,
        "prophet_forecast_image_url": None,
        "sarimax_forecast_image_url": None
    })

@app.route('/api/stars', methods=['GET'])
def generate_star_chart():
    repositories = [
        "ollama/ollama",
        "langchain-ai/langchain",
        "langchain-ai/langgraph",
        "microsoft/autogen",
        "openai/openai-cookbook",
        "meta-llama/llama3",
        "elastic/elasticsearch",
        "milvus-io/pymilvus"
    ]

    github_token = os.getenv('GITHUB_TOKEN', 'GITHUB_TOKEN')
    headers = {"Authorization": f'token {github_token}'} if github_token else {}

    GITHUB_API_URL = "https://api.github.com/repos/"

    repo_star_data = []
    for repo_full_name in repositories:
        url = GITHUB_API_URL + repo_full_name
        response = requests.get(url, headers=headers)
        repo_info = response.json()
        repo_star_data.append({
            "name": repo_full_name.split("/")[-1],  # Repo name only (not user/org)
            "stars": repo_info.get("stargazers_count", 0)
        })

    # Post to LSTM microservice to generate bar chart
    STAR_CHART_API = "https://lstm-app-708210591622.us-central1.run.app/api/stars"
    star_chart_response = requests.post(
        STAR_CHART_API,
        json={"repos": repo_star_data},
        headers={'content-type': 'application/json'}
    )

    if star_chart_response.status_code == 200:
        return jsonify({
            "star_bar_chart_url": star_chart_response.json().get("star_bar_chart_url")
        })
    else:
        return jsonify({
            "error": "Failed to generate star chart"
        }), 500
@app.route('/api/forks', methods=['GET'])
def generate_fork_chart():
    """
    Fetches fork counts for multiple GitHub repositories,
    sends them to the forecasting microservice,
    and returns the generated bar chart URL.
    """
    repositories = [
        "ollama/ollama",
        "langchain-ai/langchain",
        "langchain-ai/langgraph",
        "microsoft/autogen",
        "openai/openai-cookbook",
        "meta-llama/llama3",
        "elastic/elasticsearch",
        "milvus-io/pymilvus"
    ]

    github_token = os.getenv('GITHUB_TOKEN', 'GITHUB_TOKEN')
    headers = {"Authorization": f'token {github_token}'} if github_token else {}

    GITHUB_API_URL = "https://api.github.com/repos/"

    repo_fork_data = []
    for repo_full_name in repositories:
        url = GITHUB_API_URL + repo_full_name
        response = requests.get(url, headers=headers)
        repo_info = response.json()
        repo_fork_data.append({
            "name": repo_full_name.split("/")[-1],  # Get only the repo name (not organization)
            "forks": repo_info.get("forks_count", 0)
        })

    # Post fork data to LSTM service
    FORK_CHART_API = "https://lstm-app-708210591622.us-central1.run.app/api/forks"
    fork_chart_response = requests.post(
        FORK_CHART_API,
        json={"repos": repo_fork_data},
        headers={'content-type': 'application/json'}
    )

    if fork_chart_response.status_code == 200:
        return jsonify({
            "forks_bar_chart_url": fork_chart_response.json().get("forks_bar_chart_url")
        })
    else:
        return jsonify({
            "error": "Failed to generate forks chart"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)

