import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_data(data, sequence_length):
    data_selected = data['total_modal_biaya'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_selected)

    X_train = []
    y_train = []
    all_dates = []

    for i in range(sequence_length, len(data_scaled)):
        X_train.append(data_scaled[i-sequence_length:i, 0])
        y_train.append(data_scaled[i, 0])
        all_dates.append(data.index[i])

    train_size = int(len(X_train) * 0.8)

    xt_train = X_train[:train_size]
    xt_val = X_train[train_size:]
    Yl_train = y_train[:train_size]
    Yl_val = y_train[train_size:]

    train_dates = all_dates[:train_size]
    val_dates = all_dates[train_size:]

    # Konversi list ke numpy array
    X_train, y_train = np.array(xt_train), np.array(Yl_train)
    X_val, y_val = np.array(xt_val), np.array(Yl_val)

    # Reshape data untuk LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

    return scaler, data_scaled, X_train, y_train, X_val, y_val, train_dates, val_dates

def prepare_multi_data(data, sequence_length, target_column='total_modal_biaya', 
                      categorical_columns=['nama_cabang']):
    """
    Mempersiapkan data dengan multiple features untuk time series prediction
    """
    # Encode categorical variables
    label_encoders = {}
    data_processed = data.copy()
    
    for col in categorical_columns:
        if col in data_processed.columns:
            le = LabelEncoder()
            data_processed[col + '_encoded'] = le.fit_transform(data_processed[col].astype(str))
            label_encoders[col] = le
    
    # Pilih kolom untuk features
    feature_columns = [target_column]
    for col in categorical_columns:
        if col + '_encoded' in data_processed.columns:
            feature_columns.append(col + '_encoded')
    
    # Tambahkan fitur temporal
    if 'tgl_transaksi' in data_processed.columns:
        data_processed['day_of_week'] = pd.to_datetime(data_processed['tgl_transaksi']).dt.dayofweek
        feature_columns.extend(['day_of_week'])
    
    # Pilih dan scale features
    features_data = data_processed[feature_columns].values
    
    # Scale semua features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features_data)
    
    # Scale target secara terpisah untuk inverse transform
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaled = target_scaler.fit_transform(data_processed[[target_column]].values)
    
    X, y = [], []
    all_dates = []
    
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])  # Multiple features sequence
        y.append(target_scaled[i, 0])  # Target value only
        all_dates.append(data_processed.index[i])
    
    # Split train/validation
    train_size = int(len(X) * 0.8)
    
    X_train = np.array(X[:train_size])
    X_val = np.array(X[train_size:])
    y_train = np.array(y[:train_size])
    y_val = np.array(y[train_size:])
    
    train_dates = all_dates[:train_size]
    val_dates = all_dates[train_size:]
    
    return (scaler, target_scaler, features_scaled, X_train, y_train, X_val, y_val, 
            train_dates, val_dates, label_encoders, feature_columns)

# Fungsi untuk membuat beberapa model berbeda
def create_ensemble_models(input_shape):
    # Model 1: LSTM
    model1_input = Input(shape=input_shape)
    x1 = LSTM(32, return_sequences=True)(model1_input)
    x1 = LSTM(16)(x1)
    x1 = Dense(16, activation='relu')(x1)
    output_lstm = Dense(1)(x1)
    model1 = Model(inputs=model1_input, outputs=output_lstm, name='lstm_model')

    # Model 2: GRU
    model2_input = Input(shape=input_shape)
    x2 = GRU(32, return_sequences=True)(model2_input)
    x2 = GRU(16)(x2)
    x2 = Dense(16, activation='relu')(x2)
    output_gru = Dense(1)(x2)
    model2 = Model(inputs=model2_input, outputs=output_gru, name='gru_model')

    # Model 3: Bidirectional LSTM
    model3_input = Input(shape=input_shape)
    x3 = Bidirectional(LSTM(32, return_sequences=True))(model3_input)
    x3 = Bidirectional(LSTM(16))(x3)
    x3 = Dense(16, activation='relu')(x3)
    output_bilstm = Dense(1)(x3)
    model3 = Model(inputs=model3_input, outputs=output_bilstm, name='bilstm_model')

    models = [model1, model2, model3]
    for model in models:
        model.compile(optimizer='adam', loss='mean_squared_error')

    return models

# Fungsi untuk melatih setiap model
def train_ensemble_models(models, X_train, y_train, X_val, y_val, epochs, batch_size=32):
    trained_models = []
    histories = []

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    for i, model in enumerate(models):
        print(f"Training model {i+1}/{len(models)}: {model.name}")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        trained_models.append(model)
        histories.append(history)

    return trained_models, histories


# Fungsi untuk membuat prediksi individual
def individual_predictions(trained_dl_models, X_train, X_val):
    individual_predictions_train = {}
    for i, model in enumerate(trained_dl_models):
        pred = model.predict(X_train)
        # pred = scaler.inverse_transform(pred)
        individual_predictions_train[model.name] = pred.flatten()
    
    individual_predictions_val = {}
    for i, model in enumerate(trained_dl_models):
        pred = model.predict(X_val)
        # pred = scaler.inverse_transform(pred)
        individual_predictions_val[model.name] = pred.flatten()
    
    return individual_predictions_train, individual_predictions_val


# Fungsi prediksi ensemble
def predict_ensemble(models, scaler, last_sequence, meta_model, prediction_days=7):

    ensemble_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(prediction_days):
        model_predictions = []
        current_sequence_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], 1))

        # Prediksi dari setiap model
        for model in models:
            next_day = model.predict(current_sequence_reshaped, verbose=0)[0][0]
            model_predictions.append(next_day)

        # prediksi dari semua model
        model_predictions = np.array(model_predictions).reshape(1, -1)

        next_day_ensemble = meta_model.predict(model_predictions)[0]

        noise = np.random.normal(0, 0.002, 1)[0]
        next_day_ensemble += noise

        # Tambahkan ke hasil prediksi
        ensemble_predictions.append(next_day_ensemble)

        # karena sentimen hanya mempengaruhi output akhir
        current_sequence = np.append(current_sequence[1:], next_day_ensemble)

    # Mengembalikan nilai ke skala asli
    ensemble_predictions_scaled = np.array(ensemble_predictions).reshape(-1, 1)
    ensemble_predictions_original = scaler.inverse_transform(ensemble_predictions_scaled)

    return ensemble_predictions_original


# Fungsi untuk memplot prediksi ensemble
def plot_ensemble_predictions(
    total_penjualan_harian,
    train_dates,
    val_dates,
    scaler,
    ensemble_preds_hist_train,
    ensemble_preds_hist_val,
    df_future_predictions,
    label_column='total_modal_biaya',
    date_column='tgl_transaksi'
):
    fig = go.Figure()

    # Plot data aktual
    fig.add_trace(go.Scatter(
        x=total_penjualan_harian[date_column],
        y=total_penjualan_harian[label_column],
        mode='lines',
        name='Aktual',
        line=dict(color='blue', width=2)
    ))

    # Plot prediksi ensemble pada data train
    train_idx = total_penjualan_harian.index[train_dates[0]:train_dates[-1]+1]
    fig.add_trace(go.Scatter(
        x=total_penjualan_harian.loc[train_idx, date_column],
        y=scaler.inverse_transform(ensemble_preds_hist_train.reshape(-1, 1)).flatten(),
        mode='lines',
        name='Prediksi Ensemble (Train)',
        line=dict(color='green', width=2, dash='solid')
    ))

    # Plot prediksi ensemble pada data validasi
    val_idx = total_penjualan_harian.index[val_dates[0]:val_dates[-1]+1]
    fig.add_trace(go.Scatter(
        x=total_penjualan_harian.loc[val_idx, date_column],
        y=scaler.inverse_transform(ensemble_preds_hist_val.reshape(-1, 1)).flatten(),
        mode='lines',
        name='Prediksi Ensemble (Validasi)',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Plot prediksi masa depan
    fig.add_trace(go.Scatter(
        x=df_future_predictions.index,
        y=df_future_predictions['Prediksi Modal'],
        mode='lines',
        name='Prediksi Masa Depan',
        line=dict(color='orange', width=2)
    ))

    # Garis batas akhir data aktual
    fig.add_vline(
        x=total_penjualan_harian[date_column].iloc[-1],
        line_dash="dash",
        line_color="black",
        name='Hari Terakhir Aktual'
    )

    # Properti grafik
    fig.update_layout(
        title='Visualisasi Aktual dan Prediksi',
        xaxis_title='Tanggal',
        yaxis_title='Total Modal',
        hovermode="x unified",
        legend_title="Keterangan",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,       
            xanchor="center",  
            x=0.5 
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig

def plot_future_predictions(future_dates, future_predictions):
    # Konversi tanggal ke string agar plotly tidak bingung
    future_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in future_dates]
    future_predictions = np.array(future_predictions).flatten()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines+markers',
        marker=dict(size=8, color='orange'),
        name='Prediksi Modal Masa Depan',
        line=dict(color='orange', width=2)
    ))
    fig.update_layout(
        title='Prediksi Modal 7 Hari ke Depan',
        xaxis_title='Tanggal',
        yaxis_title='Modal+Biaya',
        hovermode="x unified",
        legend_title="Legend",
        height=500
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    return fig

#fungsi untuk evaluasi model
def evaluate_model(y_val, ensemble_preds_hist_val):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    mape = mean_absolute_percentage_error(y_val, ensemble_preds_hist_val)
    mae = mean_absolute_error(y_val, ensemble_preds_hist_val)
    mse = mean_squared_error(y_val, ensemble_preds_hist_val)
    rmse = np.sqrt(mse)

    print(f"Evaluasi hasil prediksi pada Data Validasi:")
    print(f"MAE  : {mae:.4f}")
    print(f"MAPE : {mape:.4f}%")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return mape, mae, mse, rmse

# fungsi main pipeline
def main(loaded_data, trained_dl_models, meta_model,  PREDICTION_DAYS, SEQUENCE_LENGTH):

    scaler, data_scaled, X_train, y_train, X_val, y_val, train_dates, val_dates = prepare_data(loaded_data, SEQUENCE_LENGTH)
    
    individual_predictions_train, individual_predictions_val = individual_predictions(trained_dl_models, X_train, X_val)

    train_preds = list(individual_predictions_train.values())
    val_preds = list(individual_predictions_val.values())

    X_stack_train = np.stack(train_preds, axis=1)
    X_stack_val = np.stack(val_preds, axis=1)

    # prediksi menggunakan meta model
    ensemble_preds_hist_train = meta_model.predict(X_stack_train)
    ensemble_preds_hist_val = meta_model.predict(X_stack_val)

    last_date = loaded_data['tgl_transaksi'].iloc[-1]
    if isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()

    future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(PREDICTION_DAYS)]

    last_sequence = data_scaled[-SEQUENCE_LENGTH:]

    future_predictions = predict_ensemble(trained_dl_models, scaler, last_sequence, meta_model)

    # Menampilkan hasil prediksi masa depan dalam bentuk DataFrame
    df_future_predictions = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi Modal': future_predictions.flatten()
    })
    df_future_predictions.set_index('Tanggal', inplace=True)

    df_future_predictions

    fig1 = plot_ensemble_predictions(
        loaded_data,
        train_dates,
        val_dates,
        scaler,
        ensemble_preds_hist_train,
        ensemble_preds_hist_val,
        df_future_predictions
    )

    fig2 = plot_future_predictions(future_dates, future_predictions)

    mape, mae, mse, rmse = evaluate_model(y_val, ensemble_preds_hist_val)

    return fig1, fig2, df_future_predictions, (mape, mae, mse, rmse)

