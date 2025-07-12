import streamlit as st
import joblib
import os
import pandas as pd
from tensorflow.keras.models import load_model
import Funtions as fn

# --- Load data dan model ---
data_modal_EMA = joblib.load('data/modal_biaya_harian_EMA.pkl')
data_modal = joblib.load('data/modal_biaya_harian.pkl')

# Model per cabang
meta_model_EMA = joblib.load('model/peramalan_EMA/meta_model.pkl')
dir_EMA = "model/peramalan_EMA"
model_files_EMA = sorted([f for f in os.listdir(dir_EMA) if f.endswith(".h5")])
trained_models_EMA = [load_model(os.path.join(dir_EMA, f)) for f in model_files_EMA]

# Model keseluruhan
meta_model = joblib.load('model/peramalan/meta_model.pkl')
dir_model = "model/peramalan"
model_files = sorted([f for f in os.listdir(dir_model) if f.endswith(".h5")])
trained_models = [load_model(os.path.join(dir_model, f)) for f in model_files]

PREDICTION_DAYS = 7
SEQUENCE_LENGTH = 30

fig1, fig2, df_future_predictions, eval_metrics = fn.main(
        loaded_data=data_modal,
        trained_dl_models=trained_models,
        meta_model=meta_model,
        PREDICTION_DAYS=PREDICTION_DAYS,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH
    )

fig1_EMA, fig2_EMA, df_future_predictions_EMA, eval_metrics_EMA = fn.main(
        loaded_data=data_modal_EMA,
        trained_dl_models=trained_models_EMA,
        meta_model=meta_model_EMA,
        PREDICTION_DAYS=PREDICTION_DAYS,
        SEQUENCE_LENGTH=SEQUENCE_LENGTH
    )

# --- Fungsi Visualisasi Streamlit ---
def main_plot(fig1, fig2, df_future_predictions, eval_metrics):
    st.subheader("Plot Ensemble")
    st.plotly_chart(fig1)
    st.subheader("Plot Future")
    st.plotly_chart(fig2)

    st.subheader("Tabel Prediksi Modal dan Biaya 7 Hari ke Depan")

    def format_rupiah(val):
        return f"Rp {val:,.0f}".replace(",", ".")

    # Format kolom prediksi
    formatted_df = df_future_predictions.copy()
    formatted_df["Prediksi Modal"] = formatted_df["Prediksi Modal"].apply(format_rupiah)

    st.dataframe(formatted_df, use_container_width=True)

    st.subheader("📊 Evaluasi Model")
    mape, mae, mse, rmse = eval_metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("**MAPE**", f"{mape:.2f}%", "↓")
    col2.metric("**MAE**", f"{mae:.2f}", "↓")
    col3.metric("**MSE**", f"{mse:.2f}", "↓")
    col4.metric("**RMSE**", f"{rmse:.2f}", "↓")

# --- Sidebar Pilihan ---
menu = st.sidebar.radio(
    "Pilih Visualisasi",
    (
        "base_data",
        "EMA_data"
    )
)

st.title("Peramalan Modal + Biaya PhiMart")

if menu == "base_data":
    main_plot(fig1, fig2, df_future_predictions, eval_metrics)
else:
    main_plot(fig1_EMA, fig2_EMA, df_future_predictions_EMA, eval_metrics_EMA)