import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest, norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.api import VAR
import yfinance as yf
import warnings
import traceback
import json
from huggingface_hub import InferenceClient
import pypdf
import requests
from bs4 import BeautifulSoup
import time

# --- Configuración ---
st.set_page_config(layout="wide", page_title="BPNos – Bonos, Fondos y Dólar")

# --- Constante para el archivo de portafolios ---
PORTFOLIO_FILE = "portfolios_data1.json"

# --- Funciones para Cargar/Guardar Portafolios ---
def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolios = json.load(f)
            return portfolios
        except Exception as e:
            st.error(f"Error al cargar portafolios desde {PORTFOLIO_FILE}: {e}")
            return {}
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        print(f"Error al guardar: {e}")
        traceback.print_exc()
        return False, str(e)

# --- Scraping genérico ---
def scrape_table(url, min_cols, max_rows=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if not table:
            return {"error": "No se encontró la tabla."}
        rows = table.find_all("tr")[1:]
        if max_rows:
            rows = rows[:max_rows]
        return {"rows": rows, "actualizado": time.strftime("%Y-%m-%d %H:%M")}
    except Exception as e:
        return {"error": f"Error al scrapear: {str(e)}"}

# --- Monedas ---
@st.cache_data(ttl=300)
def scrape_iol_monedas():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"
    result = scrape_table(url, min_cols=5)
    if "error" in result:
        return result
    data = []
    for row in result["rows"]:
        cols = row.find_all("td")
        if len(cols) >= 5:
            moneda = cols[0].get_text(strip=True)
            compra = cols[1].get_text(strip=True).replace(".", "").replace(",", ".")
            venta = cols[2].get_text(strip=True).replace(".", "").replace(",", ".")
            fecha = cols[3].get_text(strip=True)
            variacion = cols[4].get_text(strip=True)
            try:
                float(compra); float(venta)
                data.append({
                    "moneda": moneda,
                    "compra": compra,
                    "venta": venta,
                    "fecha": fecha,
                    "variacion": variacion
                })
            except ValueError:
                continue
    if not data:
        return {"error": "No hay datos válidos en Monedas."}
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Fondos ---
@st.cache_data(ttl=600)
def scrape_iol_fondos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondos/todos"
    result = scrape_table(url, min_cols=9)
    if "error" in result:
        return result
    data = []
    for row in result["rows"]:
        cols = row.find_all("td")
        if len(cols) >= 9:
            fondo = cols[0].get_text(strip=True)
            tipo = cols[1].get_text(strip=True)
            horizonte = cols[2].get_text(strip=True)
            ultimo_str = cols[3].get_text(strip=True).replace("AR$ ", "").replace("US$ ", "")
            var = cols[4].get_text(strip=True)
            if ultimo_str and ultimo_str != "-":
                try:
                    ultimo = float(ultimo_str.replace(".", "").replace(",", "."))
                    data.append({
                        "fondo": fondo,
                        "tipo_y_moneda": tipo,
                        "horizonte": horizonte,
                        "ultimo": ultimo,
                        "variacion": var
                    })
                except ValueError:
                    continue
    if not data:
        return {"error": "No hay datos válidos en Fondos."}
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Bonos ---
@st.cache_data(ttl=600)
def scrape_iol_bonos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if not table:
            return {"error": "No se encontró la tabla de bonos."}
        rows = table.find_all("tr")[1:]  # Saltar encabezado
        data = []
        for row in rows[:30]:  # Tomar las primeras 30 filas
            cols = row.find_all("td")
            if len(cols) >= 13:
                simbolo = cols[0].get_text(strip=True).replace("\n", "").strip()
                ultimo_str = cols[1].get_text(strip=True).replace(".", "").replace(",", ".")
                var = cols[2].get_text(strip=True)
                # Validar que 'ultimo' sea numérico y no vacío
                if ultimo_str and ultimo_str != "-":
                    try:
                        ultimo = float(ultimo_str)
                        data.append({
                            "simbolo": simbolo,
                            "ultimo": ultimo,
                            "variacion": var,
                            "maximo": cols[7].get_text(strip=True).replace(".", "").replace(",", "."),
                            "minimo": cols[8].get_text(strip=True).replace(".", "").replace(",", "."),
                            "cierre": cols[9].get_text(strip=True).replace(".", "").replace(",", "."),
                            "monto": cols[10].get_text(strip=True).replace(".", "").replace(",", "."),
                            "tir": cols[11].get_text(strip=True).replace(",", "."),
                            "duracion": cols[12].get_text(strip=True)
                        })
                    except ValueError:
                        continue
        if not data:
            return {"error": "No se pudieron extraer datos válidos de bonos."}
        return {
            "fuente": url,
            "datos": data,
            "actualizado": time.strftime("%Y-%m-%d %H:%M")
        }
    except Exception as e:
        return {"error": f"Error al scrapear bonos: {str(e)}"}

# --- Nueva página: Datos en Vivo – IOL ---
def page_datos_en_vivo_iol():
    st.header("📡 Datos en Vivo – InvertirOnline (IOL)")
    st.markdown("Cotizaciones actualizadas del mercado argentino.")

    tabs = st.tabs(["💱 Monedas", "📊 Fondos", "🎫 Bonos"])

    with tabs[0]:  # Monedas
        with st.spinner("Cargando monedas..."):
            data = scrape_iol_monedas()
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Monedas]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[1]:  # Fondos
        with st.spinner("Cargando fondos..."):
            data = scrape_iol_fondos()
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Fondos]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[2]:  # Bonos
        with st.spinner("Cargando bonos..."):
            data = scrape_iol_bonos()
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Bonos]({data['fuente']}) | Actualizado: {data['actualizado']}")

# --- Funciones auxiliares mínimas (para evitar errores) ---
def main_page():
    st.title("BPNos – Bonos, Fondos y Dólar")
    st.write("Bienvenido. Usa el menú lateral para navegar.")

def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    portfolios = st.session_state.get("portfolios", {})
    portfolio_name = st.text_input("Nombre del portafolio")
    tickers = st.text_area("Tickers (separados por comas)", "AAPL, MSFT")
    weights = st.text_area("Pesos (deben sumar 1.0)", "0.5, 0.5")
    
    if st.button("Guardar Portafolio"):
        if portfolio_name and tickers and weights:
            tickers_list = [t.strip() for t in tickers.split(",")]
            weights_list = [float(w.strip()) for w in weights.split(",")]
            if len(tickers_list) != len(weights_list):
                st.error("Número de tickers y pesos debe coincidir.")
            elif abs(sum(weights_list) - 1.0) > 1e-6:
                st.error("Los pesos deben sumar 1.0")
            else:
                portfolios[portfolio_name] = {
                    "tickers": tickers_list,
                    "weights": weights_list
                }
                success, msg = save_portfolios_to_file(portfolios)
                if success:
                    st.session_state.portfolios = portfolios
                    st.success("✅ Portafolio guardado.")
                else:
                    st.error(f"❌ Error al guardar: {msg}")
        else:
            st.warning("Completa todos los campos.")

def page_view_portfolio_returns():
    st.header("📈 Ver Rendimiento de Portafolio")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("No hay portafolios guardados.")
        return
    name = st.selectbox("Selecciona un portafolio", list(portfolios.keys()))
    st.json(portfolios[name])

def page_risk_management():
    st.header("⚠️ Gestión de Riesgo")
    st.info("Próximamente: VaR, Drawdowns, Stress Testing.")

def page_investment_insights_chat():
    st.header("💬 Chat de Análisis Cualitativo")
    st.info("Conecta tu API de Hugging Face para análisis con IA.")

def load_stock_data(tickers, start, end, returns=True):
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close']
        if len(tickers) == 1:
            data = data.to_frame(name=tickers[0])
        df_list = []
        for col in data.columns:
            series = data[col].dropna()
            if returns:
                series = series.pct_change().dropna() * 100
            df_list.append(pd.DataFrame({
                'Date': series.index,
                'Entity': col,
                'Actual': series.values
            }))
        return pd.concat(df_list, ignore_index=True)
    except Exception as e:
        st.error(f"Error al cargar datos de Yahoo Finance: {e}")
        return None

def data_visualization():
    st.subheader("Data Visualization")
    for entity in st.session_state.entities:
        st.write(f"**{entity}**")
        st.line_chart(st.session_state.dataframes[entity])

def decomposition():
    st.subheader("Series Decomposition")
    for entity in st.session_state.entities:
        df = st.session_state.dataframes[entity]
        if len(df) < 24:
            st.warning(f"{entity}: se necesitan al menos 24 observaciones.")
            continue
        try:
            result = seasonal_decompose(df['Actual'], model='additive', period=12)
            fig, ax = plt.subplots(4, 1, figsize=(10, 8))
            result.observed.plot(ax=ax[0], title='Original')
            result.trend.plot(ax=ax[1], title='Trend')
            result.seasonal.plot(ax=ax[2], title='Seasonal')
            result.resid.plot(ax=ax[3], title='Residual')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error en descomposición para {entity}: {e}")

def optimal_lags():
    st.subheader("Stationarity & Lags")
    for entity in st.session_state.entities:
        df = st.session_state.dataframes[entity]
        result = adfuller(df['Actual'].dropna())
        st.write(f"**{entity}** – ADF p-value: {result[1]:.4f} ({'Estacionaria' if result[1] < 0.05 else 'No estacionaria'})")
        fig, ax = plt.subplots()
        plot_pacf(df['Actual'].dropna(), lags=20, ax=ax)
        st.pyplot(fig)

def forecast_models():
    st.subheader("Forecasting Models")
    for entity in st.session_state.entities:
        df = st.session_state.dataframes[entity]
        if len(df) < 30:
            st.warning(f"{entity}: se necesitan más datos para forecasting.")
            continue
        try:
            model = auto_arima(df['Actual'], seasonal=False, stepwise=True, suppress_warnings=True)
            forecast = model.predict(n_periods=5)
            st.write(f"**{entity}** – Pronóstico próximos 5 días: {forecast.values}")
        except Exception as e:
            st.error(f"Error en forecasting para {entity}: {e}")

# --- Inicialización de estado ---
default_session_values = {
    'selected_page': "Welcome Page",
    'entities': [], 'dataframes': {}, 'data_type': 'returns',
    'apply_outlier_treatment': False, 'iqr_factor': 1.5,
    'hf_api_key': "",
    'insights_messages': [],
    'portfolio_chat_messages': {},
}
for k, v in default_session_values.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = load_portfolios_from_file()

# --- Sidebar: Configuración General ---
st.sidebar.title("Configuración General")
hf_api_key = st.sidebar.text_input("Hugging Face API Key", type="password", value=st.session_state.get('hf_api_key', ''))
if hf_api_key:
    st.session_state.hf_api_key = hf_api_key
st.session_state.hf_model = st.sidebar.selectbox("Modelo de Lenguaje", ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-8B-Instruct", "google/gemma-7b-it"])
st.session_state.hf_temp = st.sidebar.slider("Temperatura", 0.1, 1.0, 0.5, 0.1)

# --- Sidebar: Forecasting ---
st.sidebar.header("Forecasting: Data Selection")
ticker_input = st.sidebar.text_area("Enter Tickers", "AAPL, MSFT")
start_date = st.sidebar.date_input("Start Date", datetime.today().date() - timedelta(days=3*365))
end_date = st.sidebar.date_input("End Date", datetime.today().date())
data_type_choice = st.sidebar.radio("Forecast Target", ('Daily Returns (%)', 'Adjusted Close Price'))
apply_outlier_treatment_ui = st.sidebar.checkbox("Apply Outlier Treatment (IQR)")
iqr_k_factor_ui = st.sidebar.number_input("IQR Factor (k)", 1.0, 5.0, 1.5, 0.1, disabled=not apply_outlier_treatment_ui)
load_button = st.sidebar.button("Load & Process Data (for Forecasting)")
if load_button:
    tickers_forecast = [t.strip().upper() for t in re.split('[,\s]+', ticker_input) if t.strip()]
    if tickers_forecast:
        df = load_stock_data(tickers_forecast, start_date, end_date, data_type_choice=='Daily Returns (%)')
        if df is not None:
            st.session_state.entities = sorted(df['Entity'].unique())
            st.session_state.dataframes = {e: df[df['Entity']==e][['Actual']] for e in st.session_state.entities}
            st.session_state.data_type = 'returns' if data_type_choice=='Daily Returns (%)' else 'prices'
            st.session_state.apply_outlier_treatment = apply_outlier_treatment_ui
            st.session_state.iqr_factor = iqr_k_factor_ui
            st.sidebar.success("Datos cargados.")
            st.rerun()

# --- Navegación ---
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page_options = [
    "Welcome Page",
    "Create/Edit Portfolios",
    "View Portfolio Returns",
    "Gestión de Riesgo",
    "Datos en Vivo – IOL",  # <-- AHORA FUNCIONAL
    "Chat de Análisis Cualitativo",
    "--- Forecasting ---",
    "Data Visualization",
    "Series Decomposition",
    "Stationarity & Lags",
    "Forecasting Models"
]
selectable_page_options = [p for p in page_options if not p.startswith("---")]
if 'selected_page' not in st.session_state or st.session_state.selected_page not in selectable_page_options:
    st.session_state.selected_page = "Welcome Page"
try:
    page_idx = page_options.index(st.session_state.selected_page)
except ValueError:
    st.session_state.selected_page = "Welcome Page"
    page_idx = page_options.index(st.session_state.selected_page)

page = st.sidebar.radio("Select Section", page_options, index=page_idx)
if page != st.session_state.selected_page and not page.startswith("---"):
    st.session_state.selected_page = page
    st.rerun()

# --- Routing ---
if st.session_state.selected_page == "Welcome Page":
    main_page()
elif st.session_state.selected_page == "Create/Edit Portfolios":
    page_create_portfolio()
elif st.session_state.selected_page == "View Portfolio Returns":
    page_view_portfolio_returns()
elif st.session_state.selected_page == "Gestión de Riesgo":
    page_risk_management()
elif st.session_state.selected_page == "Datos en Vivo – IOL":
    page_datos_en_vivo_iol()
elif st.session_state.selected_page == "Chat de Análisis Cualitativo":
    page_investment_insights_chat()
elif st.session_state.selected_page in ["Data Visualization", "Series Decomposition", "Stationarity & Lags", "Forecasting Models"]:
    if not st.session_state.get('entities'):
        st.warning("Carga datos en la barra lateral primero.")
    else:
        globals()[st.session_state.selected_page.replace(" ", "_").lower()]()
else:
    main_page()
