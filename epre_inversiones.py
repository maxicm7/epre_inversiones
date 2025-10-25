import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import traceback
import json
from huggingface_hub import InferenceClient
import yfinance as yf

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
        response = requests.get(url, headers=headers, timeout=8)
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
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"  # ✅ Sin espacio al final
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
            if compra != "-" and venta != "-":
                try:
                    float(compra)
                    float(venta)
                    data.append({
                        "moneda": moneda,
                        "compra": compra,
                        "venta": venta,
                        "fecha": fecha,
                        "variacion": variacion
                    })
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Fondos ---
@st.cache_data(ttl=600)
def scrape_iol_fondos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondos/todos"  # ✅ Sin espacio
    result = scrape_table(url, min_cols=9)
    if "error" in result:
        return result
    data = []
    for row in result["rows"][:20]:
        cols = row.find_all("td")
        if len(cols) >= 9:
            fondo = cols[0].get_text(strip=True)
            ultimo_str = cols[3].get_text(strip=True).replace("AR$ ", "").replace("US$ ", "")
            var = cols[4].get_text(strip=True)
            if ultimo_str and ultimo_str != "-":
                try:
                    ultimo = float(ultimo_str.replace(".", "").replace(",", "."))
                    data.append({
                        "fondo": fondo,
                        "ultimo": ultimo,
                        "variacion": var
                    })
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Bonos ---
@st.cache_data(ttl=600)
def scrape_iol_bonos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"  # ✅ Sin espacio
    result = scrape_table(url, min_cols=13)
    if "error" in result:
        return result
    data = []
    for row in result["rows"][:30]:
        cols = row.find_all("td")
        if len(cols) >= 13:
            simbolo = cols[0].get_text(strip=True).replace("\n", "").strip()
            ultimo_str = cols[1].get_text(strip=True)
            var = cols[2].get_text(strip=True)
            if ultimo_str and ultimo_str != "-":
                try:
                    ultimo = float(ultimo_str.replace(".", "").replace(",", "."))
                    data.append({
                        "simbolo": simbolo,
                        "ultimo": ultimo,
                        "variacion": var
                    })
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Página: Datos en Vivo – IOL ---
def page_datos_en_vivo_iol():
    st.header("📡 Datos en Vivo – InvertirOnline (IOL)")
    st.markdown("Cotizaciones actualizadas del mercado argentino.")

    tabs = st.tabs(["💱 Monedas", "📊 Fondos", "🎫 Bonos"])

    with tabs[0]:
        with st.spinner("Cargando monedas..."):
            data = scrape_iol_monedas()
        if "error" in data:  # ✅ CORREGIDO: faltaba `data`
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Monedas]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[1]:
        with st.spinner("Cargando fondos..."):
            data = scrape_iol_fondos()
        if "error" in data:  # ✅ CORREGIDO
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Fondos]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[2]:
        with st.spinner("Cargando bonos..."):
            data = scrape_iol_bonos()
        if "error" in data:  # ✅ CORREGIDO
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Bonos]({data['fuente']}) | Actualizado: {data['actualizado']}")

# --- Funciones auxiliares ---
def main_page():
    st.title("BPNos – Bonos, Fondos y Dólar")
    st.write("Bienvenido. Usa el menú lateral para navegar.")

def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            st.error("No se encontraron datos para los tickers ingresados.")
            return None
        return data
    except Exception as e:
        st.error(f"Error al cargar datos de Yahoo Finance: {e}")
        return None

def calculate_portfolio_performance(prices, weights):
    returns = prices.pct_change().dropna()
    portfolio_return = (returns * weights).sum(axis=1)
    cumulative_return = (1 + portfolio_return).cumprod()
    return cumulative_return

def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    portfolios = st.session_state.get("portfolios", {})
    portfolio_name = st.text_input("Nombre del portafolio")
    tickers = st.text_area("Tickers (separados por comas)", "AAPL, MSFT")
    weights = st.text_area("Pesos (deben sumar 1.0)", "0.5, 0.5")
    
    if st.button("Guardar Portafolio"):
        if portfolio_name and tickers and weights:
            tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            try:
                weights_list = [float(w.strip()) for w in weights.split(",") if w.strip()]
            except ValueError:
                st.error("Los pesos deben ser números.")
                return
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
    portfolio = portfolios[name]
    st.json(portfolio)

    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime("today"))
    
    if st.button("Calcular Rendimiento"):
        prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None:
            cum_return = calculate_portfolio_performance(prices, portfolio["weights"])
            st.line_chart(cum_return)

def page_risk_management():
    st.header("⚠️ Gestión de Riesgo")
    st.info("Próximamente: VaR, Drawdowns, Stress Testing.")

def get_hf_response(prompt, api_key, model, temp=0.5):
    try:
        client = InferenceClient(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=temp
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error con Hugging Face: {str(e)}"

def page_investment_insights_chat():
    st.header("💬 Chat de Análisis Cualitativo")
    if not st.session_state.get('hf_api_key'):
        st.warning("Ingresa tu Hugging Face API Key en la barra lateral.")
        return

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Escribe tu consulta de inversión..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Pensando..."):
            response = get_hf_response(
                prompt,
                st.session_state.hf_api_key,
                st.session_state.hf_model,
                st.session_state.hf_temp
            )
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# --- Inicialización de estado ---
default_session_values = {
    'selected_page': "Welcome Page",
    'hf_api_key': "",
    'hf_model': "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'hf_temp': 0.5,
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
st.session_state.hf_model = st.sidebar.selectbox("Modelo de Lenguaje", [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-7b-it"
])
st.session_state.hf_temp = st.sidebar.slider("Temperatura", 0.1, 1.0, 0.5, 0.1)

# --- Navegación ---
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
page_options = [
    "Welcome Page",
    "Create/Edit Portfolios",
    "View Portfolio Returns",
    "Gestión de Riesgo",
    "Datos en Vivo – IOL",
    "Chat de Análisis Cualitativo"
]
if 'selected_page' not in st.session_state or st.session_state.selected_page not in page_options:
    st.session_state.selected_page = "Welcome Page"

page = st.sidebar.radio("Select Section", page_options, index=page_options.index(st.session_state.selected_page))
if page != st.session_state.selected_page:
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
else:
    main_page()
