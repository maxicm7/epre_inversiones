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
from scipy.optimize import minimize

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
            if compra != "-" and venta != "-":
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
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# --- Fondos ---
@st.cache_data(ttl=600)
def scrape_iol_fondos():
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondos/todos"
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
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
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
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Monedas]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[1]:
        with st.spinner("Cargando fondos..."):
            data = scrape_iol_fondos()
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Fondos]({data['fuente']}) | Actualizado: {data['actualizado']}")
            # Almacenar datos de fondos en session_state
            st.session_state.iol_data = st.session_state.get('iol_data', {})
            st.session_state.iol_data['fondos'] = data

    with tabs[2]:
        with st.spinner("Cargando bonos..."):
            data = scrape_iol_bonos()
        if "error" in data:
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Bonos]({data['fuente']}) | Actualizado: {data['actualizado']}")
            # Almacenar datos de bonos en session_state
            st.session_state.iol_data = st.session_state.get('iol_data', {})
            st.session_state.iol_data['bonos'] = data

# --- Funciones auxiliares ---
def main_page():
    st.title("BPNos – Bonos, Fondos y Dólar")
    st.write("Bienvenido. Usa el menú lateral para navegar.")

def get_price_from_iol(symbol):
    """
    Busca el precio actual de un símbolo scrapeado de IOL.
    Retorna el precio 'ultimo' o None si no lo encuentra.
    """
    scraped_data = st.session_state.get('iol_data', {})
    for key in ['fondos', 'bonos']:
        data = scraped_data.get(key, {}).get('datos', [])
        for item in data:
            if item.get('simbolo', '').upper() == symbol.upper() or item.get('fondo', '').upper() == symbol.upper():
                return item.get('ultimo')
    return None

def fetch_prices_for_portfolio(tickers, start_date, end_date):
    """
    Intenta obtener precios de yfinance. Si falla o no soporta el símbolo,
    intenta usar los datos scrapeados de IOL como punto de partida.
    """
    # Intentar primero con yfinance
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            st.info("No se encontraron datos en Yahoo Finance.")
        else:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                st.warning("Usando 'Close' en lugar de 'Adj Close'.")
                prices = data['Close']
            else:
                st.error("No se encontraron precios de cierre en los datos de Yahoo.")
                return None

            if isinstance(prices, pd.Series):
                prices = prices.to_frame()

            # Asegurar que las columnas coincidan con tickers
            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(0)

            # Verificar si todos los tickers están presentes
            missing_tickers = set(tickers) - set(prices.columns)
            if missing_tickers:
                st.info(f"Algunos tickers no están en Yahoo Finance: {missing_tickers}. Intentando con IOL...")

            # Intentar rellenar los faltantes con IOL
            for ticker in missing_tickers:
                price = get_price_from_iol(ticker)
                if price is not None:
                    # Simular una serie constante para el ticker faltante
                    prices[ticker] = price
                    st.info(f"Usando precio constante de IOL para {ticker}: {price}")
                else:
                    st.error(f"No se encontró precio para {ticker} ni en Yahoo ni en IOL.")
                    return None

            return prices
    except Exception as e:
        st.warning(f"Error con Yahoo Finance: {str(e)}. Intentando con datos IOL...")

    # Si Yahoo falla completamente, usar datos scrapeados
    scraped_prices = {}
    for ticker in tickers:
        price = get_price_from_iol(ticker)
        if price:
            scraped_prices[ticker] = price
        else:
            st.error(f"No se encontró precio para {ticker} en IOL.")
            return None

    # Crear un DataFrame simulado con el precio actual para cada ticker
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame(index=date_range)
    for ticker, price in scraped_prices.items():
        df[ticker] = price  # Simula que el precio fue constante en el periodo
    st.info("Usando datos scrapeados de IOL como precios constantes (sin historial real).")
    return df

def calculate_portfolio_performance(prices, weights):
    returns = prices.pct_change().dropna()
    # Asegurar que los pesos coincidan con las columnas de precios
    weights_series = pd.Series(weights, index=prices.columns)
    portfolio_return = (returns * weights_series).sum(axis=1)
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
        prices = fetch_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
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

def optimize_portfolio(returns_df, method='min_vol'):
    """
    Optimiza un portafolio basado en rendimientos históricos.
    - method: 'min_vol' o 'max_sharpe'
    """
    if returns_df.isnull().any().any():
        st.error("Datos de rendimientos contienen valores nulos. No se puede optimizar.")
        return None, None

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    n = len(mean_returns)
    init_weights = np.array([1 / n] * n)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))

    if method == 'min_vol':
        objective = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    elif method == 'max_sharpe':
        risk_free_rate = 0.02 / 252  # Asumiendo 2% anual, diario
        def sharpe_ratio(w):
            portfolio_return = np.dot(w.T, mean_returns)
            portfolio_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(portfolio_return - risk_free_rate) / portfolio_vol # Minimize -Sharpe
        objective = sharpe_ratio

    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        st.error(f"Optimización fallida: {result.message}")
        return None, None

    return result.x, result

def page_optimize_portfolio():
    st.header("⚖️ Optimización de Portafolio")
    st.markdown("Selecciona activos de IOL para optimizar pesos.")

    scraped_data = st.session_state.get('iol_data', {})
    symbols_fondos = [item['fondo'] for item in scraped_data.get('fondos', {}).get('datos', [])]
    symbols_bonos = [item['simbolo'] for item in scraped_data.get('bonos', {}).get('datos', [])]
    all_symbols = symbols_fondos + symbols_bonos

    selected_symbols = st.multiselect("Selecciona activos", all_symbols, default=all_symbols[:3])
    if not selected_symbols:
        st.warning("Selecciona al menos un activo.")
        return

    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime("today"))

    method = st.radio("Método de optimización", ["min_vol", "max_sharpe"])

    if st.button("Optimizar"):
        prices = fetch_prices_for_portfolio(selected_symbols, start_date, end_date)
        if prices is not None:
            returns = prices.pct_change().dropna()
            if returns.empty or len(returns.columns) < 2:
                st.error("No hay suficientes datos de rendimientos para los activos seleccionados.")
                return

            weights, result = optimize_portfolio(returns, method=method)
            if weights is not None:
                st.success("✅ Optimización exitosa.")
                df_weights = pd.DataFrame({'Activo': selected_symbols, 'Peso': weights.round(4)})
                st.table(df_weights)
                # Opcional: Mostrar info del resultado
                # st.json(result)

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
    "Chat de Análisis Cualitativo",
    "Optimización de Portafolios" # Nueva opción
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
elif st.session_state.selected_page == "Optimización de Portafolios":
    page_optimize_portfolio()
else:
    main_page()
