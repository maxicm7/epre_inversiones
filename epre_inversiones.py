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
import plotly.express as px
import io

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
    url = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"  # ✅ Sin espacio
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
        if "error" in data:  # ✅ CORREGIDO
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Monedas]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[1]:
        with st.spinner("Cargando fondos..."):
            data = scrape_iol_fondos()
        if "error" in data: # ✅ CORREGIDO
            st.error(data["error"])
        else:
            df = pd.DataFrame(data["datos"])
            st.dataframe(df, use_container_width=True)
            st.caption(f"Fuente: [IOL Fondos]({data['fuente']}) | Actualizado: {data['actualizado']}")

    with tabs[2]:
        with st.spinner("Cargando bonos..."):
            data = scrape_iol_bonos()
        if "error" in data: # ✅ CORREGIDO
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
        # Primero intentar con Yahoo Finance
        data = yf.download(tickers, start=start_date, end=end_date)
        
        if not data.empty:
            # Intentar usar 'Adj Close', si no existe, usar 'Close'
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            elif 'Close' in data.columns:
                st.warning("Usando 'Close' en lugar de 'Adj Close' (sin ajustes por dividendos/splits).")
                prices = data['Close']
            else:
                st.error("No se encontró ni 'Adj Close' ni 'Close' en los datos descargados de Yahoo.")
                return None
            
            # Si hay múltiples tickers, asegurar que sea un DataFrame
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            
            return prices

        # Si Yahoo no devuelve datos, intentar con IOL
        st.info("No se encontraron datos en Yahoo Finance. Intentando con datos scrapeados de IOL...")
        
        # Obtener datos de bonos y fondos desde IOL
        bonos_data = scrape_iol_bonos()
        fondos_data = scrape_iol_fondos()
        
        # Crear un DataFrame con precios actuales de IOL
        iol_prices = {}
        for ticker in tickers:
            # Buscar en bonos
            for row in bonos_data.get("datos", []):
                if row["simbolo"] == ticker:
                    iol_prices[ticker] = row["ultimo"]
                    break
            # Si no está en bonos, buscar en fondos
            if ticker not in iol_prices:
                for row in fondos_data.get("datos", []):
                    if row["fondo"] == ticker:
                        iol_prices[ticker] = row["ultimo"]
                        break
        
        if not iol_prices:
            st.error("No se encontraron datos para los tickers en IOL.")
            return None
        
        # Crear un DataFrame con un solo valor (precio actual) para cada ticker
        df = pd.DataFrame([iol_prices], index=[pd.to_datetime(end_date)])
        
        st.warning("⚠️ Los datos son de IOL (precios actuales). No hay serie temporal histórica.")
        return df

    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

def calculate_portfolio_performance(prices, weights):
    returns = prices.pct_change().dropna()
    portfolio_return = (returns * weights).sum(axis=1)
    cumulative_return = (1 + portfolio_return).cumprod()
    return cumulative_return

def optimize_portfolio(prices, risk_free_rate=0.0):
    """
    Optimiza los pesos de la cartera usando la frontera eficiente.
    """
    returns = prices.pct_change().dropna()
    
    if returns.empty:
        st.error("No hay datos de rendimientos para optimizar.")
        return None
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    
    # Función objetivo: minimizar la varianza (riesgo)
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Restricción: suma de pesos = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Peso inicial
    init_guess = np.array([1/num_assets] * num_assets)
    
    # Optimizar
    result = minimize(portfolio_variance, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        expected_return = np.sum(mean_returns * optimal_weights)
        expected_volatility = np.sqrt(portfolio_variance(optimal_weights))
        sharpe_ratio = (expected_return - risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        return {
            "weights": optimal_weights,
            "expected_return": expected_return,
            "volatility": expected_volatility,
            "sharpe_ratio": sharpe_ratio,
            "tickers": list(prices.columns)
        }
    else:
        st.error("No se pudo optimizar la cartera.")
        return None

def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    portfolio_name = st.text_input("Nombre del portafolio")
    tickers_input = st.text_area("Tickers (separados por comas)", "AAPL, MSFT")
    weights_input = st.text_area("Pesos (deben sumar 1.0)", "0.5, 0.5")
    
    if tickers_input and weights_input:
        tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        try:
            weights_list = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
        except ValueError:
            st.error("Los pesos deben ser números.")
            return

        if len(tickers_list) != len(weights_list):
            st.error("Número de tickers y pesos debe coincidir.")
        elif abs(sum(weights_list) - 1.0) > 1e-6:
            st.error("Los pesos deben sumar 1.0")
        else:
            # Crear DataFrame para visualización
            df_data = []
            # Intentar obtener precios de IOL
            bonos_data = scrape_iol_bonos()
            fondos_data = scrape_iol_fondos()
            monedas_data = scrape_iol_monedas() # Opcional, si se quieren precios de monedas

            for ticker, weight in zip(tickers_list, weights_list):
                precio_iol = None
                variacion_iol = None
                tipo = "Desconocido"

                # Buscar en bonos
                for row in bonos_data.get("datos", []):
                    if row["simbolo"] == ticker:
                        precio_iol = row["ultimo"]
                        variacion_iol = row["variacion"]
                        tipo = "Bonos"
                        break
                # Si no está en bonos, buscar en fondos
                if precio_iol is None:
                    for row in fondos_data.get("datos", []):
                        if row["fondo"] == ticker:
                            precio_iol = row["ultimo"]
                            variacion_iol = row["variacion"]
                            tipo = "Fondos"
                            break
                # Si no está en fondos, buscar en monedas
                if precio_iol is None:
                    for row in monedas_data.get("datos", []):
                        if row["moneda"] == ticker:
                            precio_iol = row["venta"] # O 'compra', dependiendo del interés
                            variacion_iol = row["variacion"]
                            tipo = "Monedas"
                            break

                df_data.append({
                    "Ticker": ticker,
                    "Peso": weight,
                    "Precio_IOL": precio_iol,
                    "Variacion_IOL": variacion_iol,
                    "Tipo": tipo
                })

            df = pd.DataFrame(df_data)

            st.subheader("Visualización de Activos Seleccionados")
            st.dataframe(df, use_container_width=True)

            # Botón para descargar el DataFrame
            csv = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Descargar tabla de activos (CSV)",
                data=csv,
                file_name=f'portfolio_{portfolio_name}_activos.csv',
                mime='text/csv',
            )

            # Guardar portafolio si se da nombre
            if portfolio_name:
                portfolios = st.session_state.get("portfolios", {})
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

def page_optimize_portfolio():
    st.header("📊 Optimización de Cartera (Markowitz)")
    st.markdown("Calcula los pesos óptimos para maximizar el ratio Sharpe.")

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("No hay portafolios guardados.")
        return

    name = st.selectbox("Selecciona un portafolio", list(portfolios.keys()))
    portfolio = portfolios[name]
    st.json(portfolio)

    start_date = st.date_input("Fecha de inicio", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Fecha de fin", value=pd.to_datetime("today"))
    
    if st.button("Optimizar Cartera"):
        prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None and len(prices) > 1:
            with st.spinner("Optimizando..."):
                result = optimize_portfolio(prices)
                if result:
                    st.success("✅ Cartera optimizada!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Retorno Esperado", f"{result['expected_return']:.2%}")
                        st.metric("Volatilidad", f"{result['volatility']:.2%}")
                    with col2:
                        st.metric("Ratio Sharpe", f"{result['sharpe_ratio']:.2f}")
                    
                    # Mostrar pesos optimizados
                    weights_df = pd.DataFrame({
                        "Ticker": result["tickers"],
                        "Peso Óptimo": result["weights"]
                    })
                    st.subheader("Pesos Óptimos")
                    st.dataframe(weights_df, use_container_width=True)
                    
                    # Gráfico de pastel
                    fig = px.pie(weights_df, values='Peso Óptimo', names='Ticker', title='Distribución de Pesos')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Necesitas al menos 2 días de datos para optimizar.")

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

    # Subir archivo CSV con activos
    uploaded_csv = st.file_uploader("Sube la tabla de activos (CSV)", type=['csv'])
    df_from_csv = None
    if uploaded_csv is not None:
        try:
            df_from_csv = pd.read_csv(uploaded_csv, sep=';') # Ajustar separador si es necesario
            st.info("Tabla de activos cargada.")
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    # Subir archivo PDF con contexto
    uploaded_pdf = st.file_uploader("Sube un PDF con contexto (opcional)", type=['pdf'])
    pdf_content = ""
    if uploaded_pdf is not None:
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
            pdf_content = ""
            for page in pdf_reader.pages:
                pdf_content += page.extract_text()
            st.info("PDF cargado y leído.")
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")

    # Inicializar historial de chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Escribe tu consulta de inversión..."):
        # Añadir contexto de la tabla CSV si existe
        csv_context = f"\nTabla de Activos:\n{df_from_csv.to_csv(index=False, sep=';')}\n" if df_from_csv is not None else ""
        # Añadir contexto del PDF si existe
        pdf_context = f"\nContenido del PDF:\n{pdf_content}\n" if pdf_content else ""

        full_prompt = f"{csv_context}{pdf_context}{prompt}"

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Pensando..."):
            response = get_hf_response(
                full_prompt,
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
    "Create/Edit Portfolios", # Aquí se agregó la visualización
    "View Portfolio Returns",
    "Optimize Portfolio",
    "Datos en Vivo – IOL",
    "Chat de Análisis Cualitativo" # Ahora puede leer CSV y PDF
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
elif st.session_state.selected_page == "Optimize Portfolio":
    page_optimize_portfolio()
elif st.session_state.selected_page == "Datos en Vivo – IOL":
    page_datos_en_vivo_iol()
elif st.session_state.selected_page == "Chat de Análisis Cualitativo":
    page_investment_insights_chat()
else:
    main_page()
