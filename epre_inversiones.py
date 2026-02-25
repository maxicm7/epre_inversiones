import os
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

# ── Módulos propios ───────────────────────────────────────────────────────
from forecast_module import page_forecast
from iol_client import page_iol_explorer, get_iol_client

# ── Configuración ─────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Bonos, Fondos y Dólar")

PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error al cargar portafolios: {e}")
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
#  SCRAPING IOL (publico, sin auth)
# ═══════════════════════════════════════════════════════════════════════════

def scrape_table(url, min_cols, max_rows=None):
    try:
        headers  = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup  = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")
        if not table:
            return {"error": "No se encontro la tabla."}
        rows = table.find_all("tr")[1:]
        if max_rows:
            rows = rows[:max_rows]
        return {"rows": rows, "actualizado": time.strftime("%Y-%m-%d %H:%M")}
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=300)
def scrape_iol_monedas():
    url    = "https://iol.invertironline.com/mercado/cotizaciones/argentina/monedas"
    result = scrape_table(url, min_cols=5)
    if "error" in result:
        return result
    data = []
    for row in result["rows"]:
        cols = row.find_all("td")
        if len(cols) >= 5:
            compra = cols[1].get_text(strip=True).replace(".", "").replace(",", ".")
            venta  = cols[2].get_text(strip=True).replace(".", "").replace(",", ".")
            if compra != "-" and venta != "-":
                try:
                    float(compra); float(venta)
                    data.append({"moneda": cols[0].get_text(strip=True),
                                 "compra": compra, "venta": venta,
                                 "fecha": cols[3].get_text(strip=True),
                                 "variacion": cols[4].get_text(strip=True)})
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

@st.cache_data(ttl=600)
def scrape_iol_fondos():
    url    = "https://iol.invertironline.com/mercado/cotizaciones/argentina/fondos/todos"
    result = scrape_table(url, min_cols=9)
    if "error" in result:
        return result
    data = []
    for row in result["rows"][:20]:
        cols = row.find_all("td")
        if len(cols) >= 9:
            s = cols[3].get_text(strip=True).replace("AR$ ", "").replace("US$ ", "")
            if s and s != "-":
                try:
                    data.append({"fondo": cols[0].get_text(strip=True),
                                 "ultimo": float(s.replace(".", "").replace(",", ".")),
                                 "variacion": cols[4].get_text(strip=True)})
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

@st.cache_data(ttl=600)
def scrape_iol_bonos():
    url    = "https://iol.invertironline.com/mercado/cotizaciones/argentina/bonos/todos"
    result = scrape_table(url, min_cols=13)
    if "error" in result:
        return result
    data = []
    for row in result["rows"][:30]:
        cols = row.find_all("td")
        if len(cols) >= 13:
            s = cols[1].get_text(strip=True)
            if s and s != "-":
                try:
                    data.append({"simbolo": cols[0].get_text(strip=True).replace("\n","").strip(),
                                 "ultimo": float(s.replace(".", "").replace(",", ".")),
                                 "variacion": cols[2].get_text(strip=True)})
                except ValueError:
                    continue
    return {"fuente": url, "datos": data, "actualizado": result["actualizado"]}

# ═══════════════════════════════════════════════════════════════════════════
#  PAGINAS
# ═══════════════════════════════════════════════════════════════════════════

def main_page():
    st.title("BPNos - Bonos, Fondos y Dolar")
    st.markdown("""
    Bienvenido. Usa el menu lateral para navegar entre las secciones.

    | Seccion | Descripcion |
    |---|---|
    | 🏦 Explorador IOL API | API oficial: acciones, CEDEARs, bonos, FCI, serie historica |
    | 💼 Crear/Editar Portafolio | Arma tu cartera manualmente o desde IOL |
    | 📈 Rendimiento | Visualiza el retorno historico de tu portafolio |
    | 📊 Optimizacion | Markowitz: minima volatilidad, Sharpe, retorno maximo |
    | 🔭 Pronostico | SARIMAX / Prophet con variables exogenas + Gemini AI |
    | 📡 Datos en Vivo | Scraping publico de monedas, fondos y bonos |
    | 💬 Chat | Analisis cualitativo con Hugging Face |
    """)


def page_datos_en_vivo_iol():
    st.header("📡 Datos en Vivo - InvertirOnline (Scraping publico)")
    tabs = st.tabs(["💱 Monedas", "📊 Fondos", "🎫 Bonos"])

    with tabs[0]:
        with st.spinner("Cargando monedas..."):
            data = scrape_iol_monedas()
        if "error" in data:
            st.error(data["error"])
        else:
            st.dataframe(pd.DataFrame(data["datos"]), use_container_width=True)
            st.caption(f"Fuente: [IOL]({data['fuente']}) | {data['actualizado']}")

    with tabs[1]:
        with st.spinner("Cargando fondos..."):
            data = scrape_iol_fondos()
        if "error" in data:
            st.error(data["error"])
        else:
            st.dataframe(pd.DataFrame(data["datos"]), use_container_width=True)
            st.caption(f"Fuente: [IOL]({data['fuente']}) | {data['actualizado']}")

    with tabs[2]:
        with st.spinner("Cargando bonos..."):
            data = scrape_iol_bonos()
        if "error" in data:
            st.error(data["error"])
        else:
            st.dataframe(pd.DataFrame(data["datos"]), use_container_width=True)
            st.caption(f"Fuente: [IOL]({data['fuente']}) | {data['actualizado']}")


def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    """
    Obtiene precios historicos:
    1. IOL API (serie historica) si hay cliente autenticado
    2. Yahoo Finance como fallback
    """
    client     = get_iol_client()
    all_prices = {}
    yf_tickers = []

    for ticker in tickers:
        fetched = False
        if client:
            # Formateamos para estar seguros que no mande Yahoo suffix ".BA" a IOL
            simbolo_iol = ticker.split(".")[0].upper()
            
            # Las fechas de Streamlit vienen como objetos date, IOL necesita string YYYY-MM-DD
            fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            fmt_end   = pd.to_datetime(end_date).strftime("%Y-%m-%d")

            df_hist = client.get_serie_historica(simbolo_iol, fmt_start, fmt_end)
            if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                s = df_hist["ultimoPrecio"].rename(ticker)
                
                # Quitar zona horaria para compatibilidad total
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                
                all_prices[ticker] = s
                fetched = True
        if not fetched:
            yf_tickers.append(ticker)

    if yf_tickers:
        try:
            raw = yf.download(yf_tickers, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
            if not raw.empty:
                close = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=yf_tickers[0])
                
                # Quitar zona horaria de Yahoo
                if close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                    
                for col in close.columns:
                    all_prices[str(col)] = close[col]
        except Exception as e:
            st.warning(f"Yahoo Finance error: {e}")

    if not all_prices:
        st.error("No se pudieron obtener precios.")
        return None

    prices = pd.concat(all_prices.values(), axis=1)
    prices.columns = list(all_prices.keys())
    prices.dropna(how="all", inplace=True)
    prices.ffill(inplace=True) # Rellenar para homogeneizar series
    return prices


def calculate_portfolio_performance(prices, weights):
    returns = prices.pct_change().dropna()
    return (1 + (returns * weights).sum(axis=1)).cumprod()


def optimize_portfolio(prices, risk_free_rate=0.0, opt_type="Minima Volatilidad"):
    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("No hay datos de rendimientos.")
        return None
    mean_returns = returns.mean()
    cov_matrix   = returns.cov()
    n            = len(mean_returns)
    constraints  = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds       = tuple((0, 1) for _ in range(n))
    init         = np.array([1/n] * n)

    if "Volatilidad" in opt_type:
        obj = lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    elif "Retorno" in opt_type:
        obj = lambda w: -np.sum(mean_returns * w)
    else:
        def obj(w):
            r = np.sum(mean_returns * w)
            v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(r - risk_free_rate) / v if v > 0 else np.inf

    res = minimize(obj, init, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        st.error(f"Optimizacion fallida: {res.message}")
        return None
    ow  = res.x
    er  = np.sum(mean_returns * ow)
    ev  = np.sqrt(np.dot(ow.T, np.dot(cov_matrix, ow)))
    out = {"weights": ow, "expected_return": er, "volatility": ev, "tickers": list(prices.columns)}
    if "Sharpe" in opt_type:
        out["sharpe_ratio"] = (er - risk_free_rate) / ev if ev > 0 else 0
    return out


def page_create_portfolio():
    st.header("💼 Crear / Editar Portafolio")
    st.info("💡 Tambien podes agregar activos desde **🏦 Explorador IOL API** → pestana ➕ Agregar a Portafolio")

    portfolio_name = st.text_input("Nombre del portafolio")
    tickers_input  = st.text_area("Tickers (separados por comas)", "AAPL, MSFT")
    weights_input  = st.text_area("Pesos (deben sumar 1.0)", "0.5, 0.5")

    if tickers_input and weights_input:
        tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        try:
            weights_list = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
        except ValueError:
            st.error("Los pesos deben ser numeros.")
            return
        if len(tickers_list) != len(weights_list):
            st.error("Numero de tickers y pesos debe coincidir.")
            return
        if abs(sum(weights_list) - 1.0) > 1e-6:
            st.error("Los pesos deben sumar 1.0")
            return

        client       = get_iol_client()
        bonos_data   = scrape_iol_bonos()
        fondos_data  = scrape_iol_fondos()
        monedas_data = scrape_iol_monedas()
        df_data      = []

        for ticker, weight in zip(tickers_list, weights_list):
            precio = None; variacion = None; fuente = "—"; tipo = "Desconocido"

            if client:
                cot = client.get_cotizacion("bCBA", ticker)
                if cot and isinstance(cot, dict):
                    precio    = cot.get("ultimoPrecio") or cot.get("ultimo")
                    variacion = cot.get("variacion")
                    fuente    = "IOL API"; tipo = cot.get("tipo", "—")

            if precio is None:
                for row in bonos_data.get("datos", []):
                    if row["simbolo"] == ticker:
                        precio = row["ultimo"]; variacion = row["variacion"]
                        fuente = "scraping"; tipo = "Bonos"; break
            if precio is None:
                for row in fondos_data.get("datos", []):
                    if row["fondo"] == ticker:
                        precio = row["ultimo"]; variacion = row["variacion"]
                        fuente = "scraping"; tipo = "FCI"; break
            if precio is None:
                for row in monedas_data.get("datos", []):
                    if row["moneda"] == ticker:
                        precio = row["venta"]; variacion = row["variacion"]
                        fuente = "scraping"; tipo = "Moneda"; break

            df_data.append({"Ticker": ticker, "Peso": weight, "Precio": precio,
                            "Variacion": variacion, "Tipo": tipo, "Fuente": fuente})

        df = pd.DataFrame(df_data)
        st.subheader("Activos del portafolio")
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False, sep=';').encode('utf-8')
        st.download_button("📥 Descargar CSV", csv,
                           file_name=f'portfolio_{portfolio_name}.csv', mime='text/csv')

        if portfolio_name:
            portfolios = st.session_state.get("portfolios", {})
            portfolios[portfolio_name] = {"tickers": tickers_list, "weights": weights_list}
            ok, msg = save_portfolios_to_file(portfolios)
            if ok:
                st.session_state.portfolios = portfolios
                st.success("✅ Portafolio guardado.")
            else:
                st.error(f"❌ Error: {msg}")


def page_view_portfolio_returns():
    st.header("📈 Rendimiento de Portafolio")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("No hay portafolios guardados.")
        return
    name      = st.selectbox("Portafolio", list(portfolios.keys()))
    portfolio = portfolios[name]
    st.json(portfolio)
    start_date = st.date_input("Desde", value=pd.to_datetime("2023-01-01"))
    end_date   = st.date_input("Hasta", value=pd.to_datetime("today"))
    if st.button("Calcular Rendimiento"):
        with st.spinner("Obteniendo precios..."):
            prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None:
            st.line_chart(calculate_portfolio_performance(prices, portfolio["weights"]))


def page_optimize_portfolio():
    st.header("📊 Optimizacion de Cartera (Markowitz)")
    portfolios = st.session_state.get("portfolios", {})
    if not portfolios:
        st.info("No hay portafolios guardados.")
        return
    name      = st.selectbox("Portafolio", list(portfolios.keys()))
    portfolio = portfolios[name]
    st.json(portfolio)
    start_date = st.date_input("Desde", value=pd.to_datetime("2023-01-01"))
    end_date   = st.date_input("Hasta", value=pd.to_datetime("today"))
    opt_type   = st.selectbox("Objetivo",
                              ["Minima Volatilidad", "Maximo Ratio Sharpe", "Retorno Maximo"])
    if st.button("Optimizar"):
        with st.spinner("Optimizando..."):
            prices = fetch_stock_prices_for_portfolio(portfolio["tickers"], start_date, end_date)
        if prices is not None and len(prices) > 1:
            result = optimize_portfolio(prices, opt_type=opt_type)
            if result:
                st.success(f"✅ Optimizado ({opt_type})")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Retorno Esperado", f"{result['expected_return']:.2%}")
                    st.metric("Volatilidad",       f"{result['volatility']:.2%}")
                with c2:
                    st.metric("Ratio Sharpe",
                              f"{result['sharpe_ratio']:.2f}" if 'sharpe_ratio' in result else "N/A")
                wdf = pd.DataFrame({"Ticker": result["tickers"], "Peso Optimo": result["weights"]})
                st.dataframe(wdf, use_container_width=True, hide_index=True)
                fig = px.pie(wdf, values='Peso Optimo', names='Ticker', title='Distribucion')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Necesitas al menos 2 dias de datos.")


def get_hf_response(prompt, api_key, model, temp=0.5):
    try:
        client   = InferenceClient(api_key=api_key)
        response = client.chat_completion(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=500, temperature=temp)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error HF: {str(e)}"


def page_investment_insights_chat():
    st.header("💬 Chat de Analisis Cualitativo")
    if not st.session_state.get('hf_api_key'):
        st.warning("Ingresa tu Hugging Face API Key en la barra lateral.")
        return
    uploaded_csv = st.file_uploader("Tabla de activos (CSV)", type=['csv'])
    df_from_csv  = None
    if uploaded_csv:
        try:
            df_from_csv = pd.read_csv(uploaded_csv, sep=';'); st.info("CSV cargado.")
        except Exception as e:
            st.error(f"Error CSV: {e}")
    uploaded_pdf = st.file_uploader("PDF de contexto (opcional)", type=['pdf'])
    pdf_content  = ""
    if uploaded_pdf:
        try:
            import PyPDF2
            r = PyPDF2.PdfReader(uploaded_pdf)
            pdf_content = "".join(p.extract_text() for p in r.pages)
            st.info("PDF cargado.")
        except Exception as e:
            st.error(f"Error PDF: {e}")
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    for msg in st.session_state.chat_messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Consulta de inversion..."):
        csv_ctx = f"\nTabla:\n{df_from_csv.to_csv(index=False,sep=';')}\n" if df_from_csv is not None else ""
        pdf_ctx = f"\nPDF:\n{pdf_content}\n" if pdf_content else ""
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Pensando..."):
            resp = get_hf_response(f"{csv_ctx}{pdf_ctx}{prompt}",
                                   st.session_state.hf_api_key,
                                   st.session_state.hf_model,
                                   st.session_state.hf_temp)
        st.session_state.chat_messages.append({"role": "assistant", "content": resp})
        st.chat_message("assistant").write(resp)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════
defaults = {
    'selected_page':  "Welcome Page",
    'hf_api_key':     "",
    'hf_model':       "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'hf_temp':        0.5,
    'gemini_api_key': "",
    'gemini_model':   "gemini-2.5-flash",
    'iol_username':   "",
    'iol_password':   "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = load_portfolios_from_file()


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
st.sidebar.title("Configuracion")

# IOL API
with st.sidebar.expander("🏦 IOL API", expanded=True):
    st.markdown("Credenciales de [InvertirOnline](https://www.invertironline.com)")
    iol_user = st.text_input("Usuario / Email",
                              value=st.session_state.get("iol_username",""), key="iol_u")
    iol_pass = st.text_input("Contrasena", type="password",
                              value=st.session_state.get("iol_password",""), key="iol_p")
    if iol_user: st.session_state.iol_username = iol_user
    if iol_pass: st.session_state.iol_password = iol_pass

    if st.button("🔐 Conectar", key="btn_iol"):
        with st.spinner("Autenticando..."):
            c = get_iol_client()
        if c: st.success("✅ Conectado")
        else: st.error("❌ Error de autenticacion")

    if st.session_state.get("iol_client"):
        st.success("✅ IOL activo")
    elif st.session_state.get("iol_username"):
        st.warning("⚠️ Sin conectar aun")
    else:
        st.info("Sin credenciales")

# Gemini
with st.sidebar.expander("🔮 Google Gemini", expanded=False):
    st.markdown("Key en [aistudio.google.com](https://aistudio.google.com/app/apikey)")
    gk = st.text_input("Gemini API Key", type="password",
                        value=st.session_state.get('gemini_api_key',''), key="gk")
    if gk: st.session_state.gemini_api_key = gk
    st.session_state.gemini_model = st.selectbox(
        "Modelo", ["gemini-2.5-flash","gemini-3"], key="gm")
    if st.session_state.get("gemini_api_key"): st.success("✅ Key OK")

# Hugging Face
with st.sidebar.expander("🤗 Hugging Face", expanded=False):
    hk = st.text_input("HF API Key", type="password",
                        value=st.session_state.get('hf_api_key',''), key="hk")
    if hk: st.session_state.hf_api_key = hk
    st.session_state.hf_model = st.selectbox("Modelo HF", [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-7b-it"], key="hm")
    st.session_state.hf_temp = st.slider("Temperatura", 0.1, 1.0, 0.5, 0.1, key="ht")

# Navegacion
st.sidebar.markdown("---")
st.sidebar.title("Navegacion")

page_options = [
    "Welcome Page",
    "🏦 Explorador IOL API",
    "💼 Crear/Editar Portafolio",
    "📈 Rendimiento de Portafolio",
    "📊 Optimizacion de Cartera",
    "🔭 Pronostico con Exogenas",
    "📡 Datos en Vivo - IOL",
    "💬 Chat de Analisis",
]

if st.session_state.selected_page not in page_options:
    st.session_state.selected_page = "Welcome Page"

page = st.sidebar.radio("Seccion", page_options,
                        index=page_options.index(st.session_state.selected_page))
if page != st.session_state.selected_page:
    st.session_state.selected_page = page
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════════════════
sel = st.session_state.selected_page

if   sel == "Welcome Page":                  main_page()
elif sel == "🏦 Explorador IOL API":         page_iol_explorer()
elif sel == "💼 Crear/Editar Portafolio":    page_create_portfolio()
elif sel == "📈 Rendimiento de Portafolio":  page_view_portfolio_returns()
elif sel == "📊 Optimizacion de Cartera":    page_optimize_portfolio()
elif sel == "🔭 Pronostico con Exogenas":    page_forecast()
elif sel == "📡 Datos en Vivo - IOL":        page_datos_en_vivo_iol()
elif sel == "💬 Chat de Analisis":           page_investment_insights_chat()
else:                                         main_page()
