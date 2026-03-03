import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import traceback
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf

# ── IMPORTACIÓN SEGURA DE GOOGLE GEMINI ──
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# ── IMPORTACIÓN SEGURA DE OPENAI (Copilot engine) ──
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# ── Dependencia opcional para optimización institucional ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios (Manejo de errores si no existen) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    def page_forecast(): st.warning("Módulo forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas", page_icon="📈")

PORTFOLIO_FILE = "portfolios_data1.json"

# ═══════════════════════════════════════════════════════════════════════════
#  GESTIÓN DE DATOS Y PORTAFOLIOS
# ═══════════════════════════════════════════════════════════════════════════

def load_portfolios_from_file():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error de lectura JSON: {e}")
    return {}

def save_portfolios_to_file(portfolios_dict):
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        return False, str(e)

# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA Y OPTIMIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    client = get_iol_client()
    all_prices = {}
    yf_tickers = []

    for ticker in tickers:
        fetched = False
        if client:
            simbolo_iol = ticker.split(".")[0].upper()
            fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            fmt_end   = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            try:
                df_hist = client.get_serie_historica(simbolo_iol, fmt_start, fmt_end)
                if not df_hist.empty and "ultimoPrecio" in df_hist.columns:
                    s = df_hist["ultimoPrecio"].rename(ticker)
                    if s.index.tz is not None: s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except:
                pass 
        if not fetched:
            yf_tickers.append(ticker)

    if yf_tickers:
        try:
            adjusted_tickers = [t if "." in t or t.endswith("=X") else t+".BA" for t in yf_tickers]
            raw = yf.download(adjusted_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if not raw.empty:
                close_data = raw["Close"] if "Close" in raw.columns else raw
                if isinstance(close_data, pd.Series):
                    close_data = close_data.to_frame(name=yf_tickers[0])
                
                if len(adjusted_tickers) == 1:
                     all_prices[yf_tickers[0]] = close_data.iloc[:, 0]
                else:
                    for col in close_data.columns:
                        clean_col = str(col).replace(".BA", "")
                        for original in yf_tickers:
                            if clean_col == original or str(col) == original:
                                all_prices[original] = close_data[col]
                                break
        except Exception as e:
            st.warning(f"Yahoo Finance warning: {e}")

    if not all_prices: return None
    prices = pd.concat(all_prices.values(), axis=1)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None: prices.index = prices.index.tz_localize(None)
    prices.sort_index(inplace=True)
    prices.ffill(inplace=True)
    prices.dropna(inplace=True)
    return prices

def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    returns = prices.pct_change().dropna()
    if returns.empty: return None

    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            
            if opt_type == "Maximo Ratio Sharpe": ef.max_sharpe(risk_free_rate=risk_free_rate)
            elif opt_type == "Minima Volatilidad": ef.min_volatility()
            else: ef.max_quadratic_utility(risk_aversion=0.0001)
            
            weights = ef.clean_weights()
            ret, vol, sharpe = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
            ow_array = np.array([weights.get(col, 0) for col in prices.columns])
            
            return {"weights": ow_array, "expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe, "tickers": list(prices.columns), "returns": returns, "method": "PyPortfolioOpt"}
        except Exception:
            pass

    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252
    n = len(mean_returns)
    
    def get_metrics(w):
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 0 else 0
        return np.array([ret, vol, sr])

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad": fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo": fun = lambda w: -get_metrics(w)[0]
    else: fun = lambda w: -get_metrics(w)[2]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    final_metrics = get_metrics(res.x) if res.success else [0,0,0]
    
    return {"weights": res.x, "expected_return": final_metrics[0], "volatility": final_metrics[1], "sharpe_ratio": final_metrics[2], "tickers": list(prices.columns), "returns": returns, "method": "Scipy/SLSQP"}

# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE RENTA FIJA (BONOS)
# ═══════════════════════════════════════════════════════════════════════════
def calc_bond_metrics(face_value, coupon_rate, ytm, years_to_maturity, freq=2):
    """Calcula Precio, Duración Macaulay, Modificada y Convexidad"""
    periods = int(years_to_maturity * freq)
    coupon = (coupon_rate / freq) * face_value
    rate = ytm / freq
    
    price = 0
    mac_dur_num = 0
    conv_num = 0
    
    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / ((1 + rate)**t)
        price += pv
        mac_dur_num += (t / freq) * pv
        conv_num += (t / freq) * ((t / freq) + (1/freq)) * cf / ((1 + rate)**(t + 2))
        
    mac_dur = mac_dur_num / price if price > 0 else 0
    mod_dur = mac_dur / (1 + rate)
    convexity = conv_num / price if price > 0 else 0
    return price, mac_dur, mod_dur, convexity

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS DE LA APLICACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    tabs = st.tabs(["💼 Mis Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])
    
    # --- TAB 1: GESTIÓN ---
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Crear Cartera")
            p_name = st.text_input("Nombre Cartera")
            p_tickers = st.text_area("Tickers (ej: AL30, GGAL)", height=100).upper()
            p_weights = st.text_area("Pesos (ej: 0.5, 0.5)", height=100)
            if st.button("Guardar", type="primary"):
                try:
                    t = [x.strip() for x in p_tickers.split(",") if x.strip()]
                    w = [float(x) for x in p_weights.split(",") if x.strip()]
                    if len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                        st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.success("Guardado.")
                    else: st.error("Error en pesos o cantidad.")
                except: st.error("Error de formato.")
        with c2:
            if st.session_state.portfolios:
                st.dataframe(pd.DataFrame(st.session_state.portfolios).T, use_container_width=True)

    portfolios = st.session_state.get("portfolios", {})
    if not portfolios: return

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    p_sel = col1.selectbox("Analizar Cartera:", list(portfolios.keys()))
    d_start = col2.date_input("Desde", pd.to_datetime("2023-01-01"))
    d_end = col3.date_input("Hasta", pd.to_datetime("today"))

    # --- TAB 2: OPTIMIZACIÓN ---
    with tabs[1]:
        st.subheader(f"Frontera Eficiente: {p_sel}")
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            risk_free = st.number_input("Tasa Libre Riesgo (RF)", 0.0, 0.5, 0.04, step=0.01)
        with c_opt2:
            target = st.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
            
        if st.button("Ejecutar Optimización"):
            with st.spinner("Optimizando..."):
                prices = fetch_stock_prices_for_portfolio(portfolios[p_sel]["tickers"], d_start, d_end)
            if prices is not None:
                res = optimize_portfolio_corporate(prices, risk_free_rate=risk_free, opt_type=target)
                if res:
                    st.session_state['last_opt_res'] = res
                    st.session_state['last_opt_target'] = target
                    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
                    c_kpi1.metric("Retorno Esperado", f"{res['expected_return']:.1%}")
                    c_kpi2.metric("Volatilidad Anual", f"{res['volatility']:.1%}")
                    c_kpi3.metric("Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")

                    w_df = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']})
                    fig = px.pie(w_df[w_df["Peso"]>0.001], values="Peso", names="Activo", title="Asignación", hole=0.4, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.error("Error al optimizar.")
            else: st.error("Error en datos.")

        # Integración de Análisis IA (Copilot/OpenAI)
        if 'last_opt_res' in st.session_state:
            st.markdown("---")
            if st.button("🧠 Analizar Portafolio con IA (Copilot)"):
                if OPENAI_OK and st.session_state.get('openai_api_key'):
                    with st.spinner("Consultando a la IA..."):
                        try:
                            client = OpenAI(api_key=st.session_state.openai_api_key)
                            data_str = f"Retorno: {st.session_state['last_opt_res']['expected_return']:.2f}, Volatilidad: {st.session_state['last_opt_res']['volatility']:.2f}. Activos: {st.session_state['last_opt_res']['tickers']}."
                            prompt = f"Actúa como un asesor financiero institucional. Analiza este portafolio: {data_str}. ¿Cuáles son los riesgos y fortalezas de esta distribución?"
                            response = client.chat.completions.create(
                                model=st.session_state.get('openai_model', 'gpt-3.5-turbo'),
                                messages=[{"role": "user", "content": prompt}]
                            )
                            st.info(response.choices[0].message.content)
                        except Exception as e: st.error(f"Error con OpenAI API: {e}")
                else:
                    st.warning("⚠️ Ingresa tu API Key de OpenAI (Copilot) en el menú lateral.")

    # --- TAB 3: FORECAST ---
    with tabs[2]:
        if 'last_opt_res' in st.session_state:
            res = st.session_state['last_opt_res']
            st.subheader("Simulación Montecarlo")
            c_sim1, c_sim2 = st.columns(2)
            days = c_sim1.slider("Días Proyección", 30, 365, 90)
            n_sims = c_sim2.selectbox("Cantidad Simulaciones", [100, 500, 1000], index=1)
            
            if st.button("Simular Escenarios Futuros"):
                dt = 1/252
                mu = res['expected_return'] * dt
                sigma = res['volatility'] * np.sqrt(dt)
                paths = np.zeros((days, n_sims))
                paths[0] = 100 
                for t in range(1, days):
                    rand = np.random.standard_normal(n_sims)
                    paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * rand)
                
                fig = go.Figure()
                p95, p05 = np.percentile(paths, 95, axis=1), np.percentile(paths, 5, axis=1)
                x_axis = np.arange(days)
                fig.add_trace(go.Scatter(x=np.concatenate([x_axis, x_axis[::-1]]), y=np.concatenate([p95, p05[::-1]]), fill='toself', fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
                fig.add_trace(go.Scatter(x=x_axis, y=np.mean(paths, axis=1), mode='lines', name='Media', line=dict(color='#00CC96')))
                fig.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Pesimista (5%)', line=dict(color='#EF553B', dash='dash')))
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  NUEVO MÓDULO: RENTA FIJA Y BONOS
# ═══════════════════════════════════════════════════════════════════════════
def page_fixed_income():
    st.title("🏛️ Renta Fija: Curva, Convexidad e Inmunización")
    
    st.markdown("""Ingresa los datos de los bonos para generar la curva de rendimiento y analizar el portafolio frente a cambios de tasas.""")
    
    # Datos por defecto
    if 'bonds_data' not in st.session_state:
        st.session_state.bonds_data = pd.DataFrame({
            "Bono": ["Bono Corto", "Bono Medio", "Bono Largo"],
            "Cupón (%)": [3.0, 4.5, 6.0],
            "YTM (%)": [4.0, 5.0, 6.5],
            "Años a Venc.": [2.0, 5.0, 10.0],
            "Nominal Invertido": [100000, 150000, 50000]
        })

    # Editor interactivo de bonos
    edited_bonds = st.data_editor(st.session_state.bonds_data, num_rows="dynamic", use_container_width=True)
    st.session_state.bonds_data = edited_bonds

    tabs = st.tabs(["📊 Análisis e Inmunización", "📈 Curva de Rendimiento", "💬 Chat IA (Especialista en Bonos)"])

    # Cálculos en segundo plano
    results = []
    total_investment = 0
    port_mac_dur = 0
    port_mod_dur = 0
    port_convexity = 0
    
    for _, row in edited_bonds.iterrows():
        try:
            p, macd, modd, conv = calc_bond_metrics(
                face_value=100, 
                coupon_rate=row["Cupón (%)"]/100, 
                ytm=row["YTM (%)"]/100, 
                years_to_maturity=row["Años a Venc."]
            )
            weight = row["Nominal Invertido"]
            total_investment += weight
            results.append({
                "Bono": row["Bono"], "Precio Calc.": p, "Mac. Dur": macd, "Mod. Dur": modd, "Convexidad": conv, "Peso": weight
            })
        except: pass

    if total_investment > 0 and results:
        df_res = pd.DataFrame(results)
        df_res["Peso %"] = df_res["Peso"] / total_investment
        port_mac_dur = (df_res["Mac. Dur"] * df_res["Peso %"]).sum()
        port_mod_dur = (df_res["Mod. Dur"] * df_res["Peso %"]).sum()
        port_convexity = (df_res["Convexidad"] * df_res["Peso %"]).sum()

    with tabs[0]:
        st.subheader("Métricas del Portafolio de Renta Fija")
        if total_investment > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Duración Macaulay (Años)", f"{port_mac_dur:.2f}")
            c2.metric("Duración Modificada", f"{port_mod_dur:.2f}")
            c3.metric("Convexidad", f"{port_convexity:.2f}")
            
            st.dataframe(df_res[["Bono", "Precio Calc.", "Mac. Dur", "Mod. Dur", "Convexidad"]].style.format("{:.2f}", subset=["Precio Calc.", "Mac. Dur", "Mod. Dur", "Convexidad"]))

            st.markdown("---")
            st.subheader("🛡️ Inmunización de Portafolio")
            horizonte = st.slider("Tu Horizonte de Inversión Objetivo (Años)", 0.5, 30.0, 5.0, 0.5)
            
            gap = port_mac_dur - horizonte
            if abs(gap) < 0.2:
                st.success(f"✅ **Portafolio Inmunizado**. La Duración Macaulay ({port_mac_dur:.2f}) coincide con tu horizonte. El riesgo de precio y de reinversión se cancelan.")
            elif gap > 0:
                st.warning(f"⚠️ **Riesgo de Tasa (Precio)**. La Duración ({port_mac_dur:.2f}) es mayor a tu horizonte ({horizonte}). Eres vulnerable a una subida de tasas.")
            else:
                st.info(f"ℹ️ **Riesgo de Reinversión**. La Duración ({port_mac_dur:.2f}) es menor a tu horizonte ({horizonte}). Eres vulnerable a una bajada de tasas.")

    with tabs[1]:
        st.subheader("Curva de Rendimiento (Yield Curve)")
        if not edited_bonds.empty:
            df_curve = edited_bonds.sort_values(by="Años a Venc.")
            fig = px.line(df_curve, x="Años a Venc.", y="YTM (%)", markers=True, text="Bono", title="Curva de Tasas del Portafolio")
            fig.update_traces(textposition="top center")
            fig.update_layout(template="plotly_dark", yaxis_title="Yield to Maturity (%)", xaxis_title="Plazo (Años)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("💬 Asistente IA para Renta Fija (Copilot/OpenAI)")
        if not OPENAI_OK or not st.session_state.get('openai_api_key'):
            st.warning("⚠️ Configura la API Key de OpenAI (Copilot) en la barra lateral para usar el chat de Bonos.")
        else:
            if "fi_messages" not in st.session_state: st.session_state.fi_messages = []
            
            for msg in st.session_state.fi_messages:
                st.chat_message(msg["role"]).write(msg["content"])
                
            if prompt := st.chat_input("Pregúntale a la IA sobre convexidad, inmunización o tu curva de tasas..."):
                st.session_state.fi_messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                # Contexto inyectado en la IA de forma oculta
                context = f"Portafolio de bonos actual: MacDur={port_mac_dur:.2f}, ModDur={port_mod_dur:.2f}, Convexidad={port_convexity:.2f}. "
                
                try:
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    messages_for_api = [{"role": "system", "content": f"Eres un experto en Renta Fija. Basa tus respuestas en este contexto matemático del portafolio del usuario: {context}"}]
                    messages_for_api.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.fi_messages[-5:]])
                    
                    response = client.chat.completions.create(
                        model=st.session_state.get('openai_model', 'gpt-3.5-turbo'),
                        messages=messages_for_api
                    )
                    reply = response.choices[0].message.content
                    st.session_state.fi_messages.append({"role": "assistant", "content": reply})
                    st.chat_message("assistant").write(reply)
                except Exception as e:
                    st.error(f"Error de API: {e}")

# (El resto de funciones de Yahoo y Gemini se mantienen igual)
def page_yahoo_explorer():
    st.title("🌎 Explorador de Mercado (Yahoo Finance)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
        if not ticker: return
    with c2:
        period = st.selectbox("Periodo", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
    
    with st.spinner(f"Descargando datos..."):
        try:
            search_ticker = ticker
            stock = yf.Ticker(search_ticker)
            hist = stock.history(period=period)
            if hist.empty:
                search_ticker = ticker + ".BA"
                stock = yf.Ticker(search_ticker)
                hist = stock.history(period=period)
            if hist.empty: st.error("No hay datos."); return

            info = stock.info
            st.subheader(f"{info.get('longName', ticker)} ({search_ticker})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio", f"${info.get('currentPrice', hist['Close'].iloc[-1]):,.2f}")
            m2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            m3.metric("Beta", info.get('beta', 'N/A'))
            m4.metric("Sector", info.get('sector', 'N/A'))

            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title="Evolución", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Error: {e}")

def page_event_analyzer_gemini():
    st.header("📰 Analizador de Noticias con IA (Gemini)")
    if not GEMINI_OK: st.error("Librería google-generativeai no instalada."); return
    if not st.session_state.get('gemini_api_key'): st.warning("Configure API Key."); return
    news_text = st.text_area("Pega la noticia aquí:", height=150)
    if st.button("🤖 Analizar"):
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            model = genai.GenerativeModel(st.session_state.gemini_model)
            with st.spinner("Analizando..."):
                st.markdown(model.generate_content(f"Analiza financieramente: '{news_text}'").text)
        except Exception as e: st.error(f"Error: {e}")

def page_chat_gemini():
    st.header("💬 Asistente Gemini General")
    if not GEMINI_OK: st.error("Librería google-generativeai no instalada."); return
    if not st.session_state.get('gemini_api_key'): st.warning("Configure API Key."); return
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message("user" if msg["role"] == "user" else "assistant").write(msg["content"])
    if prompt := st.chat_input("Consulta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            model = genai.GenerativeModel(st.session_state.gemini_model)
            hist = [{'role': ('user' if m['role']=='user' else 'model'), 'parts': [m['content']]} for m in st.session_state.messages[-6:]]
            resp = model.start_chat(history=hist[:-1]).send_message(prompt).text
            st.session_state.messages.append({"role": "model", "content": resp})
            st.chat_message("assistant").write(resp)
        except Exception as e: st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN
# ═══════════════════════════════════════════════════════════════════════════

if 'selected_page' not in st.session_state: st.session_state.selected_page = "Inicio"
if 'portfolios' not in st.session_state: st.session_state.portfolios = load_portfolios_from_file()

st.sidebar.title("Configuración y Accesos")

with st.sidebar.expander("🤖 IA (OpenAI / Copilot)", expanded=True):
    st.markdown("<small>Usado para Renta Fija y Portafolios</small>", unsafe_allow_html=True)
    st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('openai_api_key', ''))
    st.session_state.openai_model = st.selectbox("Modelo OpenAI", ["gpt-3.5-turbo", "gpt-4o"])

with st.sidebar.expander("🧠 IA (Gemini)", expanded=False):
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get('gemini_api_key', ''))
    st.session_state.gemini_model = st.selectbox("Modelo", ["gemini-1.5-flash", "gemini-pro"])

with st.sidebar.expander("🏦 IOL"):
    user_iol = st.text_input("Usuario IOL")
    pass_iol = st.text_input("Pass IOL", type="password")
    if st.button("Conectar"): st.session_state.iol_username, st.session_state.iol_password = user_iol, pass_iol

st.sidebar.markdown("---")
opciones = [
    "Inicio", 
    "📊 Dashboard Corporativo", 
    "🏛️ Renta Fija (Bonos y Curvas)",  # <--- NUEVA PESTAÑA
    "🏦 Explorador IOL API", 
    "🌎 Explorador Global (Yahoo)", 
    "🔭 Modelos Avanzados (Forecast)", 
    "📰 Analizador Eventos (Gemini)", 
    "💬 Chat IA General (Gemini)"
]

sel = st.sidebar.radio("Navegación", opciones, index=opciones.index(st.session_state.selected_page) if st.session_state.selected_page in opciones else 0)

if sel != st.session_state.selected_page: st.session_state.selected_page = sel; st.rerun()

if sel == "Inicio": st.title("BPNos - Finanzas Corporativas"); st.info("Seleccione módulo en sidebar.")
elif sel == "📊 Dashboard Corporativo": page_corporate_dashboard()
elif sel == "🏛️ Renta Fija (Bonos y Curvas)": page_fixed_income() # <--- LLAMADA A NUEVA FUNCIÓN
elif sel == "🏦 Explorador IOL API": page_iol_explorer()
elif sel == "🌎 Explorador Global (Yahoo)": page_yahoo_explorer()
elif sel == "🔭 Modelos Avanzados (Forecast)": page_forecast()
elif sel == "📰 Analizador Eventos (Gemini)": page_event_analyzer_gemini()
elif sel == "💬 Chat IA General (Gemini)": page_chat_gemini()
