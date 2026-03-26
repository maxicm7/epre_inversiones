import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf

# ── IMPORTACIÓN SEGURA DE GOOGLE SHEETS ──────────────────────────────────
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_OK = True
except ImportError:
    GSHEETS_OK = False
    st.warning("⚠️ Instala: pip install gspread google-auth")

# ── IMPORTACIÓN SEGURA DE GOOGLE GEMINI ──
try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# ── IMPORTACIÓN SEGURA DE OPENAI ──
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# ── PyPortfolioOpt (optimización institucional) ──
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_OK = True
except ImportError:
    PYPFOPT_OK = False

# ── Módulos propios (fallback si no existen) ──
try:
    from forecast_module import page_forecast
    from iol_client import page_iol_explorer, get_iol_client
except ImportError:
    def page_forecast(): st.warning("🔧 Módulo forecast_module.py no encontrado.")
    def page_iol_explorer(): st.warning("🔧 Módulo iol_client.py no encontrado.")
    def get_iol_client(): return None

# ── Configuración Global ──────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="BPNos – Finanzas Corporativas", page_icon="📈")

# Configuración Google Sheets desde Secrets
SHEET_NAME = st.secrets.get("google_sheets", {}).get("sheet_name", "BPNos_Portfolios")
SHEET_ID = st.secrets.get("google_sheets", {}).get("sheet_id", "")
WORKSHEET_NAME = "portfolios"
PORTFOLIO_FILE = "portfolios_data.json"

# ═══════════════════════════════════════════════════════════════════════════
#  MÓDULO CAFCI (Fondos Comunes de Inversión - Argentina)
# ═══════════════════════════════════════════════════════════════════════════

def search_cafci_funds(query):
    """Busca fondos en la API oficial de CAFCI."""
    if not query or len(query) < 3:
        return []
    try:
        url = f"https://api.cafci.org.ar/fondo?nombre={query}&limit=20"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('data', [])
    except Exception as e:
        print(f"⚠️ Error búsqueda CAFCI: {e}")
    return []

def fetch_cafci_historical_vcp(fondo_id, clase_id, start_date, end_date):
    """Obtiene historial de Valor de Cuotaparte desde CAFCI."""
    try:
        fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        fmt_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        url = f"https://api.cafci.org.ar/fondo/{fondo_id}/clase/{clase_id}/cotizacion?desde={fmt_start}&hasta={fmt_end}"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            raw_data = response.json().get('data', [])
            if not raw_data:
                return pd.Series()
            df = pd.DataFrame(raw_data)
            df['fecha'] = pd.to_datetime(df['fecha'])
            df['vcp'] = pd.to_numeric(df['vcp'], errors='coerce')
            df = df.set_index('fecha').sort_index()
            df = df[~df.index.duplicated(keep='first')].dropna(subset=['vcp'])
            return df['vcp']
    except Exception as e:
        st.error(f"⚠️ Error CAFCI Data: {e}")
    return pd.Series()

# ═══════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS BACKEND (NUEVA API: open_by_key + scope mínimo)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_gsheets_client():
    """Cliente gspread autenticado con Service Account (scope: solo Sheets)."""
    if not GSHEETS_OK:
        return None
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        if "private_key" in creds_dict:
            creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.sidebar.error(f"❌ Error Google Sheets Auth: {e}")
        return None

def get_or_create_worksheet(client, sheet_name, worksheet_name):
    """Abre Sheet por ID (evita error 403 de Drive) y crea worksheet si no existe."""
    try:
        spreadsheet = client.open_by_key(SHEET_ID) if SHEET_ID else client.open(sheet_name)
    except Exception as e:
        st.error(f"❌ No se pudo abrir el Sheet. Verificá sheet_id en Secrets y permisos. Detalle: {e}")
        raise
    try:
        return spreadsheet.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=worksheet_name, rows=500, cols=5)
        ws.append_row(["name", "tickers", "weights"])
        return ws

def _load_portfolios_local_fallback() -> dict:
    """Fallback: carga desde JSON local si Sheets falla."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"⚠️ Error lectura JSON local: {e}")
    return {}

def _save_portfolios_local_fallback(portfolios_dict: dict) -> tuple[bool, str]:
    """Fallback: guarda en JSON local."""
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolios_dict, f, indent=4)
        return True, ""
    except Exception as e:
        return False, str(e)

def load_portfolios_from_file() -> dict:
    """Carga portafolios: prioriza Google Sheets, fallback a JSON local."""
    client = get_gsheets_client()
    if client is None:
        return _load_portfolios_local_fallback()
    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        records = ws.get_all_records()
        portfolios = {}
        for row in records:
            name = str(row.get("name", "")).strip()
            raw_tickers = str(row.get("tickers", "")).strip()
            raw_weights = str(row.get("weights", "")).strip()
            if not name or not raw_tickers:
                continue
            tickers = [t.strip() for t in raw_tickers.split(",") if t.strip()]
            try:
                weights = [float(w.strip()) for w in raw_weights.split(",") if w.strip()]
            except ValueError:
                weights = [1.0 / len(tickers)] * len(tickers)
            portfolios[name] = {"tickers": tickers, "weights": weights}
        return portfolios
    except Exception as e:
        st.error(f"⚠️ Error leyendo Google Sheets: {e}")
        return _load_portfolios_local_fallback()

def save_portfolios_to_file(portfolios_dict: dict) -> tuple[bool, str]:
    """Guarda portafolios: escribe en Google Sheets + respaldo JSON local."""
    client = get_gsheets_client()
    if client is None:
        return _save_portfolios_local_fallback(portfolios_dict)
    try:
        ws = get_or_create_worksheet(client, SHEET_NAME, WORKSHEET_NAME)
        ws.clear()
        rows = [["name", "tickers", "weights"]]
        for name, data in portfolios_dict.items():
            rows.append([name, ", ".join(data["tickers"]), ", ".join(map(str, data["weights"]))])
        ws.update(rows, "A1")
        _save_portfolios_local_fallback(portfolios_dict)
        return True, ""
    except Exception as e:
        st.error(f"⚠️ Error guardando en Google Sheets: {e}")
        return _save_portfolios_local_fallback(portfolios_dict)

def render_gsheets_status():
    """Muestra estado de conexión a Google Sheets en sidebar."""
    client = get_gsheets_client()
    if client:
        st.sidebar.success(f"🟢 Google Sheets: {SHEET_NAME}")
    else:
        st.sidebar.warning("🟡 Google Sheets: usando respaldo local")

# ═══════════════════════════════════════════════════════════════════════════
#  CORE FINANCIERO: DESCARGA, OPTIMIZACIÓN Y MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_prices_for_portfolio(tickers, start_date, end_date):
    """Descarga precios multifuente: CAFCI → IOL → Yahoo Finance."""
    client = get_iol_client()
    all_prices = {}
    yf_tickers = []

    for ticker in tickers:
        fetched = False
        # 1. CAFCI (FCI argentinos)
        if ticker.startswith("CAFCI:"):
            parts = ticker.split(":")
            f_id, c_id = parts[1], parts[2]
            s = fetch_cafci_historical_vcp(f_id, c_id, start_date, end_date)
            if not s.empty:
                name = parts[3] if len(parts) > 3 else f"FCI_{f_id}"
                all_prices[ticker] = s.rename(name)
                fetched = True
        # 2. IOL (mercado local)
        if not fetched and client:
            simbolo = ticker.split(".")[0].upper()
            fmt_start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            fmt_end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            try:
                df = client.get_serie_historica(simbolo, fmt_start, fmt_end)
                if not df.empty and "ultimoPrecio" in df.columns:
                    s = df["ultimoPrecio"].rename(ticker)
                    if s.index.tz is not None:
                        s.index = s.index.tz_localize(None)
                    all_prices[ticker] = s
                    fetched = True
            except:
                pass
        # 3. Yahoo Finance (fallback global)
        if not fetched:
            yf_tickers.append(ticker)

    # Descarga Yahoo Finance batch
    if yf_tickers:
        try:
            adjusted = [t if "." in t or t.endswith("=X") else t + ".BA" for t in yf_tickers]
            raw = yf.download(adjusted, start=start_date, end=end_date, progress=False)
            if not raw.empty:
                close = raw['Close'] if 'Close' in raw.columns else raw
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=yf_tickers[0])
                for col in close.columns:
                    clean = str(col).replace(".BA", "")
                    for orig in yf_tickers:
                        if clean == orig or str(col) == orig:
                            all_prices[orig] = close[col]
                            break
        except Exception as e:
            st.warning(f"⚠️ Yahoo Finance: {e}")

    if not all_prices:
        return None
    
    prices = pd.concat(all_prices.values(), axis=1).ffill().dropna()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices

def optimize_portfolio_corporate(prices, risk_free_rate=0.02, opt_type="Maximo Ratio Sharpe"):
    """Optimización: PyPortfolioOpt (si está) o fallback con SciPy."""
    returns = prices.pct_change().dropna()
    if returns.empty or len(returns) < 10:
        return None

    # Intento con PyPortfolioOpt
    if PYPFOPT_OK:
        try:
            mu = expected_returns.mean_historical_return(prices, frequency=252)
            S = risk_models.sample_cov(prices, frequency=252)
            if not mu.isnull().any() and not S.isnull().values.any():
                ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                if opt_type == "Maximo Ratio Sharpe":
                    ef.max_sharpe(risk_free_rate=risk_free_rate)
                elif opt_type == "Minima Volatilidad":
                    ef.min_volatility()
                else:
                    ef.max_quadratic_utility(risk_aversion=0.01)
                weights = ef.clean_weights()
                ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
                w_array = np.array([weights.get(c, 0) for c in prices.columns])
                return {
                    "weights": w_array, "expected_return": float(ret), "volatility": float(vol),
                    "sharpe_ratio": float(sharpe), "tickers": list(prices.columns),
                    "returns": returns, "method": "PyPortfolioOpt"
                }
        except Exception:
            pass  # Fallback a SciPy

    # Fallback con SciPy
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(mean_returns)

    def get_metrics(w):
        ret = np.sum(mean_returns * w)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (ret - risk_free_rate) / vol if vol > 1e-8 else 0
        return np.array([ret, vol, sr])

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.array([1/n] * n)

    if opt_type == "Minima Volatilidad":
        fun = lambda w: get_metrics(w)[1]
    elif opt_type == "Retorno Maximo":
        fun = lambda w: -get_metrics(w)[0]
    else:
        fun = lambda w: -get_metrics(w)[2] if (mean_returns > risk_free_rate).any() else get_metrics(w)[1]

    res = minimize(fun, init, method='SLSQP', bounds=bounds, constraints=constraints)
    final_w = np.maximum(res.x if res.success else init, 0)
    if np.sum(final_w) > 0:
        final_w = final_w / np.sum(final_w)
    
    m = get_metrics(final_w)
    return {
        "weights": final_w, "expected_return": float(m[0]), "volatility": float(m[1]),
        "sharpe_ratio": float(m[2]), "tickers": list(prices.columns),
        "returns": returns, "method": "Scipy/SLSQP"
    }

def calc_bond_metrics(face_value, coupon_rate, ytm, years_to_maturity, freq=2):
    """Calcula precio, duración Macaulay, duración modificada y convexidad de un bono."""
    periods = int(years_to_maturity * freq)
    coupon = (coupon_rate / freq) * face_value
    rate = ytm / freq
    price, mac_num, conv_num = 0, 0, 0
    
    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / ((1 + rate) ** t)
        price += pv
        mac_num += (t / freq) * pv
        conv_num += (t / freq) * ((t / freq) + (1/freq)) * cf / ((1 + rate) ** (t + 2))
    
    mac_dur = mac_num / price if price > 0 else 0
    mod_dur = mac_dur / (1 + rate)
    convexity = conv_num / price if price > 0 else 0
    return price, mac_dur, mod_dur, convexity

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA: DASHBOARD CORPORATIVO
# ═══════════════════════════════════════════════════════════════════════════

def page_corporate_dashboard():
    st.title("📊 Dashboard Corporativo Integral")
    tabs = st.tabs(["💼 Gestión de Portafolios", "🚀 Optimización & Riesgo", "🔮 Forecast & Simulación"])

    with tabs[0]:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.subheader("Administrar Carteras")
            action = st.radio("Acción:", ["✨ Crear Nueva", "✏️ Editar / 🗑️ Eliminar"], horizontal=True)

            if action == "✨ Crear Nueva":
                p_name = st.text_input("Nombre de la Cartera")
                
                # Buscador CAFCI integrado
                with st.expander("🔍 Buscador de Fondos FCI (CAFCI)", expanded=False):
                    fci_q = st.text_input("Buscar fondo por nombre...")
                    if fci_q and len(fci_q) >= 3:
                        res = search_cafci_funds(fci_q)
                        for f in res:
                            for cl in f.get('clases', []):
                                t_fci = f"CAFCI:{f['id']}:{cl['id']}:{f['nombre']} {cl['nombre']}"
                                if st.button(f"➕ {f['nombre']} ({cl['nombre']})", key=f"add_{cl['id']}"):
                                    curr = st.session_state.get('tmp_tickers', "")
                                    st.session_state.tmp_tickers = curr + f"{t_fci}, "
                                    st.rerun()
                
                p_tickers = st.text_area("Tickers / Activos", value=st.session_state.get('tmp_tickers', "AL30, GGAL")).upper()
                p_weights = st.text_area("Pesos (deben sumar 1.0)", "0.5, 0.5")

                if st.button("💾 Guardar Cartera", type="primary"):
                    try:
                        t = [x.strip() for x in p_tickers.split(",") if x.strip()]
                        w = [float(x) for x in p_weights.split(",") if x.strip()]
                        if not p_name:
                            st.error("El nombre no puede estar vacío.")
                        elif len(t) == len(w) and abs(sum(w) - 1.0) < 0.02:
                            st.session_state.portfolios[p_name] = {"tickers": t, "weights": w}
                            ok, err = save_portfolios_to_file(st.session_state.portfolios)
                            if ok:
                                st.success("✅ Guardado en Google Sheets + respaldo local.")
                                st.session_state.tmp_tickers = ""
                                st.rerun()
                            else:
                                st.error(f"⚠️ Error al guardar: {err}")
                        else:
                            st.error("⚠️ Los pesos no coinciden con los tickers o no suman 1.0")
                    except Exception as e:
                        st.error(f"⚠️ Error de formato: {e}")
            
            else:  # Editar/Eliminar
                if st.session_state.portfolios:
                    sel = st.selectbox("Seleccionar cartera:", list(st.session_state.portfolios.keys()))
                    curr = st.session_state.portfolios[sel]
                    new_n = st.text_input("Renombrar", value=sel)
                    new_t = st.text_area("Tickers", value=", ".join(curr["tickers"])).upper()
                    new_w = st.text_area("Pesos", value=", ".join(map(str, curr["weights"])))
                    
                    col_b1, col_b2 = st.columns(2)
                    if col_b1.button("🔄 Actualizar", type="primary", use_container_width=True):
                        try:
                            t = [x.strip() for x in new_t.split(",") if x.strip()]
                            w = [float(x) for x in new_w.split(",") if x.strip()]
                            if len(t) == len(w) and abs(sum(w)-1.0) < 0.02:
                                if new_n != sel:
                                    del st.session_state.portfolios[sel]
                                st.session_state.portfolios[new_n] = {"tickers": t, "weights": w}
                                save_portfolios_to_file(st.session_state.portfolios)
                                st.success("✅ Cartera actualizada.")
                                st.rerun()
                            else:
                                st.error("⚠️ Revisar validación de pesos y tickers.")
                        except:
                            st.error("⚠️ Error de formato.")
                    
                    if col_b2.button("🗑️ Eliminar", type="primary", use_container_width=True):
                        del st.session_state.portfolios[sel]
                        save_portfolios_to_file(st.session_state.portfolios)
                        st.warning("🗑️ Cartera eliminada.")
                        st.rerun()
                else:
                    st.info("📭 No hay carteras guardadas aún.")

        with c2:
            st.subheader("Base de Datos Activa")
            if st.session_state.portfolios:
                df_p = pd.DataFrame([
                    {"Nombre": k, "Activos": len(v["tickers"]), 
                     "Pesos": ", ".join([f"{w*100:.1f}%" for w in v["weights"]])}
                    for k, v in st.session_state.portfolios.items()
                ])
                st.dataframe(df_p, use_container_width=True, hide_index=True)
            else:
                st.info("Agregá tu primera cartera para comenzar.")

    with tabs[1]:
        if not st.session_state.portfolios:
            st.info("💡 Creá una cartera en la pestaña anterior para analizar.")
            return
            
        col1, col2, col3 = st.columns(3)
        p_sel = col1.selectbox("Cartera a Analizar", list(st.session_state.portfolios.keys()))
        d_s = col2.date_input("Desde", pd.to_datetime("2023-01-01"))
        d_e = col3.date_input("Hasta", pd.to_datetime("today"))

        # Rendimiento Histórico
        st.subheader("📈 Rendimiento Histórico")
        if st.button("📊 Calcular Rendimiento"):
            with st.spinner("Descargando datos multifuente..."):
                prices = fetch_stock_prices_for_portfolio(
                    st.session_state.portfolios[p_sel]["tickers"], d_s, d_e)
            
            if prices is not None:
                # Ajuste de pesos si faltan datos de algún ticker
                tickers_avail = [t for t in st.session_state.portfolios[p_sel]["tickers"] if t in prices.columns]
                if len(tickers_avail) < len(st.session_state.portfolios[p_sel]["tickers"]):
                    missing = set(st.session_state.portfolios[p_sel]["tickers"]) - set(tickers_avail)
                    st.warning(f"⚠️ Sin datos para: {', '.join(missing)}. Pesos recalculados.")
                    idx = [st.session_state.portfolios[p_sel]["tickers"].index(t) for t in tickers_avail]
                    raw_w = [st.session_state.portfolios[p_sel]["weights"][i] for i in idx]
                    total = sum(raw_w)
                    weights_adj = [w/total for w in raw_w] if total > 0 else [1/len(tickers_avail)]*len(tickers_avail)
                else:
                    weights_adj = st.session_state.portfolios[p_sel]["weights"]
                
                prices_f = prices[tickers_avail]
                norm_p = prices_f / prices_f.iloc[0]
                port_val = (norm_p * np.array(weights_adj)).sum(axis=1) * 100
                
                # Métricas
                total_ret = (port_val.iloc[-1] / port_val.iloc[0] - 1) * 100
                daily_r = port_val.pct_change().dropna()
                ann_vol = daily_r.std() * np.sqrt(252) * 100
                max_dd = ((port_val / port_val.cummax()) - 1).min() * 100
                sharpe_h = (daily_r.mean() * 252) / (daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Retorno Total", f"{total_ret:.2f}%")
                m2.metric("Volatilidad Anual", f"{ann_vol:.2f}%")
                m3.metric("Máx. Drawdown", f"{max_dd:.2f}%")
                m4.metric("Sharpe Hist.", f"{sharpe_h:.2f}")
                
                # Gráfico evolución
                fig = go.Figure()
                colors = px.colors.qualitative.Pastel
                for i, t in enumerate(tickers_avail):
                    fig.add_trace(go.Scatter(x=norm_p.index, y=norm_p[t]*100, name=t,
                        line=dict(width=1.2, dash='dot', color=colors[i%len(colors)]), opacity=0.6))
                fig.add_trace(go.Scatter(x=port_val.index, y=port_val, name=f"📂 {p_sel}",
                    line=dict(width=3, color='#00CC96'), fill='tozeroy', fillcolor='rgba(0,204,150,0.08)'))
                fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4)
                fig.update_layout(title=f"Evolución – {p_sel} (Base 100)", template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown
                dd_series = (port_val / port_val.cummax() - 1) * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd_series.index, y=dd_series, mode='lines',
                    fill='tozeroy', fillcolor='rgba(239,85,59,0.2)', line=dict(color='#EF553B')))
                fig_dd.update_layout(title="Drawdown del Portafolio (%)", template="plotly_dark", height=200)
                st.plotly_chart(fig_dd, use_container_width=True)
            else:
                st.error("⚠️ No se pudieron obtener datos de precios.")

        # Optimización
        st.markdown("---")
        st.subheader("⚙️ Optimización de Frontera Eficiente")
        c_opt1, c_opt2 = st.columns(2)
        rf = c_opt1.number_input("Tasa Libre de Riesgo", 0.0, 0.5, 0.04, step=0.01)
        target = c_opt2.selectbox("Objetivo", ["Maximo Ratio Sharpe", "Minima Volatilidad", "Retorno Maximo"])
        
        if st.button("🚀 Ejecutar Optimización"):
            with st.spinner("Optimizando portafolio..."):
                prices = fetch_stock_prices_for_portfolio(
                    st.session_state.portfolios[p_sel]["tickers"], d_s, d_e)
            if prices is not None:
                res = optimize_portfolio_corporate(prices, risk_free_rate=rf, opt_type=target)
                if res:
                    st.session_state.last_opt_res = res
                    c_k1, c_k2, c_k3 = st.columns(3)
                    c_k1.metric("Retorno Esperado", f"{res['expected_return']:.1%}")
                    c_k2.metric("Volatilidad Anual", f"{res['volatility']:.1%}")
                    c_k3.metric("Ratio Sharpe", f"{res['sharpe_ratio']:.2f}")
                    
                    w_df = pd.DataFrame({"Activo": res['tickers'], "Peso": res['weights']}).query("Peso > 0.001")
                    if not w_df.empty:
                        fig_pie = px.pie(w_df, values="Peso", names="Activo", title="Asignación Óptima", 
                                        hole=0.4, template="plotly_dark")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Botón análisis con IA
                    if st.button("🧠 Analizar con IA"):
                        if not st.session_state.get('preferred_ai'):
                            st.warning("⚠️ Configurá una API Key en el menú lateral.")
                        else:
                            with st.spinner(f"Consultando {st.session_state.preferred_ai}..."):
                                try:
                                    ctx = f"Retorno: {res['expected_return']:.2%}, Vol: {res['volatility']:.2%}, Sharpe: {res['sharpe_ratio']:.2f}. Activos: {res['tickers']}"
                                    prompt = f"Como asesor financiero institucional, analizá este portafolio óptimo: {ctx}. ¿Cuáles son sus fortalezas, riesgos y recomendaciones?"
                                    if st.session_state.preferred_ai == "OpenAI":
                                        client = OpenAI(api_key=st.session_state.openai_api_key)
                                        resp = client.chat.completions.create(
                                            model=st.session_state.openai_model,
                                            messages=[{"role": "user", "content": prompt}])
                                        st.info(resp.choices[0].message.content)
                                    else:
                                        genai.configure(api_key=st.session_state.gemini_api_key)
                                        model = genai.GenerativeModel(st.session_state.gemini_model)
                                        st.info(model.generate_content(prompt).text)
                                except Exception as e:
                                    st.error(f"⚠️ Error IA: {e}")
                else:
                    st.error("⚠️ No se pudo optimizar. Verificá que haya datos suficientes.")
            else:
                st.error("⚠️ Error descargando datos para optimización.")

    with tabs[2]:
        if 'last_opt_res' not in st.session_state:
            st.info("💡 Optimizá un portafolio primero para ver la simulación Montecarlo.")
            return
            
        res = st.session_state.last_opt_res
        st.subheader("🎲 Simulación Montecarlo")
        c_s1, c_s2 = st.columns(2)
        days = c_s1.slider("Días a proyectar", 30, 730, 252)
        sims = c_s2.selectbox("Simulaciones", [100, 500, 1000, 5000], index=1)
        
        if st.button("🚀 Correr Simulación"):
            dt = 1/252
            mu = res['expected_return'] * dt
            sigma = res['volatility'] * np.sqrt(dt)
            paths = np.zeros((days, sims))
            paths[0] = 100
            for t in range(1, days):
                paths[t] = paths[t-1] * np.exp((mu - 0.5*sigma**2) + sigma * np.random.standard_normal(sims))
            
            fig_m = go.Figure()
            x_ax = np.arange(days)
            p95, p05 = np.percentile(paths, 95, axis=1), np.percentile(paths, 5, axis=1)
            fig_m.add_trace(go.Scatter(x=np.concatenate([x_ax, x_ax[::-1]]),
                y=np.concatenate([p95, p05[::-1]]), fill='toself',
                fillcolor='rgba(255,255,255,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Intervalo 90%'))
            fig_m.add_trace(go.Scatter(x=x_ax, y=np.mean(paths, axis=1), name='Media',
                line=dict(color='#00CC96', width=3)))
            fig_m.add_trace(go.Scatter(x=x_ax, y=p05, name='Escenario Pesimista (5%)',
                line=dict(color='#EF553B', dash='dash')))
            fig_m.update_layout(title=f"Proyección a {days} días", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_m, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA: RENTA FIJA (BONOS, DURACIÓN, INMUNIZACIÓN)
# ═══════════════════════════════════════════════════════════════════════════

def page_fixed_income():
    st.title("🏛️ Renta Fija: Análisis, Sensibilidad e Inmunización")
    st.markdown("Calculá duración, convexidad y medí el impacto de cambios en tasas sobre tu cartera de bonos.")
    
    # Inicializar datos de ejemplo
    if 'bonds_data' not in st.session_state:
        st.session_state.bonds_data = pd.DataFrame({
            "Bono": ["AL30", "GD30", "TX26"],
            "Cupón (%)": [0.75, 0.75, 4.0],
            "YTM (%)": [45.0, 38.0, 12.0],
            "Años a Venc.": [6.0, 11.0, 1.5],
            "Nominal Invertido": [10000, 5000, 20000]
        })
    
    # Editor dinámico de bonos
    edited = st.data_editor(st.session_state.bonds_data, num_rows="dynamic", use_container_width=True)
    st.session_state.bonds_data = edited
    
    # Cálculos por bono
    results, total_inv = [], 0
    for _, r in edited.iterrows():
        try:
            p, macd, modd, conv = calc_bond_metrics(
                face_value=100, coupon_rate=r["Cupón (%)"]/100,
                ytm=r["YTM (%)"]/100, years_to_maturity=r["Años a Venc."])
            val = (p/100) * r["Nominal Invertido"]
            total_inv += val
            results.append({"Bono": r["Bono"], "Precio": p, "Mac. Dur": macd, 
                          "Mod. Dur": modd, "Convexidad": conv, "Market Val": val})
        except:
            pass
    
    if total_inv > 0 and results:
        df_res = pd.DataFrame(results)
        df_res["Peso %"] = df_res["Market Val"] / total_inv
        port_mod_dur = (df_res["Mod. Dur"] * df_res["Peso %"]).sum()
        port_conv = (df_res["Convexidad"] * df_res["Peso %"]).sum()
        
        tabs = st.tabs(["📊 Métricas de Riesgo", "📉 Test de Estrés", "📈 Curva de Rendimiento", "💬 Chat IA Bonos"])
        
        with tabs[0]:
            c1, c2, c3 = st.columns(3)
            c1.metric("Modified Duration Cartera", f"{port_mod_dur:.2f}")
            c2.metric("Convexidad Cartera", f"{port_conv:.4f}")
            c3.metric("Valor Total Market", f"${total_inv:,.0f}")
            st.dataframe(df_res.style.format({
                "Precio": "{:.2f}", "Mac. Dur": "{:.2f}", "Mod. Dur": "{:.2f}",
                "Convexidad": "{:.4f}", "Peso %": "{:.2%}", "Market Val": "${:,.0f}"
            }), use_container_width=True)
            
            st.markdown("---")
            st.subheader("🛡️ Análisis de Inmunización")
            horizonte = st.slider("Tu Horizonte de Inversión (Años)", 0.5, 20.0, 5.0, 0.5)
            gap = (df_res["Mac. Dur"] * df_res["Peso %"]).sum() - horizonte
            if abs(gap) < 0.25:
                st.success(f"✅ Portafolio INMUNIZADO: Duración Macaulay coincide con tu horizonte.")
            elif gap > 0:
                st.warning(f"⚠️ Riesgo de PRECIO: Duración > horizonte. Suba de tasas reduce valor final.")
            else:
                st.info(f"ℹ️ Riesgo de REINVERSIÓN: Duración < horizonte. Baja de tasas reduce ingresos futuros.")
        
        with tabs[1]:
            st.subheader("📉 Test de Estrés (Aproximación Taylor)")
            shock_bps = st.slider("Shock en Tasas (puntos básicos)", -1000, 1000, 100, 50)
            shock_pct = shock_bps / 10000
            impacto = (-port_mod_dur * shock_pct + 0.5 * port_conv * shock_pct**2) * 100
            st.metric("Impacto Estimado en Cartera", f"{impacto:.2f}%", delta=f"{shock_bps} bps")
            df_res["Impacto %"] = (-df_res["Mod. Dur"]*shock_pct + 0.5*df_res["Convexidad"]*shock_pct**2) * 100
            fig_stress = px.bar(df_res, x="Bono", y="Impacto %", color="Impacto %",
                              title=f"Impacto por Activo ante shock de {shock_bps} bps",
                              color_continuous_scale="RdYlGn")
            st.plotly_chart(fig_stress, use_container_width=True)
        
        with tabs[2]:
            st.subheader("📈 Curva de Rendimiento del Portafolio")
            df_curve = edited.sort_values("Años a Venc.")
            fig_curve = px.line(df_curve, x="Años a Venc.", y="YTM (%)", markers=True, text="Bono",
                              title="Estructura Temporal de Tasas (YTM vs Plazo)")
            fig_curve.update_traces(textposition="top center")
            fig_curve.update_layout(template="plotly_dark")
            st.plotly_chart(fig_curve, use_container_width=True)
        
        with tabs[3]:
            st.subheader("💬 Asistente IA Especialista en Renta Fija")
            if not st.session_state.get('preferred_ai'):
                st.warning("⚠️ Configurá una API Key en el menú lateral para usar IA.")
            else:
                ctx = f"Portafolio: ModDur={port_mod_dur:.2f}, Conv={port_conv:.4f}, Inversión=${total_inv:,.0f}"
                if "bond_chat" not in st.session_state:
                    st.session_state.bond_chat = []
                for m in st.session_state.bond_chat:
                    st.chat_message(m["role"]).write(m["content"])
                if prompt := st.chat_input("Consultá sobre duración, inmunización o estrategia de bonos..."):
                    st.session_state.bond_chat.append({"role": "user", "content": prompt})
                    st.chat_message("user").write(prompt)
                    with st.spinner("Analizando..."):
                        try:
                            full_prompt = f"Contexto técnico: {ctx}. Pregunta del usuario: {prompt}"
                            if st.session_state.preferred_ai == "OpenAI":
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                resp = client.chat.completions.create(
                                    model=st.session_state.openai_model,
                                    messages=[{"role": "user", "content": full_prompt}])
                                reply = resp.choices[0].message.content
                            else:
                                genai.configure(api_key=st.session_state.gemini_api_key)
                                model = genai.GenerativeModel(st.session_state.gemini_model)
                                reply = model.generate_content(full_prompt).text
                            st.session_state.bond_chat.append({"role": "assistant", "content": reply})
                            st.chat_message("assistant").write(reply)
                        except Exception as e:
                            st.error(f"⚠️ Error IA: {e}")
    else:
        st.info("📋 Agregá bonos a la tabla para ver el análisis de riesgo.")

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA: ASISTENTE QUANT (ESTRATEGIA CON IA)
# ═══════════════════════════════════════════════════════════════════════════

def page_ai_strategy_assistant():
    st.header("🧠 Asistente Quant: Estrategia con IA")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configurá una API Key en el menú lateral para usar esta función.")
        return
    
    user_prompt = st.text_area("Describe tu objetivo de inversión:", height=150,
        placeholder="Ej: 'Busco un portafolio agresivo con bonos soberanos argentinos y acciones tecnológicas globales, con horizonte a 3 años...'")
    
    if st.button("🤖 Traducir a Filtros Quant", type="primary"):
        sys_prompt = """Eres un experto quant institucional. Traducí la estrategia del usuario a JSON válido con estas claves EXACTAS:
        - asset_allocation: dict con % por clase de activo (ej: {"Acciones US": 40, "Bonos AR": 30})
        - risk_level: entero 1-10
        - suggested_tickers: lista de tickers relevantes
        - duration_target: rango de duración objetivo para renta fija (ej: "2-4 años")
        Respondé SOLO con el bloque JSON, sin texto adicional."""
        
        with st.spinner(f"Consultando {st.session_state.preferred_ai}..."):
            try:
                if st.session_state.preferred_ai == "OpenAI":
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    resp = client.chat.completions.create(
                        model=st.session_state.openai_model,
                        messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                        temperature=0.1)
                    txt = resp.choices[0].message.content
                else:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel(st.session_state.gemini_model)
                    txt = model.generate_content(f"{sys_prompt}\n\n{user_prompt}",
                        generation_config=genai.types.GenerationConfig(temperature=0.1)).text
                
                # Extraer JSON del response
                json_match = re.search(r'\{.*\}', txt, re.DOTALL)
                if json_match:
                    strategy = json.loads(json_match.group(0))
                    st.markdown("### 🎯 Estrategia Sugerida")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Distribución de Activos:**")
                        st.json(strategy.get("asset_allocation", {}))
                        st.write(f"🎲 Nivel de Riesgo: {strategy.get('risk_level', 'N/A')}/10")
                        st.write(f"📊 Tickers Sugeridos: {', '.join(strategy.get('suggested_tickers', []))}")
                    with c2:
                        st.write(f"⏱️ Duración Objetivo: {strategy.get('duration_target', 'N/A')}")
                        st.info("💡 Tip: Usá estos filtros en el Dashboard Corporativo para construir tu portafolio.")
                    st.code(json.dumps(strategy, indent=2, ensure_ascii=False), language="json")
                else:
                    st.warning("⚠️ La IA no devolvió un JSON válido. Respuesta recibida:")
                    st.markdown(txt)
            except Exception as e:
                st.error(f"⚠️ Error en IA: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINAS ADICIONALES (Yahoo, Chat General, etc.)
# ═══════════════════════════════════════════════════════════════════════════

def page_yahoo_explorer():
    st.title("🌎 Explorador Global (Yahoo Finance)")
    c1, c2 = st.columns([2, 1])
    with c1: ticker = st.text_input("Ticker", value="AAPL").upper()
    with c2: period = st.selectbox("Período", ["1mo","3mo","6mo","1y","2y","5y","ytd","max"], index=3)
    
    if not ticker:
        return
    with st.spinner("Descargando datos..."):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                stock = yf.Ticker(ticker + ".BA")
                hist = stock.history(period=period)
            if hist.empty:
                st.error("⚠️ No hay datos disponibles para este ticker.")
                return
            info = stock.info
            st.subheader(f"{info.get('longName', ticker)}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precio", f"${info.get('currentPrice', hist['Close'].iloc[-1]):,.2f}")
            m2.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            m3.metric("Beta", info.get('beta', 'N/A'))
            m4.metric("Sector", info.get('sector', 'N/A'))
            fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                low=hist['Low'], close=hist['Close'])])
            fig.update_layout(title=f"Evolución de {ticker}", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

def page_chat_general():
    st.header("💬 Chat Financiero General")
    if not st.session_state.get('preferred_ai'):
        st.warning("⚠️ Configurá una API Key en el menú lateral.")
        return
    if "chat_hist" not in st.session_state:
        st.session_state.chat_hist = []
    for m in st.session_state.chat_hist:
        st.chat_message(m["role"]).write(m["content"])
    if prompt := st.chat_input("Escribí tu consulta financiera..."):
        st.session_state.chat_hist.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("Pensando..."):
            try:
                if st.session_state.preferred_ai == "OpenAI":
                    client = OpenAI(api_key=st.session_state.openai_api_key)
                    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_hist[-8:]]
                    resp = client.chat.completions.create(model=st.session_state.openai_model, messages=msgs)
                    reply = resp.choices[0].message.content
                else:
                    genai.configure(api_key=st.session_state.gemini_api_key)
                    model = genai.GenerativeModel(st.session_state.gemini_model)
                    hist = [{'role': 'user' if m['role']=='user' else 'model', 'parts': [m['content']]} 
                           for m in st.session_state.chat_hist[-8:-1]]
                    chat = model.start_chat(history=hist)
                    reply = chat.send_message(prompt).text
                st.session_state.chat_hist.append({"role": "assistant", "content": reply})
                st.chat_message("assistant").write(reply)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR Y NAVEGACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

# Inicialización de estado
if 'portfolios' not in st.session_state:
    st.session_state.portfolios = load_portfolios_from_file()
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Inicio"

st.sidebar.title("⚙️ BPNos – Configuración")

# Estado Google Sheets
render_gsheets_status()
st.sidebar.markdown("---")

# Configuración IA
with st.sidebar.expander("🤖 Configuración de IA", expanded=True):
    st.session_state.openai_api_key = st.text_input("🔑 OpenAI API Key", type="password",
        value=st.session_state.get('openai_api_key', st.secrets.get("openai", {}).get("api_key", "")))
    st.session_state.openai_model = st.selectbox("Modelo OpenAI", 
        ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], index=0)
    st.session_state.gemini_api_key = st.text_input("🔑 Gemini API Key", type="password",
        value=st.session_state.get('gemini_api_key', st.secrets.get("gemini", {}).get("api_key", "")))
    st.session_state.gemini_model = st.selectbox("Modelo Gemini",
        ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"], index=0)

# Selector de motor IA
available_ais = []
if OPENAI_OK and st.session_state.get('openai_api_key'): available_ais.append("OpenAI")
if GEMINI_OK and st.session_state.get('gemini_api_key'): available_ais.append("Gemini")
if available_ais:
    st.session_state.preferred_ai = st.sidebar.radio("✨ Motor IA Activo", available_ais)
else:
    st.session_state.preferred_ai = None
    st.sidebar.info("💡 Ingresá una API Key para activar las funciones de IA")

# Conexión IOL
with st.sidebar.expander("🏦 Conexión IOL", expanded=True):
    u = st.text_input("Usuario IOL", value=st.session_state.get('iol_username', ''))
    p = st.text_input("Contraseña IOL", type="password", value=st.session_state.get('iol_password', ''))
    if st.button("🔗 Conectar", use_container_width=True):
        st.session_state.iol_username = u
        st.session_state.iol_password = p
        with st.spinner("Validando..."):
            client = get_iol_client()
            st.session_state.iol_connected = client is not None
    if st.session_state.get('iol_connected'):
        st.success(f"🟢 Conectado: {st.session_state.iol_username}")
    else:
        st.caption("🔴 Desconectado (opcional: Yahoo Finance como fallback)")

st.sidebar.markdown("---")

# Menú de navegación
menu = [
    "Inicio", "📊 Dashboard Corporativo", "🏛️ Renta Fija Avanzada",
    "🧠 Asistente Quant IA", "🏦 Explorador IOL API",
    "🌎 Explorador Global (Yahoo)", "💬 Chat Financiero"
]
choice = st.sidebar.radio("🧭 Navegación", menu,
    index=menu.index(st.session_state.selected_page) if st.session_state.selected_page in menu else 0)

if choice != st.session_state.selected_page:
    st.session_state.selected_page = choice
    st.rerun()

# Ruteo de páginas
if choice == "Inicio":
    st.title("🚀 BPNos: Terminal de Finanzas Corporativas")
    st.markdown("""
    ### Plataforma integral para análisis financiero institucional
    
    🔹 **CAFCI**: Integración nativa con Fondos Comunes de Inversión argentinos  
    🔹 **IOL/Yahoo**: Datos en tiempo real de bonos, acciones y FX  
    🔹 **Quant**: Optimización de carteras, Montecarlo y métricas de riesgo  
    🔹 **IA**: Traducción de estrategias, análisis de noticias y asistencia experta  
    🔹 **Renta Fija**: Duración, convexidad, inmunización y test de estrés  
    
    > 💡 *Configurá tus API Keys en el menú lateral para activar todas las funciones.*
    """)
    st.image("https://images.unsplash.com/photo-1611974717482-98252c00d632?auto=format&fit=crop&q=80&w=1200", 
             use_column_width=True)
    
elif choice == "📊 Dashboard Corporativo":
    page_corporate_dashboard()
elif choice == "🏛️ Renta Fija Avanzada":
    page_fixed_income()
elif choice == "🧠 Asistente Quant IA":
    page_ai_strategy_assistant()
elif choice == "🏦 Explorador IOL API":
    page_iol_explorer()
elif choice == "🌎 Explorador Global (Yahoo)":
    page_yahoo_explorer()
elif choice == "💬 Chat Financiero":
    page_chat_general()
