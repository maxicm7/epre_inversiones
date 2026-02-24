"""
iol_client.py
─────────────────────────────────────────────────────────────────
Cliente para la API oficial de InvertirOnline (IOL) v2.
Maneja autenticación Bearer, refresco de token y todos los
endpoints necesarios para cotizaciones, FCI y serie histórica.
"""

import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

# ── Base URL ──────────────────────────────────────────────────────────────
BASE     = "https://api.invertironline.com"  # ✅ Fixed: No trailing spaces
AUTH_URL = f"{BASE}/token"
API_URL  = f"{BASE}/api/v2"

# ── Instrumentos y mercados disponibles ───────────────────────────────────
INSTRUMENTOS = {
    "Acciones":        "Acciones",
    "CEDEARs":         "CEDEARs",
    "Bonos":           "Bonos",
    "Obligaciones":    "ObligacionesNegociables",
    "Cauciones":       "Cauciones",
    "Opciones":        "Opciones",
    "FCI":             "FCI",
    "Futuros":         "Futuros",
}

PANELES = {
    "Todos":           "Todos",
    "Merval":          "Merval",
    "Merval 25":       "Merval25",
    "General":         "General",
    "Bonos Públicos":  "BonosPublicos",
    "Bonos Privados":  "BonosPrivados",
    "Letras":          "Letras",
}

PAISES = {
    "Argentina": "argentina",
    "USA":       "estados_unidos",
}

MERCADOS = {
    "BCBA":  "bCBA",
    "NYSE":  "nYSE",
    "NASDAQ":"nASDAQ",
}


# ═══════════════════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class IOLClient:
    """
    Cliente liviano para la API de IOL.
    Guarda el token en st.session_state para no reautenticar en cada rerun.
    """

    def __init__(self, username: str, password: str):
        self.username = username.strip()
        self.password = password
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    # ── Autenticación ─────────────────────────────────────────────────────
    def authenticate(self) -> bool:
        """Obtiene un token Bearer y lo almacena."""
        try:
            resp = requests.post(
                AUTH_URL,
                data={
                    "username":   self.username,
                    "password":   self.password,
                    "grant_type": "password",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token  = data.get("access_token")
            expires_in   = int(data.get("expires_in", 3600))
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
            return True
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ IOL Auth error {e.response.status_code}: credenciales inválidas o servidor caído.")
            return False
        except Exception as e:
            st.error(f"❌ IOL Auth error: {e}")
            return False

    def _ensure_token(self) -> bool:
        """Reautentica si el token expiró o no existe."""
        if not self._token or (self._token_expiry and datetime.now() >= self._token_expiry):
            return self.authenticate()
        return True

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type":  "application/json",
        }

    # ── Request genérico ──────────────────────────────────────────────────
    def _get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """GET autenticado. Reintenta una vez si el token expiró (401)."""
        if not self._ensure_token():
            return None
        
        # ✅ Fixed: Ensure clean URL construction
        url = f"{API_URL}{endpoint}"
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            if resp.status_code == 401:
                # Token expirado → reautenticar y reintentar
                self._token = None
                if not self.authenticate():
                    return None
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ IOL HTTP {e.response.status_code} en {endpoint}")
            return None
        except Exception as e:
            st.error(f"❌ IOL error en {endpoint}: {e}")
            return None

    # ══════════════════════════════════════════════════════════════════════
    #  ENDPOINTS
    # ══════════════════════════════════════════════════════════════════════

    # ── 1. Cotizaciones por instrumento (todos) ───────────────────────────
    def get_cotizaciones_todos(self, instrumento: str, pais: str = "argentina") -> pd.DataFrame:
        """
        GET /api/v2/Cotizaciones/{Instrumento}/{Pais}/Todos
        Devuelve todas las cotizaciones del instrumento.
        """
        data = self._get(f"/Cotizaciones/{instrumento}/{pais}/Todos")
        return self._parse_cotizaciones(data)

    # ── 2. Cotizaciones por panel ─────────────────────────────────────────
    def get_cotizaciones_panel(self, instrumento: str, panel: str,
                                pais: str = "argentina") -> pd.DataFrame:
        """
        GET /api/v2/Cotizaciones/{Instrumento}/{Panel}/{Pais}
        """
        data = self._get(f"/Cotizaciones/{instrumento}/{panel}/{pais}")
        return self._parse_cotizaciones(data)

    # ── 3. Cotización individual ──────────────────────────────────────────
    def get_cotizacion(self, mercado: str, simbolo: str) -> Optional[dict]:
        """
        GET /api/v2/{Mercado}/Titulos/{Simbolo}/Cotizacion
        """
        return self._get(f"/{mercado}/Titulos/{simbolo}/Cotizacion")

    # ── 4. Detalle de cotización ──────────────────────────────────────────
    def get_cotizacion_detalle(self, mercado: str, simbolo: str) -> Optional[dict]:
        """
        GET /api/v2/{mercado}/Titulos/{simbolo}/CotizacionDetalle
        """
        return self._get(f"/{mercado}/Titulos/{simbolo}/CotizacionDetalle")

    # ── 5. Serie histórica ────────────────────────────────────────────────
    def get_serie_historica(self, simbolo: str, fecha_desde: str, fecha_hasta: str,
                             ajustada: str = "ajustada",
                             mercado: str = "bCBA") -> pd.DataFrame:
        """
        GET /api/v2/{mercado}/Titulos/{simbolo}/Cotizacion/seriehistorica/
                {fechaDesde}/{fechaHasta}/{ajustada}
        """
        if not self._ensure_token():
            return pd.DataFrame()

        # ✅ Validar parámetros
        debug_lines = ["🔍 Validación de parámetros:"]
        
        if not simbolo or not str(simbolo).strip():
            st.error("❌ El símbolo está vacío")
            return pd.DataFrame()
        
        if not mercado or not str(mercado).strip():
            st.error("❌ El mercado está vacío")
            return pd.DataFrame()
        
        simbolo = str(simbolo).strip().upper()
        mercado = str(mercado).strip()
        
        debug_lines.append(f"   Símbolo: '{simbolo}'")
        debug_lines.append(f"   Mercado: '{mercado}'")
        debug_lines.append(f"   Desde: '{fecha_desde}'")
        debug_lines.append(f"   Hasta: '{fecha_hasta}'")
        debug_lines.append(f"   Ajuste: '{ajustada}'")
        debug_lines.append("")

        # ✅ Convertir fechas correctamente
        try:
            d_desde = datetime.strptime(fecha_desde, "%Y-%m-%d")
            d_hasta = datetime.strptime(fecha_hasta, "%Y-%m-%d")
            
            if d_desde > d_hasta:
                st.error("❌ La fecha 'Desde' no puede ser mayor que 'Hasta'")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"❌ Error en fechas: {e}")
            return pd.DataFrame()

        # ✅ Formato que usa IOL: dd-MM-yyyy
        fmt_desde = d_desde.strftime("%d-%m-%Y")
        fmt_hasta = d_hasta.strftime("%d-%m-%Y")

        # ✅ Construir endpoint completo
        endpoint = f"/{mercado}/Titulos/{simbolo}/Cotizacion/seriehistorica/{fmt_desde}/{fmt_hasta}/{ajustada}"
        url = f"{API_URL}{endpoint}"
        
        debug_lines.append(f"📡 URL construida:")
        debug_lines.append(f"   {url}")
        debug_lines.append(f"   Token: {'✓' if self._token else '✗'}")
        debug_lines.append("")

        try:
            debug_lines.append(f"🔄 Enviando petición...")
            resp = requests.get(url, headers=self.headers, timeout=20)
            debug_lines.append(f"   Status: {resp.status_code}")

            if resp.status_code == 401:
                debug_lines.append("   ⚠️ Token expirado, reintentando...")
                self._token = None
                if self.authenticate():
                    resp = requests.get(url, headers=self.headers, timeout=20)
                    debug_lines.append(f"   Reintentó status: {resp.status_code}")
                else:
                    debug_lines.append("   ❌ No se pudo renovar el token")
                    with st.expander("🔍 Debug – Serie histórica (sin datos)", expanded=True):
                        st.code("\n".join(debug_lines))
                    return pd.DataFrame()

            if resp.status_code == 400:
                debug_lines.append(f"   ❌ Error 400: Request inválido")
                try:
                    error_data = resp.json()
                    debug_lines.append(f"   Detalle: {error_data}")
                except:
                    debug_lines.append(f"   Response: {resp.text[:200]}")
                
                with st.expander("🔍 Debug – Serie histórica (sin datos)", expanded=True):
                    st.code("\n".join(debug_lines))
                return pd.DataFrame()
                
            elif resp.status_code != 200:
                debug_lines.append(f"   ❌ Error HTTP {resp.status_code}")
                debug_lines.append(f"   Response: {resp.text[:200]}")
                with st.expander("🔍 Debug – Serie histórica (sin datos)", expanded=True):
                    st.code("\n".join(debug_lines))
                return pd.DataFrame()

            # ✅ Parsear respuesta
            data = resp.json()
            debug_lines.append(f"   → Tipo respuesta: {type(data).__name__}")
            
            if isinstance(data, list):
                debug_lines.append(f"   → Cantidad de registros: {len(data)}")
            elif isinstance(data, dict):
                debug_lines.append(f"   → Claves: {list(data.keys())}")

            # Normalizar estructura
            if isinstance(data, dict):
                data = data.get("cotizaciones",
                       data.get("data",
                       data.get("items",
                       data.get("historico", []))))

            if not data or (isinstance(data, list) and len(data) == 0):
                debug_lines.append("   ⚠️ Respuesta vacía")
                with st.expander("🔍 Debug – Serie histórica (sin datos)", expanded=True):
                    st.code("\n".join(debug_lines))
                return pd.DataFrame()

            # ✅ Crear DataFrame
            df = pd.DataFrame(data)
            debug_lines.append(f"   → Columnas: {list(df.columns)}")
            debug_lines.append(f"   → {len(df)} filas obtenidas ✅")

            # Procesar fecha
            fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
            if fecha_col:
                df[fecha_col] = pd.to_datetime(df[fecha_col], format="ISO8601")
                df.set_index(fecha_col, inplace=True)
                df.index = df.index.normalize()

            # Procesar precio
            precio_col = next((c for c in df.columns
                               if any(p in c.lower()
                                      for p in ["ultimo", "cierre", "close", "precio"])), None)
            if precio_col:
                df[precio_col] = pd.to_numeric(df[precio_col], errors="coerce")
                if precio_col != "ultimoPrecio":
                    df.rename(columns={precio_col: "ultimoPrecio"}, inplace=True)

            with st.expander("🔍 Debug – Serie histórica"):
                st.code("\n".join(debug_lines))

            return df.sort_index()

        except Exception as e:
            debug_lines.append(f"   ❌ Excepción: {type(e).__name__}: {e}")
            with st.expander("🔍 Debug – Serie histórica (sin datos)", expanded=True):
                st.code("\n".join(debug_lines))
            return pd.DataFrame()

    # ── 6. FCI – todos los fondos ─────────────────────────────────────────
    def get_fci_todos(self) -> pd.DataFrame:
        """GET /api/v2/Titulos/FCI"""
        data = self._get("/Titulos/FCI")
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    # ── 7. FCI – detalle de un fondo ──────────────────────────────────────
    def get_fci_simbolo(self, simbolo: str) -> Optional[dict]:
        """GET /api/v2/Titulos/FCI/{simbolo}"""
        return self._get(f"/Titulos/FCI/{simbolo}")

    # ── 8. FCI – tipos de fondos ──────────────────────────────────────────
    def get_fci_tipos(self) -> list:
        """GET /api/v2/Titulos/FCI/TipoFondos"""
        data = self._get("/Titulos/FCI/TipoFondos")
        return data if isinstance(data, list) else []

    # ── 9. FCI – administradoras ──────────────────────────────────────────
    def get_fci_administradoras(self) -> list:
        """GET /api/v2/Titulos/FCI/Administradoras"""
        data = self._get("/Titulos/FCI/Administradoras")
        return data if isinstance(data, list) else []

    # ── 10. FCI por administradora y tipo ─────────────────────────────────
    def get_fci_por_admin_tipo(self, administradora: str, tipo_fondo: str) -> pd.DataFrame:
        """GET /api/v2/Titulos/FCI/Administradoras/{administradora}/TipoFondos/{tipoFondo}"""
        data = self._get(f"/Titulos/FCI/Administradoras/{administradora}/TipoFondos/{tipo_fondo}")
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    # ── 11. MEP ───────────────────────────────────────────────────────────
    def get_mep(self, simbolo: str) -> Optional[dict]:
        """GET /api/v2/Cotizaciones/MEP/{simbolo}"""
        return self._get(f"/Cotizaciones/MEP/{simbolo}")

    # ── 12. Instrumentos por país (lista de paneles) ───────────────────────
    def get_instrumentos_pais(self, pais: str = "argentina") -> pd.DataFrame:
        """GET /api/v2/{pais}/Titulos/Cotizacion/Instrumentos"""
        data = self._get(f"/{pais}/Titulos/Cotizacion/Instrumentos")
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    # ── 13. Paneles por instrumento ───────────────────────────────────────
    def get_paneles_instrumento(self, instrumento: str,
                                 pais: str = "argentina") -> pd.DataFrame:
        """GET /api/v2/{pais}/Titulos/Cotizacion/Paneles/{instrumento}"""
        data = self._get(f"/{pais}/Titulos/Cotizacion/Paneles/{instrumento}")
        if data is None:
            return pd.DataFrame()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()

    # ── Parser genérico de cotizaciones ───────────────────────────────────
    @staticmethod
    def _parse_cotizaciones(data) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        # La API puede devolver una lista directa o un dict con clave "titulos"
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("titulos", data.get("items", [data]))
        else:
            return pd.DataFrame()
        if not items:
            return pd.DataFrame()

        df = pd.DataFrame(items)

        # Renombrar columnas comunes para uniformidad
        rename_map = {
            "simbolo":       "Símbolo",
            "descripcion":   "Descripción",
            "ultimoPrecio":  "Último",
            "variacion":     "Variación %",
            "apertura":      "Apertura",
            "maximo":        "Máximo",
            "minimo":        "Mínimo",
            "volumen":       "Volumen",
            "cantidadOperaciones": "Operaciones",
            "fechaHora":     "Fecha/Hora",
            "montoOperado":  "Monto Operado",
            "puntasCompra":  "Compra",
            "puntasVenta":   "Venta",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        # Convertir numéricos
        for col in ["Último", "Variación %", "Apertura", "Máximo", "Mínimo", "Volumen"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ═══════════════════════════════════════════════════════════════════════════
#  CACHÉ EN SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def get_iol_client() -> Optional[IOLClient]:
    """
    Devuelve el cliente IOL autenticado desde session_state.
    Retorna None si no hay credenciales configuradas.
    """
    username = st.session_state.get("iol_username", "").strip()
    password = st.session_state.get("iol_password", "").strip()
    if not username or not password:
        return None

    # Reutilizar cliente existente si las credenciales no cambiaron
    existing = st.session_state.get("iol_client")
    if existing and existing.username == username:
        return existing

    client = IOLClient(username, password)
    if client.authenticate():
        st.session_state.iol_client = client
        return client
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA DE EXPLORADOR IOL
# ═══════════════════════════════════════════════════════════════════════════

def page_iol_explorer():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');
    .iol-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.9rem;
                 background:linear-gradient(90deg,#60A5FA,#34D399); -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent; }
    .iol-sub   { font-family:'Space Mono',monospace; font-size:.75rem; color:#6B7280; letter-spacing:.06em; }
    .tag-add   { background:#1E3A5F; color:#93C5FD; border-radius:4px; padding:2px 10px;
                 font-size:.75rem; cursor:pointer; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="iol-title">🏦 Explorador IOL – API Oficial</p>', unsafe_allow_html=True)
    st.markdown('<p class="iol-sub">ACCIONES · CEDEARS · BONOS · FCI · OPCIONES · SERIE HISTÓRICA</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    client = get_iol_client()
    if not client:
        st.warning("⚠️ Ingresá tus credenciales de IOL en la barra lateral (sección 🏦 IOL API).")
        return

    st.success("✅ Conectado a IOL API")

    tabs = st.tabs([
        "📊 Cotizaciones",
        "💼 FCI",
        "📈 Serie Histórica",
        "💵 MEP",
        "➕ Agregar a Portafolio"
    ])

    # ── Tab 1: Cotizaciones ──────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Cotizaciones por instrumento")
        c1, c2, c3 = st.columns(3)
        with c1:
            instrumento = st.selectbox("Instrumento", list(INSTRUMENTOS.keys()), key="cot_inst")
        with c2:
            panel_sel = st.selectbox("Panel", ["Todos"] + list(PANELES.keys())[1:], key="cot_panel")
        with c3:
            pais_sel = st.selectbox("País", list(PAISES.keys()), key="cot_pais")

        if st.button("🔄 Cargar cotizaciones", key="btn_cot"):
            with st.spinner("Consultando IOL..."):
                inst_val  = INSTRUMENTOS[instrumento]
                pais_val  = PAISES[pais_sel]
                panel_val = PANELES.get(panel_sel, "Todos")

                if panel_val == "Todos":
                    df = client.get_cotizaciones_todos(inst_val, pais_val)
                else:
                    df = client.get_cotizaciones_panel(inst_val, panel_val, pais_val)

            if df.empty:
                st.warning("Sin datos. Verificá el instrumento/panel seleccionado.")
            else:
                st.session_state["iol_last_df"]   = df
                st.session_state["iol_last_label"] = instrumento
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"{len(df)} activos encontrados")

                # Descarga CSV
                csv = df.to_csv(index=False, sep=";").encode("utf-8")
                st.download_button("📥 Descargar CSV", csv,
                                   file_name=f"iol_{instrumento}_{pais_sel}.csv",
                                   mime="text/csv")

    # ── Tab 2: FCI ───────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Fondos Comunes de Inversión")
        fci_mode = st.radio("Ver", ["Todos los FCI", "Por administradora y tipo"], horizontal=True)

        if fci_mode == "Todos los FCI":
            if st.button("🔄 Cargar FCI", key="btn_fci"):
                with st.spinner("Consultando IOL..."):
                    df_fci = client.get_fci_todos()
                if df_fci.empty:
                    st.warning("Sin datos de FCI.")
                else:
                    st.session_state["iol_fci_df"] = df_fci
                    st.dataframe(df_fci, use_container_width=True, hide_index=True)
                    csv = df_fci.to_csv(index=False, sep=";").encode("utf-8")
                    st.download_button("📥 CSV", csv, file_name="iol_fci.csv", mime="text/csv")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                admins = client.get_fci_administradoras()
                admin_sel = st.selectbox("Administradora", admins if admins else ["—"])
            with col_b:
                tipos = client.get_fci_tipos()
                tipo_sel = st.selectbox("Tipo de fondo", tipos if tipos else ["—"])

            if st.button("🔄 Cargar", key="btn_fci2") and admin_sel != "—":
                with st.spinner("Consultando..."):
                    df_fci2 = client.get_fci_por_admin_tipo(admin_sel, tipo_sel)
                if not df_fci2.empty:
                    st.dataframe(df_fci2, use_container_width=True, hide_index=True)

    # ── Tab 3: Serie Histórica ───────────────────────────────────────────
    with tabs[2]:
        st.subheader("Serie histórica de precios (IOL)")

        st.info("""
        **Símbolos válidos en BCBA:** AL30, GD30, GD35, AE38, AL41, GGAL, BMA, PAMP, YPF, TXAR  
        **Nota:** Usá el símbolo **sin** sufijos como D/C (ej: `AE38` no `AE38D`)
        """)

        c1, c2 = st.columns(2)
        with c1:
            simbolo_hist = st.text_input("Símbolo", "AL30", key="hist_sim",
                                          help="Ej: AL30, GD30, GGAL, BMA, PAMP — sin sufijo D/C")
        with c2:
            mercado_hist = st.selectbox("Mercado", list(MERCADOS.keys()), key="hist_merc")

        c3, c4, c5 = st.columns(3)
        with c3:
            desde = st.date_input("Desde", value=datetime(2023, 1, 1), key="hist_desde")
        with c4:
            hasta = st.date_input("Hasta", value=datetime.today(), key="hist_hasta")
        with c5:
            ajustada = st.selectbox("Ajuste", ["ajustada", "sinAjustar"], key="hist_ajuste")

        col_test, col_get = st.columns(2)

        with col_test:
            if st.button("🔎 Verificar símbolo", key="btn_test_sim"):
                with st.spinner("Consultando cotización actual..."):
                    merc_val = MERCADOS[mercado_hist]
                    cot = client.get_cotizacion(merc_val, simbolo_hist)
                if cot:
                    st.success(f"✅ Símbolo válido | Último precio: {cot.get('ultimoPrecio','—')}")
                    st.json(cot)
                else:
                    st.error("❌ Símbolo no encontrado. Probá sin sufijos (AE38 en vez de AE38D)")

        with col_get:
            if st.button("📈 Obtener serie", key="btn_hist"):
                # ✅ Validar antes de llamar
                if not simbolo_hist.strip():
                    st.error("❌ Ingrese un símbolo válido")
                elif desde >= hasta:
                    st.error("❌ La fecha 'Desde' debe ser menor que 'Hasta'")
                else:
                    with st.spinner("Descargando serie histórica..."):
                        df_hist = client.get_serie_historica(
                            simbolo_hist.strip().upper(),  # ✅ Mayúsculas y sin espacios
                            str(desde), 
                            str(hasta), 
                            ajustada,
                            mercado=MERCADOS[mercado_hist]
                        )
                    if df_hist.empty:
                        st.warning("Sin datos. Usá **Verificar símbolo** primero para confirmar que existe.")
                    else:
                        st.session_state["iol_hist_df"]      = df_hist
                        st.session_state["iol_hist_simbolo"] = simbolo_hist
                        precio_col = "ultimoPrecio" if "ultimoPrecio" in df_hist.columns else df_hist.columns[0]
                        st.line_chart(df_hist[precio_col])
                        st.dataframe(df_hist, use_container_width=True)
                        csv = df_hist.to_csv(sep=";").encode("utf-8")
                        st.download_button("📥 CSV", csv,
                                           file_name=f"hist_{simbolo_hist}.csv", mime="text/csv")

    # ── Tab 4: MEP ───────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Cotización Dólar MEP")
        sim_mep = st.text_input("Símbolo bono para MEP", "AL30", key="mep_sim")
        if st.button("💵 Consultar MEP", key="btn_mep"):
            with st.spinner("Consultando..."):
                mep = client.get_mep(sim_mep)
            if mep:
                st.json(mep)
            else:
                st.warning("Sin datos MEP.")

    # ── Tab 5: Agregar a Portafolio ──────────────────────────────────────
    with tabs[4]:
        st.subheader("➕ Agregar activos IOL a un portafolio")
        st.markdown("Seleccioná activos de la tabla de cotizaciones cargada y agregalos a un portafolio.")

        df_last = st.session_state.get("iol_last_df", pd.DataFrame())

        if df_last.empty:
            st.info("Primero cargá cotizaciones en la pestaña **📊 Cotizaciones**.")
        else:
            sym_col = "Símbolo" if "Símbolo" in df_last.columns else df_last.columns[0]
            simbolos_disponibles = df_last[sym_col].dropna().tolist()

            selected_simbolos = st.multiselect(
                "Seleccioná los activos a agregar",
                simbolos_disponibles,
                key="add_simbolos"
            )

            if selected_simbolos:
                # Pesos
                st.markdown("**Asigná pesos (deben sumar 1.0)**")
                peso_total = 0.0
                pesos_nuevos = {}
                cols = st.columns(min(len(selected_simbolos), 4))
                for i, sim in enumerate(selected_simbolos):
                    with cols[i % 4]:
                        w = st.number_input(f"{sim}", min_value=0.0, max_value=1.0,
                                            value=round(1/len(selected_simbolos), 2),
                                            step=0.01, key=f"w_{sim}")
                        pesos_nuevos[sim] = w
                        peso_total += w

                st.metric("Suma de pesos ingresados", f"{peso_total:.3f}",
                          delta=f"{peso_total - 1:.3f}" if abs(peso_total - 1) > 0.001 else "✅ OK")

                # Portafolio destino
                portfolios = st.session_state.get("portfolios", {})
                port_opts  = list(portfolios.keys()) + ["➕ Nuevo portafolio"]
                dest_port  = st.selectbox("Portafolio destino", port_opts, key="dest_port")

                nuevo_nombre = ""
                if dest_port == "➕ Nuevo portafolio":
                    nuevo_nombre = st.text_input("Nombre del nuevo portafolio", key="new_port_name")

                if st.button("💾 Guardar en portafolio", key="btn_save_port"):
                    if abs(peso_total - 1.0) > 0.01:
                        st.error("⚠️ Los pesos deben sumar 1.0")
                    else:
                        nombre_final = nuevo_nombre if dest_port == "➕ Nuevo portafolio" else dest_port
                        if not nombre_final:
                            st.error("⚠️ Ingresá un nombre para el portafolio.")
                        else:
                            # ✅ Preparar datos del portafolio
                            if nombre_final in portfolios:
                                # Agregar a existente (reemplaza tickers duplicados)
                                existing_tickers = portfolios[nombre_final]["tickers"]
                                existing_weights = portfolios[nombre_final]["weights"]
                                ticker_dict = dict(zip(existing_tickers, existing_weights))
                                ticker_dict.update(pesos_nuevos)
                                # Renormalizar
                                total_w = sum(ticker_dict.values())
                                ticker_dict = {k: round(v/total_w, 4) for k, v in ticker_dict.items()}
                                portfolios[nombre_final]["tickers"] = list(ticker_dict.keys())
                                portfolios[nombre_final]["weights"] = list(ticker_dict.values())
                            else:
                                portfolios[nombre_final] = {
                                    "tickers": list(pesos_nuevos.keys()),
                                    "weights": list(pesos_nuevos.values()),
                                    "fuente":  "IOL"
                                }

                            # ✅ ACTUALIZAR SESSION STATE
                            st.session_state.portfolios = portfolios

                            # ✅ IMPORT CONDICIONAL (evita ModuleNotFoundError)
                            try:
                                from app import save_portfolios_to_file
                                ok, msg = save_portfolios_to_file(portfolios)
                                if ok:
                                    st.success(f"✅ Portafolio **{nombre_final}** guardado con {len(pesos_nuevos)} activos IOL.")
                                    st.balloons()
                                else:
                                    st.warning(f"⚠️ Guardado en sesión, pero error al guardar en archivo: {msg}")
                            except ImportError:
                                # Si no encuentra app.py, solo guarda en sesión
                                st.success(f"✅ Portafolio **{nombre_final}** guardado en sesión temporal.")
                                st.info("📝 Los datos se perderán al cerrar la app. Para guardado permanente, aseguráte que app.py esté disponible.")
                            except Exception as e:
                                st.warning(f"⚠️ Guardado en sesión. Error al guardar en archivo: {e}")
