import streamlit as st
import pandas as pd
import joblib
 
# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Super Predictor XLogP",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. GESTIÓN DE NAVEGACIÓN (Estado de la sesión) ---
# Esto sirve para recordar si el usuario ya pasó la pantalla de bienvenida
if 'etapa' not in st.session_state:
    st.session_state.etapa = 'bienvenida'

# Función para reiniciar la predicción sin salir de la app
def reiniciar():
    pass

# --- 3. PANTALLA DE BIENVENIDA ---
if st.session_state.etapa == 'bienvenida':
    st.markdown("<h1 style='text-align: center;'>Super Predictor XLogP 🧬</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Inteligencia Artificial para Química Computacional</h3>", unsafe_allow_html=True)
    
    st.write("") 
    st.write("") 
    
    # Columnas para centrar el botón
    col_centro = st.columns([1, 2, 1])
    with col_centro[1]:
        if st.button("Entrar al Sistema", type="primary", use_container_width=True):
            st.session_state.etapa = 'app_principal'
            st.rerun()

# --- 4. APLICACIÓN PRINCIPAL ---
elif st.session_state.etapa == 'app_principal':
    
    # --- A. CARGA DE ARCHIVOS ---
    try:
        # Cargamos el cerebro (modelo), la memoria (columnas) y las notas (métricas)
        model = joblib.load("modelo_final.pkl")
        columnas = joblib.load("columnas_entrenamiento.pkl")
        metricas = joblib.load("metricas_modelo.pkl") 
        
        # Extraer valores numéricos guardados desde Colab
        r2_val = metricas['r2']
        rmsd_val = metricas['rmsd']
        
    except Exception as e:
        st.error(f"⚠️ Error Crítico: No se encuentran los archivos del sistema.")
        st.warning(f"Detalle del error: {e}")
        st.info("Asegúrate de haber subido 'modelo_final.pkl', 'columnas_entrenamiento.pkl' y 'metricas_modelo.pkl' al repositorio.")
        st.stop()

    # --- B. TEXTO INTRODUCTORIO Y CONTEXTO ---
    st.title("Super Predictor XLogP")
    
    st.info(f"""
    **Predictor de lipofilia de moléculas a partir de fórmula molecular.**
    
    Este sistema utiliza un modelo de **Gradient Boosting** entrenado con **6,500 moléculas** análogas a la aspirina.
    
    📊 **Métricas de Precisión del Modelo:**
    - Coeficiente R²: **{r2_val:.4f}**
    - Error RMSD: **{rmsd_val:.4f}**
    
    Las estructuras fueron extraídas de la base de datos oficial [PubChem](https://pubchem.ncbi.nlm.nih.gov/).
    """)

    st.markdown("---")
    st.subheader("🧪 Composición Molecular")
    st.write("Ingrese la cantidad de átomos presentes en su fórmula:")

    # --- C. FORMULARIO DE ÁTOMOS ---
    inputs_usuario = {}
    
    # Definimos los átomos principales que siempre deben verse
    atomos_comunes = ['C', 'H', 'O', 'N']
    
    # Separamos las columnas del modelo en "Comunes" y "Otros"
    cols_comunes = [col for col in columnas if col in atomos_comunes]
    cols_otros = [col for col in columnas if col not in atomos_comunes]

    # 1. Mostrar C, H, O, N destacados en 4 columnas
    col1, col2, col3, col4 = st.columns(4)
    cols_ui = [col1, col2, col3, col4]
    
    for i, atomo in enumerate(atomos_comunes):
        # Solo mostramos el input si el modelo realmente usa ese átomo
        if atomo in columnas:
            with cols_ui[i]:
                inputs_usuario[atomo] = st.number_input(f"{atomo}", min_value=0, value=0, step=1)
    
    # 2. Mostrar Otros Átomos (Ocultos en un desplegable por defecto)
    if cols_otros:
        st.write("")
        with st.expander("➕ Agregar otros átomos (S, F, Cl, etc.)"):
            st.write("Seleccione los elementos adicionales presentes en la molécula:")
            
            # Multiselect para elegir qué inputs mostrar
            elementos_extra = st.multiselect("Elementos disponibles:", cols_otros)
            
            if elementos_extra:
                st.write("Indique la cantidad de átomos:")
                c1, c2 = st.columns(2)
                for idx, col in enumerate(elementos_extra):
                    # Distribuir en 2 columnas para orden visual
                    with (c1 if idx % 2 == 0 else c2):
                        inputs_usuario[col] = st.number_input(f"{col}", min_value=0, value=0, step=1, key=f"input_{col}")
            
            # Rellenar con 0 los átomos que el usuario NO seleccionó (vital para el modelo)
            for col in cols_otros:
                if col not in inputs_usuario:
                    inputs_usuario[col] = 0

    # --- D. BOTÓN DE PREDICCIÓN ---
    st.write("")
    st.write("")
    
    if st.button("🔮 Predecir XLogP", type="primary", use_container_width=True):
        
        try:
            # 1. Ordenar los datos: El modelo necesita las columnas en el orden EXACTO del entrenamiento
            datos_ordenados = {col: inputs_usuario.get(col, 0) for col in columnas}
            
            # 2. Crear DataFrame de una sola fila
            df_entrada = pd.DataFrame([datos_ordenados])
            
            # 3. Realizar la predicción
            prediccion = model.predict(df_entrada)[0]
            
            # 4. Mostrar Resultados
            st.markdown("---")
            st.success("✅ Cálculo Finalizado")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="Valor XLogP Predicho", value=f"{prediccion:.4f}")
            with col_res2:
                st.metric(label="Margen de Error (RMSD)", value=f"± {rmsd_val:.4f}")
            
            # Animación de celebración
            st.balloons()
            
            # Botón para limpiar y empezar de nuevo
            st.write("")
            if st.button("🔄 Predecir otra molécula"):
                st.rerun()
                
        except Exception as e:
            st.error(f"Ocurrió un error matemático al predecir: {e}")
