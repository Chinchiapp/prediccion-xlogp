import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(page_title="Calculadora XLogP", layout="centered")

st.title("🧪 Predicción de XLogP")
st.write("Ingresa los atributos moleculares para calcular el valor.")

# 1. Cargar el modelo y las columnas
try:
    model = joblib.load("modelo_final.pkl")
    columnas = joblib.load("columnas_entrenamiento.pkl")
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error cargando archivos: {e}")
    st.stop()

# 2. Formulario dinámico
inputs_usuario = {}

with st.form("formulario_prediccion"):
    st.subheader("Parámetros")
    
    # Crea una casilla por cada columna necesaria automáticamente
    for col in columnas:
        # Creamos un input numérico para cada atributo
        inputs_usuario[col] = st.number_input(f"{col}", value=0.0, format="%.4f")
    
    # Botón de acción
    enviado = st.form_submit_button("Calcular Resultado")

# 3. Predecir
if enviado:
    try:
        # Convertir a DataFrame
        df_entrada = pd.DataFrame([inputs_usuario])
        
        # Calcular
        prediccion = model.predict(df_entrada)[0]
        
        # Mostrar resultado grande
        st.markdown("---")
        st.metric(label="Valor XLogP Predicho", value=f"{prediccion:.4f}")
        
    except Exception as e:
        st.error(f"Ocurrió un error en el cálculo: {e}")
