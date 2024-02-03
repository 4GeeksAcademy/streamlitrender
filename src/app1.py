import streamlit as st
import pickle

with open('models/normscaler.pk', "rb") as openfile:
    scaler_model = pickle.load(openfile)
with open("models/linealmodel.pk", "rb") as openfile:
    model = pickle.load(openfile)

def main():
    
    # Aplicar text-align: center y color al contenedor principal de la página
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)

    # Centrar el título y cambiar el color utilizando HTML
    st.markdown("<h1 style='color: #9D1B28;'>MODELO DE REGRESION LINEAL</h1>", unsafe_allow_html=True)

    st.write("En este ejercicio hemos entrenado un modelo de machine learning y lo hemos representado en Streamlit. Pulsa el botón para realizar una predicción.")

    placeholder = st.empty()

    if st.button("Ir a la Página de Predicción"):
        st.session_state.pagina_actual = "pagina_2"

    placeholder.text(" ")
   

def pagina_2():
    st.title("Realiza tu Predicción")

    st.write("Completa la información para obtener una predicción de costos médicos.")

    age_val = st.slider("Por favor, ingresa tu edad", min_value=18, max_value=64, step=1, value=30)
    bmi_val = st.slider("Ingresa tu BMI", min_value=15.0, max_value=53.0, step=0.01, value=25.0)
    child_val = st.slider("Número de hijos", min_value=0, max_value=5, step=1, value=1)
    sex_val = st.selectbox('Ingresa tu género', ("Male", "Female", "None"), index=0)
    smoker_val = st.selectbox("¿Eres fumador?", ("Yes", "No"), index=1)
    region_val = st.selectbox("Ingresa tu región", ("Southwest", "Southeast", "Northwest", "Northeast"), index=0)

    if st.button("Predecir"):
        sex_parse_dict = {"Male": 0, "Female": 1}
        smoker_parse_dict = {"Yes": 0, "No": 1}
        region_parse_dict = {"Southwest": 0, "Southeast": 1, "Northwest": 2, "Northeast": 3}

        row = [age_val, bmi_val, child_val, sex_parse_dict[sex_val], smoker_parse_dict[smoker_val], region_parse_dict[region_val]]
        scaled_row = scaler_model.transform([row])
        predicted_cost = model.predict(scaled_row)[0]

        st.success(f"La predicción de costos médicos es de: ${predicted_cost:.2f}")

    if st.button("Volver a la Página Principal"):
        st.session_state.pagina_actual = "main"

# Configuración inicial del estado de la aplicación
if "pagina_actual" not in st.session_state:
    st.session_state.pagina_actual = "main"

# Lógica de navegación basada en el estado actual
if st.session_state.pagina_actual == "main":
    main()
elif st.session_state.pagina_actual == "pagina_2":
    pagina_2()


