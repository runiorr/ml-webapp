import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Desenvolvedor - Predição salarial")

    st.write("""#### Preenche o formulário abaixo""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    COUNTRY_CHOICES = {"United States": "Estados Unidos",
        "India": "Índia",
        "United Kingdom": "Reino Unido",
        "Germany": "Alemanha",
        "Canada": "Canadá",
        "Brazil": "Brasil",
        "France": "França",
        "Spain": "Espanha",
        "Australia": "Austrália",
        "Netherlands": "Holanda",
        "Poland": "Polônia",
        "Italy": "Itália",
        "Russian Federation": "Rússia",
        "Sweden": "Suécia"}

    def format_country(option):
        return COUNTRY_CHOICES[option]

    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree",
        "Post grad",
    )

    EDUCATION_CHOICES = {
        "Less than a Bachelors": "Médio",
        "Bachelor's degree": "Bacharelado",
        "Master's degree": "Mestrado",
        "Post grad": "Doutorado"
    }

    def format_education(option):
        return EDUCATION_CHOICES[option]

    country = st.selectbox("Países", options=list(COUNTRY_CHOICES.keys()), format_func=format_country)
    education = st.selectbox("Educação", options=list(EDUCATION_CHOICES.keys()), format_func=format_education)

    experience = st.slider("Anos de experiência", 0, 50, 3)

    ok = st.button("Calcule o salário")
    if ok:
        x = np.array([[country, education, experience]])
        x[:, 0] = le_country.transform(x[:,0])
        x[:, 1] = le_education.transform(x[:,1])
        x = x.astype(float)

        salary = regressor.predict(x)
        st.subheader(f"O salário estimado é ${salary[0]:.2f}")

