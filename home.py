# Importando bibliotecas necessárias
import geopandas as gpd 
import numpy as np
import pandas as pd
import streamlit as st

# Função para carregar modelos salvos
from joblib import load  

# Importando constantes de configuração
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

# Função para carregar dados limpos, com cache para evitar recarregamento desnecessário
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)


# Função para carregar dados geoespaciais, com cache para evitar recarregamento desnecessário
@st.cache_data
def carregar_dados_geo():
    return gpd.read_parquet(DADOS_GEO_MEDIAN)


# Função para carregar o modelo de machine learning, com cache para evitar recarregamento desnecessário
@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)


# Carregando os dados e o modelo
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

# Definindo o título da aplicação Streamlit
st.title("Previsão de preços de imóveis")

# Criando campos para entrada de dados
longitude = st.number_input("Longitude", value=-122.33)
latitude = st.number_input("Latitude", value=37.88)

housing_median_age = st.number_input("Idade do imóvel", value=10)

total_rooms = st.number_input("Total de cômodos", value=800)
total_bedrooms = st.number_input("Total de quartos", value=100)
population = st.number_input("População", value=300)
households = st.number_input("Domicílios", value=100)

median_income = st.slider("Renda média (múltiplos de US$ 10k)", 0.5, 15.0, 4.5, 0.5)

ocean_proximity = st.selectbox("Proximidade do oceano", df["ocean_proximity"].unique())

median_income_cat = st.number_input("Categoria de renda", value=4)

rooms_per_household = st.number_input("Quartos por domicílio", value=7)
bedrooms_per_room = st.number_input("Quartos por cômodo", value=0.2)
population_per_household = st.number_input("Pessoas por domicílio", value=2)

# Criando um dicionário com os dados de entrada
entrada_modelo = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "median_income_cat": median_income_cat,
    "rooms_per_household": rooms_per_household,
    "bedrooms_per_room": bedrooms_per_room,
    "population_per_household": population_per_household
}

# Criando um DataFrame com os dados de entrada
df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])

# Criando um botão para fazer a previsão
botao_previsao = st.button("Prever preço")

if botao_previsao:
    # Fazendo a previsão com o modelo carregado
    preco_previsto = modelo.predict(df_entrada_modelo)
    st.write(f"Preço previsto: US$ {preco_previsto[0][0]:.2f}")