# Importando bibliotecas necessárias
import geopandas as gpd 
import numpy as np
import pandas as pd
import pydeck as pdk
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
condados = list(gdf_geo["name"].sort_values()) # Lista de condados ordenada alfabeticamente

coluna1, coluna2 = st.columns(2) # Dividindo a tela em duas colunas

# Coluna com campos de entrada de dados
with coluna1:

    selecionar_condado = st.selectbox("Condado", condados) # Campo de seleção de condado

    longitude = gdf_geo.query("name == @selecionar_condado")["longitude"].values # Longitude do condado selecionado
    latitude = gdf_geo.query("name == @selecionar_condado")["latitude"].values # Latitude do condado selecionado

    housing_median_age = st.number_input("Idade do imóvel", value=10, min_value=1, max_value=50) # Idade do imóvel

    total_rooms = gdf_geo.query("name == @selecionar_condado")["total_rooms"].values # Total de quartos do condado selecionado
    total_bedrooms = gdf_geo.query("name == @selecionar_condado")["total_bedrooms"].values # Total de quartos do condado selecionado
    population = gdf_geo.query("name == @selecionar_condado")["population"].values # População do condado selecionado
    households = gdf_geo.query("name == @selecionar_condado")["households"].values # Domicílios do condado selecionado

    median_income = st.slider("Renda média (milhares de US$)", 5.0, 100.0, 45.0, 5.0) # Renda média
    median_income_scale = median_income / 10 # Escala da renda média

    ocean_proximity = gdf_geo.query("name == @selecionar_condado")["ocean_proximity"].values # Proximidade do oceano

    bins_income = [0, 1.5, 3, 4.5, 6, np.inf] # Limites para categorização da renda média
    median_income_cat = np.digitize(median_income_scale, bins=bins_income) # Categorização da renda média

    rooms_per_household = gdf_geo.query("name == @selecionar_condado")["rooms_per_household"].values # Quartos por domicílio
    bedrooms_per_room = gdf_geo.query("name == @selecionar_condado")["bedrooms_per_room"].values # Quartos por cômodo
    population_per_household = gdf_geo.query("name == @selecionar_condado")\
        ["population_per_household"].values # População por domicílio

    # Criando um dicionário com os dados de entrada
    entrada_modelo = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income_scale,
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

# Coluna com mapa
with coluna2:

    # Configurando o mapa
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),
        longitude=float(longitude[0]),
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    # Criando o mapa
    mapa = pdk.Deck(
        initial_view_state=view_state,
        map_style="light",
    )

    # Adicionando camada de mapa
    st.pydeck_chart(mapa)