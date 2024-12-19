# Importando bibliotecas necessárias
import geopandas as gpd 
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
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
def carregar_dados_geo():

    # Carrega os dados geoespaciais de um arquivo Parquet
    gdf_geo = gpd.read_parquet(DADOS_GEO_MEDIAN)

    # Explode MultiPolygons em polígonos individuais
    gdf_geo = gdf_geo.explode(ignore_index=True)

    # Função para verificar e corrigir geometrias inválidas
    def fix_and_orient_geometry(geometry):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)  # Corrige a geometria inválida
        # Orienta o polígono para ser no sentido anti-horário se for um Polígono ou MultiPolígono
        if isinstance(geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        return geometry

    # Aplica a função de correção e orientação às geometrias
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fix_and_orient_geometry)

    # Função para extrair coordenadas de polígonos
    def get_polygon_coordinates(geometry):
        return (
            [[[x, y] for x, y in geometry.exterior.coords]]
            if isinstance(geometry, shapely.geometry.Polygon)
            else [
                [[x, y] for x, y in polygon.exterior.coords]
                for polygon in geometry.geoms
            ]
        )

    # Aplica a conversão de coordenadas e armazena em uma nova coluna
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(get_polygon_coordinates)

    return gdf_geo


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
condados = sorted(list(gdf_geo["name"].unique())) # Lista de condados ordenada alfabeticamente

coluna1, coluna2 = st.columns(2) # Dividindo a tela em duas colunas

# Coluna com campos de entrada de dados
with coluna1:

    with st.form(key="formulario"): # Formulário para entrada de dados

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
        df_entrada_modelo = pd.DataFrame(entrada_modelo)

        # Criando um botão para fazer a previsão
        botao_previsao = st.form_submit_button("Prever preço")

        if botao_previsao:
            # Fazendo a previsão com o modelo carregado
            preco_previsto = modelo.predict(df_entrada_modelo)
            st.metric(label="Preço previsto: (US$)", value=f"{preco_previsto[0][0]:.2f}")

# Coluna com mapa
with coluna2:

    # Configurando o mapa
    view_state = pdk.ViewState(
        latitude=float(latitude[0]),  # Define a latitude inicial do mapa
        longitude=float(longitude[0]),  # Define a longitude inicial do mapa
        zoom=5,  # Define o nível de zoom inicial
        min_zoom=5,  # Define o nível de zoom mínimo permitido
        max_zoom=15,  # Define o nível de zoom máximo permitido
    )

    # Criando uma camada de polígonos
    polygon_layer = pdk.Layer(
        "PolygonLayer",  # Tipo da camada
        data=gdf_geo[["name", "geometry"]],  # Dados da camada, contendo nome e geometria
        get_polygon="geometry",  # Define a coluna que contém os polígonos
        get_fill_color=[0, 0, 255, 100],  # Cor de preenchimento dos polígonos (azul com transparência)
        get_line_color=[255, 255, 255],  # Cor das linhas dos polígonos (branco)
        get_line_width=50,  # Largura das linhas dos polígonos
        pickable=True,  # Permite que os polígonos sejam clicáveis
        auto_highlight=True,  # Destaca automaticamente os polígonos ao passar o mouse
    )

    # Filtrando o condado selecionado
    condado_selecionado = gdf_geo.query("name == @selecionar_condado")

    # Destacando o condado selecionado
    highlight_layer = pdk.Layer(
        "PolygonLayer",  # Tipo da camada
        data=condado_selecionado[["name", "geometry"]],  # Dados do condado selecionado
        get_polygon="geometry",  # Define a coluna que contém os polígonos
        get_fill_color=[255, 0, 0, 100],  # Cor de preenchimento do condado selecionado (vermelho com transparência)
        get_line_color=[0, 0, 0],  # Cor das linhas do condado selecionado (preto)
        get_line_width=500,  # Largura das linhas do condado selecionado
        pickable=True,  # Permite que o condado selecionado seja clicável
        auto_highlight=True,  # Destaca automaticamente o condado selecionado ao passar o mouse
    )

    # Configurando o tooltip (dica de ferramenta) para exibir o nome do condado
    tooltip = {
        "html": "<b>Condado:</b> {name}",  # HTML para exibir o nome do condado
        "style": {"backgroundColor": "steelblue", "color": "white", "fontsize": "10px"},  # Estilo do tooltip
    }

    # Criando o mapa
    mapa = pdk.Deck(
        initial_view_state=view_state,  # Define o estado inicial da visualização do mapa
        map_style="light",  # Define o estilo do mapa (claro)
        layers=[polygon_layer, highlight_layer],  # Adiciona as camadas de polígonos e destaque
        tooltip=tooltip,  # Adiciona o tooltip ao mapa
    )

    # Exibindo o mapa no Streamlit
    st.pydeck_chart(mapa)  # Renderiza o mapa usando o componente pydeck_chart do Streamlit