import pandas as pd

# Função para criar um DataFrame com os coeficientes do modelo e os nomes das colunas
def dataframe_coeficientes(coefs, colunas):
    # Cria um DataFrame com os coeficientes e os nomes das colunas
    return pd.DataFrame(data=coefs, index=colunas, columns=["coeficiente"]).sort_values(
        by="coeficiente"  # Ordena o DataFrame pelos valores dos coeficientes em ordem crescente
    )