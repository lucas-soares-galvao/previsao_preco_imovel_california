import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# Função para construir um pipeline de modelo de regressão
def construir_pipeline_modelo_regressao(
    regressor, preprocessor=None, target_transformer=None
):
    # Verifica se há um pré-processador fornecido
    if preprocessor is not None:
        # Cria um pipeline com o pré-processador e o regressor
        pipeline = Pipeline([("preprocessor", preprocessor), ("reg", regressor)])
    else:
        # Cria um pipeline apenas com o regressor
        pipeline = Pipeline([("reg", regressor)])

    # Verifica se há um transformador de alvo fornecido
    if target_transformer is not None:
        # Cria um TransformedTargetRegressor com o pipeline e o transformador de alvo
        model = TransformedTargetRegressor(
            regressor=pipeline, transformer=target_transformer
        )
    else:
        # Usa o pipeline diretamente como modelo
        model = pipeline
    return model

# Função para treinar e validar um modelo de regressão
def treinar_e_validar_modelo_regressao(
    X,
    y,
    regressor,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
):
    # Constrói o pipeline do modelo com o regressor, pré-processador e transformador de alvo
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Cria um objeto KFold para validação cruzada com n_splits divisões, embaralhando os dados
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Realiza a validação cruzada do modelo nos dados X e y
    scores = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=[
            "r2",  # Coeficiente de determinação R²
            "neg_mean_absolute_error",  # Erro absoluto médio negativo
            "neg_root_mean_squared_error",  # Raiz quadrada do erro quadrático médio negativo
        ],
    )

    # Retorna os scores da validação cruzada
    return scores

# Função para realizar uma busca em grade com validação cruzada para um regressor
def grid_search_cv_regressor(
    regressor,
    param_grid,
    preprocessor=None,
    target_transformer=None,
    n_splits=5,
    random_state=RANDOM_STATE,
    return_train_score=False,
):
    # Constrói o pipeline do modelo com o regressor, pré-processador e transformador de alvo
    model = construir_pipeline_modelo_regressao(
        regressor, preprocessor, target_transformer
    )

    # Cria um objeto KFold para validação cruzada com n_splits divisões, embaralhando os dados
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Cria um objeto GridSearchCV para realizar a busca em grade
    grid_search = GridSearchCV(
        model,
        cv=kf,
        param_grid=param_grid,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        return_train_score=return_train_score,
        verbose=1,
    )

    # Retorna o objeto GridSearchCV
    return grid_search

# Função para organizar os resultados da validação cruzada
def organiza_resultados(resultados):
    # Adiciona uma nova chave "time_seconds" com a soma de "fit_time" e "score_time" para cada modelo
    for chave, valor in resultados.items():
        resultados[chave]["time_seconds"] = (
            resultados[chave]["fit_time"] + resultados[chave]["score_time"]
        )

    # Converte o dicionário de resultados em um DataFrame e renomeia a coluna de índice para "model"
    df_resultados = (
        pd.DataFrame(resultados).T.reset_index().rename(columns={"index": "model"})
    )

    # Expande o DataFrame para que cada métrica tenha uma linha separada por divisão de validação cruzada
    df_resultados_expandido = df_resultados.explode(
        df_resultados.columns[1:].to_list()
    ).reset_index(drop=True)

    # Tenta converter todas as colunas do DataFrame expandido para valores numéricos
    try:
        df_resultados_expandido = df_resultados_expandido.apply(pd.to_numeric)
    except ValueError:
        pass

    # Retorna o DataFrame expandido e organizado
    return df_resultados_expandido