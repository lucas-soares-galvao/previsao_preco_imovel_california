import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay

from .models import RANDOM_STATE

# Define o tema do Seaborn para os gráficos
sns.set_theme(palette="bright")

# Define a paleta de cores e a transparência dos pontos de dispersão
PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2

# Função para plotar os coeficientes de um modelo
def plot_coeficientes(df_coefs, titulo="Coeficientes"):
    # Plota os coeficientes como um gráfico de barras horizontal
    df_coefs.plot.barh()
    # Define o título do gráfico
    plt.title(titulo)
    # Adiciona uma linha vertical no eixo x=0
    plt.axvline(x=0, color=".5")
    # Define o rótulo do eixo x
    plt.xlabel("Coeficientes")
    # Remove a legenda do gráfico
    plt.gca().get_legend().remove()
    # Exibe o gráfico
    plt.show()

# Função para plotar os resíduos de um modelo
def plot_residuos(y_true, y_pred):
    # Calcula os resíduos (diferença entre valores verdadeiros e preditos)
    residuos = y_true - y_pred

    # Cria uma figura com 3 subplots lado a lado
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # Plota a distribuição dos resíduos com uma linha KDE no primeiro subplot
    sns.histplot(residuos, kde=True, ax=axs[0])

    # Plota os resíduos vs predições no segundo subplot
    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    # Plota os valores verdadeiros vs predições no terceiro subplot
    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    # Ajusta o layout dos subplots para evitar sobreposição
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()

# Função para plotar os resíduos de um estimador
def plot_residuos_estimador(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):
    # Cria uma figura com 3 subplots lado a lado
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    # Plota os resíduos vs predições usando o estimador
    error_display_01 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="residual_vs_predicted",
        ax=axs[1],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    # Plota os valores verdadeiros vs predições usando o estimador
    error_display_02 = PredictionErrorDisplay.from_estimator(
        estimator,
        X,
        y,
        kind="actual_vs_predicted",
        ax=axs[2],
        random_state=RANDOM_STATE,
        scatter_kwargs={"alpha": SCATTER_ALPHA},
        subsample=fracao_amostra,
    )

    # Calcula os resíduos
    residuos = error_display_01.y_true - error_display_01.y_pred

    # Plota a distribuição dos resíduos com uma linha KDE no primeiro subplot
    sns.histplot(residuos, kde=True, ax=axs[0])

    # Aplica o EngFormatter aos eixos se eng_formatter for True
    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    # Ajusta o layout dos subplots para evitar sobreposição
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()

# Função para comparar métricas de diferentes modelos
def plot_comparar_metricas_modelos(df_resultados):
    # Cria uma figura com 4 subplots organizados em uma grade 2x2, compartilhando o eixo x
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    # Lista de métricas a serem comparadas
    comparar_metricas = [
        "time_seconds",  # Tempo de execução em segundos
        "test_r2",  # Coeficiente de determinação R²
        "test_neg_mean_absolute_error",  # Erro absoluto médio negativo
        "test_neg_root_mean_squared_error",  # Raiz quadrada do erro quadrático médio negativo
    ]

    # Nomes das métricas para os títulos dos gráficos
    nomes_metricas = [
        "Tempo (s)",  # Tempo de execução
        "R²",  # Coeficiente de determinação
        "MAE",  # Erro absoluto médio
        "RMSE",  # Raiz quadrada do erro quadrático médio
    ]

    # Itera sobre cada eixo do subplot, métrica e nome da métrica
    for ax, metrica, nome in zip(axs.flatten(), comparar_metricas, nomes_metricas):
        # Cria um boxplot para cada métrica, comparando os modelos
        sns.boxplot(
            x="model",  # Nome do modelo no eixo x
            y=metrica,  # Métrica no eixo y
            data=df_resultados,  # DataFrame com os resultados
            ax=ax,  # Eixo do subplot
            showmeans=True,  # Mostra a média no boxplot
        )
        # Define o título do subplot com o nome da métrica
        ax.set_title(nome)
        # Define o rótulo do eixo y com o nome da métrica
        ax.set_ylabel(nome)
        # Rotaciona os rótulos do eixo x para 90 graus
        ax.tick_params(axis="x", rotation=90)

    # Ajusta o layout dos subplots para evitar sobreposição
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()