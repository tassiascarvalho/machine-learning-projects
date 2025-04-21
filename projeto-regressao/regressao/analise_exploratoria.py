import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


# Dicionário para mapear os índices das colunas com os nomes das características (features)
feature_mapping = {
    0: 'age',
    1: 'sex',
    2: 'bmi',
    3: 'children',
    4: 'smoker',
    5: 'region',
    6: 'charges'
}

#Plota histograma e densidade para uma única feature.
def save_hist_and_density_pdf(X: np.ndarray, output_path="histogramas.pdf"):    
    with PdfPages(output_path) as pdf:
        for i in range(X.shape[1]):
            feature = X[:, i]
            name = feature_mapping.get(i, f"Feature {i}")    
            plt.figure(figsize=(12, 5))
            # Gráfico de barras - Histograma
            plt.subplot(1, 2, 1)
            plt.hist(feature, bins=20, alpha=0.6, color='g', edgecolor='black')
            plt.title(f'Histograma da Feature: {name}')
            plt.xlabel(name)
            plt.ylabel('Frequência')
            # Estimativa de densidade - Gráfico de densidade
            
            plt.subplot(1, 2, 2)
            try:
                sns.kdeplot(feature, color='b', linewidth=2)
                plt.title(f'Densidade da Feature: {name}')
                plt.xlabel(name)
                plt.ylabel('Densidade')
            except:
                plt.title("Sem variância — não exibido")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

######################################################################################
#   Funções para obtenção do Correlação

def save_correlation_plots_pdf(X: np.ndarray, y: np.ndarray, output_path="correlacoes.pdf"):
    with PdfPages(output_path) as pdf:
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]): 
                # Plota o gráfico de dispersão entre a característica 'i' e 'j']
                # a) Observando os gráficos de dispersão entre pares de características (features independentes)
                # O objetivo aqui é ver como as características se relacionam umas com as outras    
                plt.scatter(X[:, i], X[:, j], alpha=0.6)
                plt.title(f'{feature_mapping[i]} vs {feature_mapping[j]}')
                plt.xlabel(feature_mapping[i])
                plt.ylabel(feature_mapping[j])
                plt.grid(True)
                pdf.savefig()
                plt.close()
                
        for i in range(X.shape[1]):
            # b) Observando os gráficos de dispersão entre cada característica e a variável dependente 'charges' (encargos)
            # Isso ajuda a identificar como cada característica influencia os encargos
            # Plota o gráfico de dispersão entre a característica 'i' e a variável dependente 'y' (encargos)
            plt.scatter(X[:, i], y, alpha=0.6)
            plt.title(f'{feature_mapping[i]} vs Encargos')
            plt.xlabel(feature_mapping[i])
            plt.ylabel('Encargos')
            plt.grid(True)
            pdf.savefig()
            plt.close()

