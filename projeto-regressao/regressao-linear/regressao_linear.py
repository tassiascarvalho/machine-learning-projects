from analise_exploratoria import save_hist_and_density_pdf, save_correlation_plots_pdf
from gradient_descent import gradient_descent
from validacao import validate_models, k_fold_validation
import numpy as np
import csv


#   Função para Carregar Dados com Tratamento das Colunas Categoricas e verificação de Nulos
def load_dataset(path):
    # Mapeamento para transformar a coluna 'region' em valores numéricos
    region_mapping = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
    X = []  # Lista de features (idade, sexo, etc.)
    y = []  # Lista de valores dependentes (encargos)

    try:
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # Pula o cabeçalho
            

            for row in csv_reader:
                # Verificação de valores nulos
                if any(value == '' for value in row):
                    continue  # Ignora a linha se algum valor estiver nulo

                # Conversão para números: por ex., 1 para masculino, 0 para feminino
                sexo = 1 if row[1] == "male" else 0
                fumante = 1 if row[4] == "yes" else 0

                # Converte a região usando o dicionário de mapeamento
                regiao = region_mapping.get(row[5], -1)  # Usando -1 como valor padrão se a região não for válida

                # Adiciona as variáveis (features) e o viés (termo constante 1 para θ0)
                X.append([float(row[0]), sexo, float(row[2]), float(row[3]), fumante, regiao, 1])
                y.append(float(row[6]))  # O alvo (encargos)
    except FileNotFoundError:
        print(f"❌ Ficheiro não encontrado: {path}")
        return None, None

    return np.asarray(X), np.asarray(y)

#   Main
def main():
    config = {
        "learning_rate": 0.0001,
        "tolerance": 1e-6,
        "dataset_path": "../dataset/insurance.csv",
        "exibir_graficos": True
    }

    # Carrega o dataset
    X, y = load_dataset(config["dataset_path"])
    if X is None:
        return

    # Análise Exploratória de Dados
    if config["exibir_graficos"]:
        save_hist_and_density_pdf(X)           # Salva histogramas e densidades
        save_correlation_plots_pdf(X, y)       # Salva correlações e feature-target

    # Inicializa theta aleatoriamente
    theta = np.random.rand(X.shape[1])

    # Treina o modelo com descida de gradiente
    theta_gd, evolution_J = gradient_descent(theta, config["learning_rate"], config["tolerance"], X, y, plot=True)

    #Aplicação da Validação dos modelos
    #validate_models(X, y, config["learning_rate"], config["tolerance"])    
    k_fold_validation(X, y, 3,  config["learning_rate"], config["tolerance"])

if __name__ == "__main__":
    main()
