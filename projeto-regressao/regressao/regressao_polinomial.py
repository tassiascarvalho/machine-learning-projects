import csv
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent
from validacao import validate_models, k_fold_validation

from sklearn.preprocessing import PolynomialFeatures

def load_dataset_polynomial(path, degree=2):
    region_mapping = {'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3}
    X = []
    y = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            if any(value == '' for value in row):
                continue
            sexo = 1 if row[1] == "male" else 0
            fumante = 1 if row[4] == "yes" else 0
            regiao = region_mapping.get(row[5], -1)
            X.append([float(row[0]), sexo, float(row[2]), float(row[3]), fumante, regiao])
            y.append(float(row[6]))

    X = np.asarray(X)
    y = np.asarray(y)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    return X_poly, y

def main():
    config = {
        "learning_rate": 1e-7,
        "tolerance": 0.01,
        "degree": 2,
        "dataset_path": "../dataset/insurance.csv"
    }

    # Carrega o dataset com transformação polinomial
    X, y = load_dataset_polynomial(config["dataset_path"], config["degree"])

    if X is None:
        print("Erro ao carregar o dataset.")
        return

    # Inicializa os pesos aleatoriamente
    theta = np.random.rand(X.shape[1])

    # Treina o modelo com descida de gradiente
    theta_gd, evolution_J = gradient_descent(theta, config["learning_rate"], config["tolerance"], X, y, plot=True)

    # Validação cruzada
    #validate_models(X, y, config["learning_rate"], config["tolerance"])
    k_fold_validation(X, y, 3,  config["learning_rate"], config["tolerance"])


if __name__ == "__main__":
    main()