import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
from gradient_descent import gradient_descent

#Calcula a raiz do erro quadrático médio (RMSE).
def calcular_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

#   Função para validar usando k-fold cross-validation
def k_fold_validation(X: np.ndarray, y: np.ndarray, k: int, lr: float, tol: float):
    # Cria uma instância do KFold para dividir os dados em 'k' partes
    kf = KFold(n_splits=k)
    errors = []  # Lista para armazenar os erros (RMSE) de cada iteração

    # Loop sobre cada divisão dos dados, onde 'train_index' e 'test_index' são os índices dos dados
    for train_index, test_index in kf.split(X):
        # Divide os dados em conjuntos de treinamento e teste
        X_train, X_test = X[train_index], X[test_index]  # Conjunto de treinamento e teste de características
        y_train, y_test = y[train_index], y[test_index]  # Conjunto de treinamento e teste de rótulos

        # Inicializa os parâmetros do modelo (theta) aleatoriamente
        theta = np.random.rand(X_train.shape[1])  # Cria um vetor de parâmetros aleatório do tamanho das características

        # Executa o algoritmo de descida de gradiente para ajustar os parâmetros usando o conjunto de treinamento
        theta, _ = gradient_descent(theta, lr, tol, X_train, y_train, plot=True)  # Atualiza 'theta' com os melhores parâmetros

        # Calcula as previsões no conjunto de teste usando os parâmetros ajustados
        preds = np.matmul(X_test, theta)  # Previsões no conjunto de teste
        # Calcula o erro (RMSE) entre as previsões e os valores reais do conjunto de teste
        errors.append(calcular_rmse(y_test, preds))  # Adiciona o erro à lista de erros

    # Retorna a média e o desvio padrão dos erros obtidos em todas as iterações
    return np.mean(errors), np.std(errors)  # Média e desvio padrão dos RMSEs


#   Função para validar usando LeaveOneOut()
def leave_one_out_validation(X: np.ndarray, y: np.ndarray, lr: float, tol: float):
    loo = LeaveOneOut()
    errors = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inicializa theta aleatoriamente
        theta = np.random.rand(X_train.shape[1])

        # Executa o algoritmo de descida de gradiente
        theta, _ = gradient_descent(theta, lr, tol, X_train, y_train, plot=False)  # Atualiza 'theta' com os melhores parâmetros

        # Calcula o erro no conjunto de teste
        preds = np.matmul(X_test, theta)        
        errors.append(calcular_rmse(y_test, preds))

    return np.mean(errors), np.std(errors)

#   Executa validação para diferentes valores de k e LOOCV.
def validate_models(X: np.ndarray, y: np.ndarray, lr: float, tol: float):
    k_values = [3, 5, 10, 15, 50, 100, 200, 400, 800, 1338]  # Kappa values
    results = []

    # Executa validação k-fold para diferentes valores de k
    for k in k_values:
        mean_error, std_error = k_fold_validation(X, y, k, lr, tol)
        results.append((k, mean_error, std_error))
        print(f"k={k}, Erro Médio: {mean_error}, Desvio Padrão: {std_error}")

    # Executa Leave-One-Out Cross-Validation
    loo_mean_error, loo_std_error = leave_one_out_validation(X, y, lr, tol)
    print(f"LOOCV, Erro Médio: {loo_mean_error}, Desvio Padrão: {loo_std_error}")

    # Plota os resultados
    kappa_vals = [result[0] for result in results]
    mean_errors = [result[1] for result in results]
    std_errors = [result[2] for result in results]

    plt.errorbar(kappa_vals, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.title('Erro Médio e Desvio Padrão para Diferentes Valores de k')
    plt.xlabel('k (Número de Grupos)')
    plt.ylabel('Erro Médio (RMSE)')
    plt.xscale('log')
    plt.grid()
    plt.axhline(y=loo_mean_error, color='r', linestyle='--', label='LOOCV (Erro Médio)')
    plt.legend()
    plt.show()
