
import numpy as np
import matplotlib.pyplot as plt


#   Função de Custo J(θ) - agora com múltiplos parâmetros
def J(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    preds = np.matmul(X, theta)  # Previsões da equação linear
    error = preds - y  # Diferença entre as previsões e os valores reais
    return np.sum(error ** 2) / (2 * len(y))  # Cálculo do Mean Squared Error (MSE)

    #Poderia ter sido utilizado o Root Mean Square Error (RMSE), o que é uma métrica válida.
    #Contudo, o cálculo do erro quadrático médio regular (MSE) costuma ser mais comum na
    #otimização com descida do gradiente, porque RMSE inclui uma raiz quadrada que pode complicar
    #o cálculo dos gradientes.

#   Método de Descida Gradiente para Regressão Linear
def gradient_descent(theta, lr, tol, X, y, plot=True):
    it = 0
    max_iter = 40
    Js = []  # Lista para armazenar a evolução da função de custo
    while True:
        # Calcula o vetor de previsões
        preds = np.dot(X, theta)
        # Erro da previsão em relação aos valores reais
        error = preds - y
        # Gradiente da função de custo
        gradient = np.dot(X.T, error) / len(y)  # Operação vetorizada
        # Armazena o valor anterior de theta
        theta_old = np.copy(theta)
        # Atualiza theta com o gradiente
        theta = theta - lr * gradient
        # Calcula a diferença entre os valores antigos e novos de theta
        #delta = np.sum(np.abs(theta - theta_old))
        delta = np.linalg.norm(theta - theta_old)
        # Calcula e armazena a função de custo
        Js.append(J(X, y, theta))
        # Imprime o status atual
        print(f"[{it}] J=%.5f, theta={theta}, delta={delta}" % Js[-1])
        # Gráfico da evolução do custo
        if plot:
            plt.figure(1)
            plt.plot(range(len(Js)), Js, '-bo')  # Gráfico de linha da função de custo
            plt.title("Evolução do Custo (J)")
            plt.xlabel("Iterações")
            plt.ylabel("Custo (J)")
            plt.grid(True)
            plt.pause(0.1)  # Pausa para que o gráfico seja atualizado em tempo real
        
        # Verifica critério de convergência
        if delta < tol:
            break
        
        if it >= max_iter:
            print(f"Parou após {max_iter} iterações (limite atingido).")
            break
        
        it += 1

    return theta, Js
