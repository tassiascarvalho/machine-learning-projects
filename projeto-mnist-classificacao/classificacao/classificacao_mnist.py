from modelo import build_model
from treino_validacao import train_and_evaluate
from avaliacao_parametrizada import avaliacao_parametrizada
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd

def carregar_dados_mnist():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28 * 28)).astype("float32") / 255.0
    testX = testX.reshape((testX.shape[0], 28 * 28)).astype("float32") / 255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    return trainX, trainY, testX, testY

#Função para encontrar as combinações de hiperparâmetros. (somente para testes)
def buscar_melhores_hiperparametros(trainX, trainY):
    learning_rates = [1e-4, 1e-3]
    batch_sizes = [32, 64]
    l2_regs = [0.01, 0.05]
    dropout_rates = [0.5]
    topologies = [(512, 256), (512, 512, 256)]
    
    resultados = []

    for lr in learning_rates:
        for bs in batch_sizes:
            for reg in l2_regs:
                for dr in dropout_rates:
                    for topo in topologies:
                        print(f"\n[INFO] Testando configuração: LR={lr}, Batch Size={bs}, L2={reg}, Dropout={dr}, Topologia={topo}")
                        resultado = train_and_evaluate(
                            trainX=trainX,
                            trainY=trainY,
                            topology=topo,
                            l2_reg=reg,
                            dropout_rate=dr,
                            learning_rate=lr,
                            batch_size=bs,
                            epochs=50,
                            k_folds=5
                        )
                        resultados.append({
                            "lr": lr,
                            "bs": bs,
                            "l2": reg,
                            "dropout": dr,
                            "topology": topo,
                            "mean_acc": resultado["mean_accuracy"],
                            "std_acc": resultado["std_accuracy"]
                        })

    melhor = max(resultados, key=lambda x: x["mean_acc"])
    print("\n🔍 Melhor configuração encontrada:")
    print(f"Topologia: {melhor['topology']}")
    print(f"LR: {melhor['lr']}, Batch: {melhor['bs']}, L2: {melhor['l2']}, Dropout: {melhor['dropout']}")
    print(f"Acurácia Média: {melhor['mean_acc']*100:.2f}% ± {melhor['std_acc']*100:.2f}%")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("resultados_validacao.csv", index=False)
    
    return resultados, melhor

def main():
    trainX, trainY, testX, testY = carregar_dados_mnist()
    
    avaliacao_parametrizada(
        trainX, trainY, testX, testY,
        input_neurons=512,
        hidden_neurons=256,
        l2_reg=0.001,
        dropout_rate=0.25,
        learning_rate=0.0001,
        epochs=10,
        batch_size=64
    )
    
    resultados, melhor = buscar_melhores_hiperparametros(trainX, trainY)

    
    
if __name__ == "__main__":
    main()
