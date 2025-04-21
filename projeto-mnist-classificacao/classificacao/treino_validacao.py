import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from modelo import build_model

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)



# Função para treinar e validar o modelo com validação cruzada K-Fold
def train_and_evaluate(trainX, trainY, topology,
                       l2_reg=0.001, dropout_rate=0.25,
                       learning_rate=0.0001, batch_size=128, epochs=50, k_folds=5):

 
    # Configurações de validação cruzada
    kf = KFold(n_splits=k_folds, shuffle=True)

    fold_accuracies = []

    # Loop para cada fold de validação cruzada
    for fold, (train_idx, val_idx) in enumerate(kf.split(trainX)):
        print(f"[INFO] Iniciando fold {fold + 1} de {k_folds}...")

        X_train, X_val = trainX[train_idx], trainX[val_idx]
        Y_train, Y_val = trainY[train_idx], trainY[val_idx]

        # Desempacota a topologia conforme a quantidade de camadas
        if len(topology) == 2:
            input_neurons, hidden_neurons = topology
            model = build_model(input_neurons, hidden_neurons, l2_reg, dropout_rate)
        elif len(topology) == 3:
            input_neurons, hidden_neurons, hidden_neurons2 = topology
            model = build_model(input_neurons, hidden_neurons, l2_reg, dropout_rate, hidden_neurons2)
        
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"]
        )

        early_stopping = EarlyStopping(
            monitor="val_accuracy", 
            patience=10, 
            min_delta=0.001,
            restore_best_weights=True)

        history_list = []

        H = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]
        )
        history_list.append(H.history)
        
        plot_history(H.history, fold, save_path="graficos")

        val_loss, val_accuracy = model.evaluate(X_val, Y_val, verbose=0)

        print(f"[INFO] Acurácia de validação no fold {fold + 1}: {val_accuracy * 100:.2f}%".encode('utf-8', errors='replace').decode('utf-8'))
        fold_accuracies.append(val_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"[INFO] Acurácia média entre {k_folds} folds: {mean_accuracy * 100:.2f}% ± {std_accuracy * 100:.2f}%")
    return {
        "topology": topology,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "history": history_list
    }


def plot_history(history, fold, save_path="."):
    plt.figure(figsize=(12, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Val')
    plt.title(f'Fold {fold+1} - Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'Fold {fold+1} - Perda')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/history_fold{fold+1}.png")