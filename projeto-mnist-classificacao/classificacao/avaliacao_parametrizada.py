import matplotlib.pyplot as plt
from modelo import build_model
from tensorflow.keras.optimizers import Adam

def avaliacao_parametrizada(trainX, trainY, testX, testY,
                         input_neurons=512, hidden_neurons=256,
                         l2_reg=0.001, dropout_rate=0.25,
                         learning_rate=0.0001, epochs=3, batch_size=128):
    
    # Avaliar o modelo final nos dados de teste
    model = build_model(input_neurons, hidden_neurons, l2_reg, dropout_rate)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Treinamento do modelo
    H_final = model.fit(trainX, trainY, 
                    validation_data=(testX, testY),
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1)

    # Avaliação no conjunto de teste
    test_loss, test_accuracy = model.evaluate(testX, testY, verbose=0)
    print(f"[INFO] Acurácia final no conjunto de teste: {test_accuracy * 100:.2f}%")

    # Gerar gráfico de evolução
    plt.figure(figsize=(10, 4))

    # Gráfico de perda
    plt.subplot(1, 2, 1)
    plt.plot(H_final.history["loss"], label="Treino")
    plt.plot(H_final.history["val_loss"], label="Validação")
    plt.title("Perda por Época")
    plt.xlabel("Época")
    plt.ylabel("Perda")
    plt.legend()

    # Gráfico de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(H_final.history["accuracy"], label="Treino")
    plt.plot(H_final.history["val_accuracy"], label="Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()

    return test_accuracy