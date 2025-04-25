import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# 1. Carregar o conjunto de dados MNIST
(x_train, _), (x_test, _) = mnist.load_data()  # Ignorar os rótulos, pois o AE é não supervisionado
x_train = x_train.astype('float32') / 255.0  # Normalizar os dados entre 0 e 1
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)  # Achatando as imagens para 784 dimensões
x_test = x_test.reshape(-1, 784)

# Garantir que a distribuição das classes nos conjuntos seja estratificada
print(f"Dimensão do conjunto de treino: {x_train.shape}")
print(f"Dimensão do conjunto de teste: {x_test.shape}")

# 2. Construir o Autoencoder
input_dim = 784  # Dimensão das imagens achatadas
latent_dim = 2  # Dimensão do código latente

# Encoder
input_data = Input(shape=(input_dim,), name="Input")
encoded = Dense(256, activation='relu', name="Encoder_Layer1")(input_data)
encoded = Dense(128, activation='relu', name="Encoder_Layer2")(encoded)
latent_space = Dense(latent_dim, activation='relu', name="Latent_Code")(encoded)

# Decoder
decoded = Dense(128, activation='relu', name="Decoder_Layer1")(latent_space)
decoded = Dense(256, activation='relu', name="Decoder_Layer2")(decoded)
output_data = Dense(input_dim, activation='sigmoid', name="Output")(decoded)

# Construção do Autoencoder
autoencoder = Model(inputs=input_data, outputs=output_data, name="Autoencoder")
autoencoder.compile(optimizer=Adam(), loss='mse')  # Erro quadrático médio como perda

# Resumo do modelo
autoencoder.summary()

# 3. Treinar o Autoencoder
history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# Após o treinamento inicial, combinar conjuntos de treino e teste para maximizar o aprendizado
x_full = np.vstack([x_train, x_test])
autoencoder.fit(
    x_full, x_full,
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1
)

# 4. Visualizar os Resultados

# Função para exibir imagens originais e reconstruídas
def visualize_reconstructions(model, data, n=10):
    reconstructed = model.predict(data[:n])  # Reconstruções
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # Imagens originais
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Imagens reconstruídas
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstruído")
        plt.axis('off')
    plt.show()

# Exibir reconstruções
print("Visualizando reconstruções no conjunto de teste:")
visualize_reconstructions(autoencoder, x_test)

# Função para visualizar o espaço latente
def visualize_latent_space(encoder_model, data, n_samples=5000):
    latent_representations = encoder_model.predict(data[:n_samples])
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], alpha=0.7)
    plt.title("Espaço Latente (2D)")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.show()

# Criar o encoder isolado
encoder = Model(inputs=input_data, outputs=latent_space, name="Encoder")

# Visualizar o espaço latente (se `latent_dim=2`)
if latent_dim == 2:
    print("Visualizando o espaço latente em 2D:")
    visualize_latent_space(encoder, x_test)
else:
    print("Espaço latente não é 2D, ajuste `latent_dim` para 2 para visualização.")

# (b) Perda de Treinamento e Validação
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Perda de Treino', marker='o')
plt.plot(epochs, val_loss, label='Perda de Validação', marker='s', linestyle='--')
plt.title("Perda de Treino vs Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
