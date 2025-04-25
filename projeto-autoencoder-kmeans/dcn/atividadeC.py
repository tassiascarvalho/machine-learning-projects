import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from tensorflow.keras import backend as K
from sklearn.manifold import TSNE
from tensorflow.keras.regularizers import l2

#Carrega os dados
def carregar_dados():
    (x_train, _), (x_test, _) = mnist.load_data()  # Ignorar rótulos (não supervisionado)
    
    # Normalizar as imagens para valores entre 0 e 1
    x_train = x_train.astype('float32') / 255.0  
    x_test = x_test.astype('float32') / 255.0
    
    # Achatar as imagens para vetores de 784 elementos (28x28)
    x_train = x_train.reshape(-1, 784)  
    x_test = x_test.reshape(-1, 784)

    print(f"Dimensão do conjunto de treino: {x_train.shape}")
    print(f"Dimensão do conjunto de teste: {x_test.shape}")
    
    return x_train, x_test

def criar_autoencoder(input_dim, latent_dim):
    # Encoder
    input_data = Input(shape=(input_dim,), name="Input")
    # Encoder mais robusto
    encoded = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name="Encoder_Layer1")(input_data)
    encoded = Dense(256, activation='relu', kernel_initializer='glorot_uniform', name="Encoder_Layer2")(encoded)
    latent_space = Dense(latent_dim, activation='relu', kernel_initializer='glorot_uniform', name="Latent_Code")(encoded)

    # Decoder
    decoded = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(encoded)
    decoded = Dense(512, activation='relu', kernel_initializer='glorot_uniform')(encoded)
    output_data = Dense(input_dim, activation='sigmoid')(decoded)

    # Construir o Autoencoder com saída do espaço latente
    autoencoder = Model(inputs=input_data, outputs={"Output": output_data, "Latent_Code": latent_space})

    #encoder = Model(inputs=input_data, outputs=latent_space, name="Encoder")
    encoder = Model(inputs=input_data, outputs=latent_space, name="Encoder")

    return autoencoder, encoder

def treinar_autoencoder_com_kmeans(autoencoder, encoder, x_train, x_test, n_clusters=10, max_iterations=10, alpha=0.2, beta=0.8, batch_size=256, epochs=10):

    # Obter representações latentes iniciais
    latent_representations_initial = encoder.predict(x_train)
    print(f"Dimensão das representações latentes iniciais: {latent_representations_initial.shape}")

    # Visualizar as representações latentes iniciais (opcional)
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations_initial[:, 0], latent_representations_initial[:, 1], alpha=0.5)
    plt.title("Representações Latentes Iniciais (Sem Treinamento)")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.show()


    # Aplicar K-Means inicial
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init="k-means++", n_init=10)
    kmeans.fit(latent_representations_initial)
    cluster_centers = K.constant(kmeans.cluster_centers_)
    kmeans_labels = kmeans.labels_

    # Exibir informações dos clusters
    print(f"Número de clusters: {n_clusters}")
    print(f"Centróides do K-Means:\n{cluster_centers}")

    # Se a dimensão latente for maior que 2, aplicar t-SNE
    if latent_representations_initial.shape[1] > 2:
        print("Aplicando t-SNE para visualizar os clusters em 2D...")

        n_samples = 5000  # Número de amostras desejado
        total_samples = latent_representations_initial.shape[0]  # Tamanho total do dataset
        idx = np.random.choice(total_samples, n_samples, replace=False)  # Seleção sem reposição
        latent_representations_subset = latent_representations_initial[idx]  # Subconjunto
        kmeans_labels_subset = kmeans_labels[idx]  # Subconjunto de rótulos

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(latent_representations_subset)

        # Visualizar clusters no espaço latente reduzido pelo t-SNE
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=kmeans_labels_subset, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
        plt.title("Visualização t-SNE com Subconjunto de Dados")
        plt.xlabel("Dimensão 1 (t-SNE)")
        plt.ylabel("Dimensão 2 (t-SNE)")
        plt.show()

    else:
        # Visualizar clusters no espaço latente inicial diretamente
        print("Dimensão latente já é 2, visualizando diretamente os clusters.")
        plt.figure(figsize=(10, 10))
        plt.scatter(latent_representations_initial[:, 0], latent_representations_initial[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(label='Cluster')
        plt.title("Clusters no Espaço Latente Inicial (K-Means)")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.show()


    # Compilar o autoencoder com perda conjunta
    def clustering_loss(y_true, y_pred):
        distances = K.sqrt(K.sum(K.square(K.expand_dims(y_pred, axis=1) - cluster_centers), axis=-1))
        distances = K.maximum(distances, K.epsilon())
        min_distances = K.min(distances, axis=1)
        return K.mean(min_distances)

    autoencoder.compile(
        optimizer=Adam(),
        loss={
            "Output": "mse",  # Perda padrão para reconstrução
            "Latent_Code": clustering_loss  # Perda personalizada para clustering
        },
        loss_weights={"Output": alpha, "Latent_Code": beta}  # Pesos para combinar perdas
    )

    # Loop de treinamento iterativo com atualização do K-Means
    previous_cluster_centers = None# Inicializar centróides anteriores
    tolerance = 1e-4  # Tolerância para verificar mudanças nos centróides

    for iteration in range(max_iterations):
        print(f"\nIteração {iteration + 1}/{max_iterations} do treinamento com K-Means atualizado:")

        # Treinar o autoencoder
        history = autoencoder.fit(
            x_train,
            {"Output": x_train, "Latent_Code": latent_representations_initial},
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, {"Output": x_test, "Latent_Code": encoder.predict(x_test)}),
            verbose=1
        )

        # Atualizar representações latentes
        latent_representations_updated = encoder.predict(x_train)

        # Reaplicar o K-Means
        kmeans.fit(latent_representations_updated)
        cluster_centers_updated = kmeans.cluster_centers_
        kmeans_labels = kmeans.labels_

        if len(np.unique(kmeans.labels_)) < 2:
            print("Todos os dados foram atribuídos a um único cluster. Ajuste o modelo ou os parâmetros.")
            break

        # Verificar a convergência dos clusters
        if previous_cluster_centers is not None:
            center_shift = np.linalg.norm(cluster_centers_updated - previous_cluster_centers, axis=1).max()
            print(f"Maior mudança nos centróides: {center_shift:.6f}")
            if center_shift < tolerance:
                print("Convergência alcançada. Parando o treinamento iterativo.")
                break

        # Atualizar centros e representações latentes
        previous_cluster_centers = cluster_centers_updated
        cluster_centers = K.constant(cluster_centers_updated)
        latent_representations_initial = latent_representations_updated

    print("Treinamento finalizado.")
    return latent_representations_updated, kmeans_labels, kmeans, history

# Função para visualizar imagens originais e reconstruídas
def visualize_reconstructions(model, data, n=10):
    # Obter a saída do modelo (dicionário de saídas)
    outputs = model.predict(data[:n])  # Obter saídas para as primeiras `n` amostras

    # Extrair a reconstrução do dicionário de saídas
    reconstructed = outputs["Output"]  # Saída de reconstrução

    # Plotar as imagens originais e reconstruídas
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Imagens originais
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Imagens reconstruídas -
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title("Reconstruído")
        plt.axis('off')
    plt.show()

# Função para visualizar com t-SNE
def visualize_with_tsne(latent_representations, labels=None, n_samples=5000):
    # Selecionar um subconjunto de dados para o t-SNE (opcional, para acelerar)
    if n_samples < latent_representations.shape[0]:
        idx = np.random.choice(latent_representations.shape[0], n_samples, replace=False)
        latent_representations = latent_representations[idx]
        labels = labels[idx] if labels is not None else None

    # Aplicar t-SNE para reduzir para 2 dimensões
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(latent_representations)

    # Plotar os resultados
    plt.figure(figsize=(10, 10))
    if labels is not None:
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='Cluster')
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
    plt.title("Representações Latentes Visualizadas com t-SNE")
    plt.xlabel("Dimensão 1 (t-SNE)")
    plt.ylabel("Dimensão 2 (t-SNE)")
    plt.show()

# Função para visualizar o espaço latente
def visualize_latent_space(encoder_model, data, n_samples=5000):
    latent_representations = encoder_model.predict(data[:n_samples])
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0], latent_representations[:, 1], alpha=0.5)
    plt.title("Espaço Latente (2D)")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.show()

x_train, x_test = carregar_dados()

input_dim = 784  # Dimensão das imagens achatadas
latent_dim = 32  # Dimensão do espaço latente

# Criar os modelos
autoencoder, encoder = criar_autoencoder(input_dim, latent_dim)

# Exibir o resumo do modelo
autoencoder.summary()

latent_dims = [2, 16, 64]  # Dimensões do espaço latente
alpha_beta_combinations = [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]  # Pesos para reconstrução e clustering
results = []

for latent_dim in latent_dims:
    for alpha, beta in alpha_beta_combinations:
        print(f"\nTreinando com dimensão latente {latent_dim}, alpha={alpha}, beta={beta}")

        # Criar novo modelo com dimensão latente especificada
        autoencoder, encoder = criar_autoencoder(input_dim, latent_dim)

        # Treinar o modelo
        latent_representations_updated, kmeans_labels, kmeans, history = treinar_autoencoder_com_kmeans(
            autoencoder,
            encoder,
            x_train,
            x_test,
            n_clusters=10,
            max_iterations=5,
            alpha=alpha,
            beta=beta,
            batch_size=256,
            epochs=5
        )

        # Coletar métricas de avaliação
        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        clustering_score = np.mean(kmeans.inertia_)  # Soma das distâncias dos pontos aos centróides

         # Salvar resultados
        iteration_result = {
            'latent_dim': latent_dim,
            'alpha': alpha,
            'beta': beta,
            'final_loss': final_loss,
            'val_loss': val_loss,
            'clustering_score': clustering_score
        }
        results.append(iteration_result)

        # Exibir resultados da iteração atual
        print(f"Resultados da Iteração:")
        print(f"  Dimensão Latente: {latent_dim}")
        print(f"  Alpha: {alpha}, Beta: {beta}")
        print(f"  Perda Final: {final_loss:.4f}")
        print(f"  Perda de Validação: {val_loss:.4f}")
        print(f"  Score de Clustering: {clustering_score:.4f}")

# Converter resultados em DataFrame
results_df = pd.DataFrame(results)

# Exibir os resultados no console
print("Resultados dos Experimentos:")
from IPython.display import display
display(results_df)


# Plotar comparações
plt.figure(figsize=(12, 6))
for latent_dim in latent_dims:
    subset = results_df[results_df['latent_dim'] == latent_dim]
    plt.plot(subset['alpha'], subset['final_loss'], label=f'Dim={latent_dim}', marker='o')
plt.title('Variação da Perda Final por Alpha')
plt.xlabel('Alpha (Peso Reconstrução)')
plt.ylabel('Perda Final')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for latent_dim in latent_dims:
    subset = results_df[results_df['latent_dim'] == latent_dim]
    plt.plot(subset['alpha'], subset['clustering_score'], label=f'Dim={latent_dim}', marker='o')
plt.title('Variação do Score de Clustering por Alpha')
plt.xlabel('Alpha (Peso Reconstrução)')
plt.ylabel('Score de Clustering')
plt.legend()
plt.grid(True)
plt.show()

# Visualizar representações latentes finais com K-Means
plt.figure(figsize=(10, 10))
plt.scatter(latent_representations_updated[:, 0], latent_representations_updated[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7)
plt.colorbar(label='Cluster')
plt.title("Representações Latentes Finais com Clustering")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.show()

# Visualizar perda de treinamento
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_loss, label='Perda de Treino', marker='o')
plt.plot(epochs_range, val_loss, label='Perda de Validação', marker='s', linestyle='--')
plt.title("Perda de Treino vs Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Exibir reconstruções
print("Visualizando reconstruções no conjunto de teste:")
visualize_reconstructions(autoencoder, x_test)

# Visualizar o espaço latente
if latent_dim == 2:
    print("Visualizando o espaço latente em 2D:")
    visualize_latent_space(encoder, x_test)  # Função existente para plotar em 2D
else:
    print("Visualizando o espaço latente com t-SNE:")
    latent_representations_updated = encoder.predict(x_train)  # Obtenha as representações latentes
    visualize_with_tsne(latent_representations_updated, labels=kmeans.labels_)  # Função t-SNE para plotar


