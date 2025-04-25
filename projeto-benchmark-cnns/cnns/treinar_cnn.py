import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from utils import generate_image_paths_csv, configurar_dataset, dividir_dados, criar_dataset_tf
from modelo_cnn import criar_modelo
from graficos_metricas import ExibirTempoEMetricas


def configurar_callbacks(config):
    """
    Configura os callbacks para o treinamento.
    """
    os.makedirs(config.pasta_saida, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(config.pasta_saida, "melhor_modelo.weights.h5"),
        save_best_only=True,
        save_weights_only=True,
        monitor='val_accuracy',
        verbose=2
    )
    logger = ExibirTempoEMetricas()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    return [checkpoint, logger, early_stopping, reduce_lr]



def pipeline_treinamento(config):    
    # Configurações iniciais
    config.exibir_configuracoes()

    # Gerar ou carregar dataset
    if not os.path.exists(config.dataset):
        generate_image_paths_csv("../AR_out", config.dataset)

    data = pd.read_csv(config.dataset)
    
    if data.empty or not {'caminho_imagem', 'genero'}.issubset(data.columns):
        raise ValueError("CSV inválido ou vazio. Verifique as colunas esperadas: 'caminho_imagem', 'genero'.")

    
    dataset, num_classes = configurar_dataset(data, config.tipo_classificacao)

    print(f"Tipo de Classificação: {config.tipo_classificacao}")
    print(f"Número de Classes: {num_classes}")
    print(f"Total de Imagens: {len(dataset)}")

    # Divisão dos dados
    train_data, val_data, test_data = dividir_dados(dataset)
    
    print(f"\nExemplos de treino: {train_data[:2]}")
    print(f"Exemplos de validação: {val_data[:2]}")
    print(f"Exemplos de teste: {test_data[:2]}")


    # Preparar datasets tf.data
    train_dataset = criar_dataset_tf(train_data, config, shuffle=True, augment=True)
    val_dataset = criar_dataset_tf(val_data, config, shuffle=False, augment=False)
    test_dataset = criar_dataset_tf(test_data, config, shuffle=False, augment=False)

    # Pesos das classes
    train_labels = np.array([label for _, label in train_data])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))
    print("Pesos das classes calculados:", class_weights)
    
    if config.fine_tuning:
        print("[INFO] Fine-tuning ativado na arquitetura:", config.arquitetura)


    # Criar modelo
    modelo = criar_modelo(config.arquitetura, num_classes, config.fine_tuning, config.taxa_aprendizado, config.dropout_rate)

    # Callbacks
    callbacks = configurar_callbacks(config)

    # Treinamento
    print("\nIniciando treinamento...")
    historico = modelo.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epocas,
        callbacks=callbacks,
        class_weight=class_weights
    )
    print("\nTreinamento concluído!")

    return {
        "modelo": modelo,
        "historico": historico,
        "test_dataset": test_dataset,
        "test_data": test_data,
        "num_classes": num_classes
    }

