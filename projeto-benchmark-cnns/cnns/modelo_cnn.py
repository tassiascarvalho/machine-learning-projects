import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization,
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
)
from tensorflow.keras.applications import (
    EfficientNetB0, EfficientNetB1, InceptionV3, VGG16, ResNet50
)

#Cria o modelo baseado na arquitetura escolhida e define lógica de fine-tuning.
def criar_modelo(arquitetura, num_classes, fine_tuning, taxa_aprendizado, dropout_rate, largura=224, altura=224):
    """
    Cria o modelo baseado na arquitetura escolhida e define lógica de fine-tuning.
    """
    # Escolha do modelo base
    if arquitetura == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))
    elif arquitetura == "EfficientNetB1":
        base_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))
    elif arquitetura == "InceptionV3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))
    elif arquitetura == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))
    elif arquitetura == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(altura, largura, 3))
    elif arquitetura == "Rede Própria":
        return criar_rede_propria(largura, altura, num_classes, taxa_aprendizado, dropout_rate)
    else:
        raise ValueError("Arquitetura inválida. Escolha entre: EfficientNetB0, EfficientNetB1, InceptionV3, VGG16, ResNet50, Rede Própria")

    # Configuração de Fine-Tuning
    if fine_tuning:
        fine_tuning_percentage = 0.7  # Ajuste a proporção conforme necessário
        total_layers = len(base_model.layers)
        num_layers_to_freeze = int(total_layers * (1 - fine_tuning_percentage))

        # Congelar camadas iniciais
        for layer in base_model.layers[:num_layers_to_freeze]:
            layer.trainable = False

        # Descongelar camadas finais
        for layer in base_model.layers[num_layers_to_freeze:]:
            layer.trainable = True

        print(f"Fine-Tuning ativo: {100 - fine_tuning_percentage * 100:.0f}% das camadas congeladas ({num_layers_to_freeze}/{total_layers}).")
    else:  # Aprendizado do zero
        for layer in base_model.layers:
            layer.trainable = True
        print("Fine-Tuning desativado: todas as camadas estão treináveis.")

    # Construção das camadas superiores
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    # Saída final com base no número de classes
    output = Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)

    # Modelo final
    modelo = Model(inputs=base_model.input, outputs=output)

    # Compilação do modelo
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado),
        loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )

    # Debugging: Exibe as camadas treináveis
    for layer in base_model.layers:
        print(f"Layer: {layer.name}, Treinável: {layer.trainable}")

    return modelo

def criar_rede_propria(largura, altura, num_classes, taxa_aprendizado, dropout_rate):
    """
    Cria uma rede neural personalizada para classificações simples.
    """
    try:
        modelo = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(altura, largura, 3)),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.4),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(dropout_rate),

            Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
        ])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado),
            loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        return modelo
    except Exception as e:
        print(f"Erro ao criar o modelo personalizado: {e}")
        raise