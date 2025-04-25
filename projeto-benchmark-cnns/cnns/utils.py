import os
import csv
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

#Gera um arquivo CSV contendo informações das imagens no dataset.
def generate_image_paths_csv(directory, output_csv):
    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            
            genero = 1 if filename[0].lower() == "m" else 0 # Extrair gênero
            id_pessoa = int(filename.split('-')[1])  # 'xx' no nome do arquivo# Extrair ID da pessoa
            expressao = int(filename.split('-')[2].split('.')[0])  # 'yy' no nome do arquivo # Extrair expressão facial
            usa_oculos = 1 if expressao in [8, 9, 10] else 0 # Determinar se usa óculos

            # Caminho completo da imagem
            caminho_imagem = os.path.join(directory, filename)

            # Adicionar os dados ao conjunto
            data.append([caminho_imagem, genero, id_pessoa, expressao, usa_oculos])

    # Escrever os dados no arquivo CSV
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["caminho_imagem", "genero", "id_pessoa", "expressao", "usa_oculos"])
        writer.writerows(data)

    print(f"Arquivo CSV gerado em: {output_csv}")

#Mapear expressões
def mapear_classes(row):
    if row['expressao'] in [1, 14]:
        return 'neutra_pura'  # Classe 0
    elif row['expressao'] in [5, 6, 7, 15, 16, 17]:
        return 'neutra_iluminacao'  # Classe 1
    elif row['expressao'] in [8, 9, 10, 18, 19, 20]:
        return 'neutra_oculos'  # Classe 2
    elif row['expressao'] in [11, 12, 13, 21, 22, 23]:
        return 'neutra_cachecol'  # Classe 3
    elif row['expressao'] == 2:
        return 'sorriso'  # Classe 4
    elif row['expressao'] == 3:
        return 'raiva'  # Classe 5
    elif row['expressao'] == 4:
        return 'grito'  # Classe 6
    else:
        return 'outra'  # Para casos inesperados


#Processa os rótulos do dataset com base no tipo de classificação.
def configurar_dataset(data, tipo_classificacao):    
    if tipo_classificacao == "Gênero":
        dataset = [(row['caminho_imagem'], row['genero']) for _, row in data.iterrows()]
        num_classes = 1  # Classificação binária
    elif tipo_classificacao == "ID":
        dataset = [(row['caminho_imagem'], row['id_pessoa']) for _, row in data.iterrows()]
        num_classes = data['id_pessoa'].nunique()
        id_map = {id_pessoa: idx for idx, id_pessoa in enumerate(sorted(data['id_pessoa'].unique()))}
        dataset = [(path, id_map[label]) for path, label in dataset]
    elif tipo_classificacao == "Expressão facial":
        # Mapear as expressões para classes ajustadas
        data['classe_ajustada'] = data.apply(mapear_classes, axis=1)
        # Mapear as classes ajustadas para índices contínuos (0, 1, 2, ...)
        classe_map = {label: idx for idx, label in enumerate(sorted(data['classe_ajustada'].unique()))}
        data['classe_ajustada'] = data['classe_ajustada'].map(classe_map)
        # Atualizar o dataset para usar a coluna 'classe_ajustada'
        dataset = [(row['caminho_imagem'], row['classe_ajustada']) for _, row in data.iterrows()]
        num_classes = len(classe_map)
        print(f"Número de classes ajustadas: {num_classes}")
    elif tipo_classificacao == "Uso de óculos":
        dataset = [(row['caminho_imagem'], row['usa_oculos']) for _, row in data.iterrows()]
        num_classes = 1  # Classificação binária
    else:
        raise ValueError(f"Tipo de classificação desconhecido: {tipo_classificacao}")

    return dataset, num_classes

#pq usar tf e não usar o input_batch
#Cria um dataset TensorFlow para treinamento, validação ou teste.
def criar_dataset_tf(data, config, shuffle=False, augment=False):
    caminhos = [item[0] for item in data]
    rotulos = [item[1] for item in data]

    def carregar_e_preprocessar(caminho, label):
        img = tf.io.read_file(caminho)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [config.altura_imagem, config.largura_imagem])
        img = img / 255.0

        if augment:
            # Aplicar aumentos de dados (data augmentation)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img, tf.cast(label, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))
    dataset = dataset.map(carregar_e_preprocessar, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    dataset = dataset.batch(config.tamanho_lote).prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def dividir_dados(dataset, tamanho_validacao=0.3, tamanho_teste=0.5):
    """
    Divide o dataset em treino, validação e teste.
    """
    train_data, temp_data = train_test_split(
        dataset,
        test_size=tamanho_validacao,
        stratify=[x[1] for x in dataset],
        random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=tamanho_teste,
        stratify=[x[1] for x in temp_data],
        random_state=42
    )
    return train_data, val_data, test_data