import os
import csv
import re
from tensorflow.keras.utils import to_categorical


#Gera um arquivo CSV contendo informações das imagens no dataset.
def gerar_csv_dataset(diretorio, arquivo_saida, one_hot=True):
    if not os.path.exists(diretorio):
        raise FileNotFoundError(f"O diretório {diretorio} não foi encontrado.")

    # Coleta todos os IDs para determinar o número total de classes (opcional para one-hot)
    ids_unicos = set()
    expressoes_unicas = set()
    
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".jpg"):
            try:
                id_pessoa = int(arquivo.split('-')[1])
                expressao = int(arquivo.split('-')[2].split('.')[0])
                ids_unicos.add(id_pessoa)
                expressoes_unicas.add(expressao)
            except:
                pass  # Ignorar arquivos com nomes inválidos

    # Mapear IDs e expressões únicos para índices consistentes
    ids_unicos = sorted(ids_unicos)
    expressoes_unicas = sorted(expressoes_unicas)
    id_map = {id_valor: idx for idx, id_valor in enumerate(ids_unicos)}
    expressao_map = {exp_valor: idx for idx, exp_valor in enumerate(expressoes_unicas)}

    dados = []
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".jpg"):
            caminho = os.path.join(diretorio, arquivo)
            try:
                # Normalizar gênero
                genero = 1 if arquivo.startswith("M") or arquivo.startswith("m") else 0  # 1 para masculino, 0 para feminino

                # Normalizar ID
                id_pessoa = int(arquivo.split('-')[1])  # Extrai o identificador de pessoa
                id_codificado = to_categorical(id_map[id_pessoa], num_classes=len(ids_unicos)).tolist() if one_hot else id_pessoa
                
                # Normalizar expressao
                expressao = int(arquivo.split('-')[2].split('.')[0])  # Extrai o tipo de expressão facial
                if 14 <= expressao <= 26:
                    expressao = expressao - 13  # Normaliza expressões da segunda sessão para corresponder às da primeira
                expressao_codificada = to_categorical(expressao_map[expressao], num_classes=len(expressoes_unicas)).tolist() if one_hot else expressao
                
                # Normalizar o óculos
                oculos = 1 if 8 <= expressao <= 10 else 0  # Determina se está usando óculos
                
                # Adicionar ao dataset
                dados.append([caminho, id_codificado, genero, expressao_codificada, oculos])
            except Exception as e:
                print(f"Erro ao processar o arquivo {arquivo}: {e}")

    # Criar e salvar o CSV
    with open(arquivo_saida, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Cabeçalho do CSV
        if one_hot:
            writer.writerow(["caminho_imagem", "id_one_hot", "genero", "expressao_one_hot", "oculos"])
        else:
            writer.writerow(["caminho_imagem", "id", "genero", "expressao", "oculos"])
        writer.writerows(dados)

    print(f"CSV gerado com sucesso: {arquivo_saida}")

if __name__ == "__main__":
    gerar_csv_dataset("../AR_out", "../content/dataset.csv")