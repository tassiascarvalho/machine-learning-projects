# config.py

class Configuracoes:
    def __init__(self):
        self.dataset = '../content/dataset.csv'           # Caminho para o CSV
        self.pasta_saida = '../content'                   # Pasta de saída para modelos e logs
        self.tamanho_lote = 32                         # Tamanho do lote (batch size)
        self.largura_imagem = 224                      # Largura das imagens
        self.altura_imagem = 224                       # Altura das imagens
        self.taxa_aprendizado = 1e-4                   # Taxa de aprendizado
        self.dropout_rate = 0.6                        # Taxa de dropout
        self.epocas = 10                               # Total de épocas
        self.arquitetura = "ResNet50"                  # Arquitetura CNN padrão
        self.tipo_classificacao = "Gênero"             # Tipo de classificação (ex: "ID", "Gênero")
        self.fine_tuning = True                        # Se será usado fine-tuning

    def exibir_configuracoes(self):
        """
        Exibe as configurações atuais de forma formatada.
        """
        print("\n✅ Parâmetros configurados com sucesso!")
        print("Resumo da Configuração:")
        print(f" - Tipo de Classificação: {self.tipo_classificacao}")
        print(f" - Arquitetura: {self.arquitetura}")
        print(f" - Fine-Tuning: {'Sim' if self.fine_tuning else 'Não'}")
        print(f" - Tamanho do Lote: {self.tamanho_lote}")
        print(f" - Taxa de Aprendizado: {self.taxa_aprendizado}")
        print(f" - Número de Épocas: {self.epocas}")
        print(f" - Taxa de Dropout: {self.dropout_rate}")
