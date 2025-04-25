import os
import tensorflow as tf
from config import Configuracoes
from treinar_cnn import pipeline_treinamento
from graficos_metricas import plotar_metricas_treinamento
from avaliar_cnn import avaliar_modelo_completo
from avaliar_cnn import exportar_resultados_para_excel


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = todos os logs, 1 = menos detalhado, 2 = avisos, 3 = erros críticos

# Listar GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restringir TensorFlow para usar apenas a primeira GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)  # Permitir uso dinâmico de memória
        print(f"Usando GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"Erro ao configurar a GPU: {e}")
else:
    print("Nenhuma GPU disponível. O código usará a CPU.")


def main():
    # Carregar configurações
    config = Configuracoes()

    # Executar pipeline de treino
    resultados = pipeline_treinamento(config)

    # Exibir métricas
    print("\n📊 Plotando métricas de treinamento...")
    plotar_metricas_treinamento(resultados["historico"])

    # Avaliar o modelo
    print("\n🧪 Avaliando modelo no conjunto de teste...")
    avaliar_modelo_completo(
        modelo=resultados["modelo"],
        test_dataset=resultados["test_dataset"],
        test_data=resultados["test_data"],
        num_classes=resultados["num_classes"]
    )

    # Exportar resultados
    print("\n💾 Exportando resultados para Excel...")
    exportar_resultados_para_excel(resultados["historico"], config.pasta_saida)


if __name__ == "__main__":
    main()
