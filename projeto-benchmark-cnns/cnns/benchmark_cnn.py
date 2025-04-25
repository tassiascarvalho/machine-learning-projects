
import os
import pandas as pd
import time
from config import Configuracoes
from treinar_cnn import pipeline_treinamento
from graficos_metricas import salvar_graficos_treinamento
from avaliar_cnn import salvar_curva_roc

def executar_experimentos():
    # Listar arquiteturas
    arquiteturas = ["EfficientNetB0", "EfficientNetB1", "InceptionV3", "VGG16", "ResNet50", "Rede Própria"]    
    tipo_classificacao = "Gênero"
    resultados = []
    output_dir = "graficos_resultados"
    os.makedirs(output_dir, exist_ok=True)

    # Iterar por cada arquitetura
    for arquitetura in arquiteturas:
        for fine_tuning in [True, False]:  # Paradigmas: Ajuste fino e aprendizado do zero
            print(f"\nTreinando {arquitetura} com {'Fine-Tuning' if fine_tuning else 'Aprendizado do Zero'}...\n")
            
            # Configurar as opções de treinamento
            config = Configuracoes()
            config.arquitetura = arquitetura
            config.tipo_classificacao = tipo_classificacao
            config.fine_tuning = fine_tuning

            try:
                # Executar o pipeline de treinamento
                inicio = time.time()

                resultados_pipeline = pipeline_treinamento(config)
                modelo = resultados_pipeline["modelo"]
                historico = resultados_pipeline["historico"]
                test_dataset = resultados_pipeline["test_dataset"]
                test_data = resultados_pipeline["test_data"]
                num_classes = resultados_pipeline["num_classes"]


                # Avaliar o modelo
                test_loss, test_acc = modelo.evaluate(test_dataset, verbose=2)  

                # Gráficos e ROC
                salvar_graficos_treinamento(historico, output_dir, arquitetura, fine_tuning)
                salvar_curva_roc(modelo, test_dataset, test_data, num_classes, output_dir, arquitetura, fine_tuning)

                duracao = time.time() - inicio
                resultados.append({
                    "Arquitetura": arquitetura,
                    "Fine-Tuning": fine_tuning,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Tempo (s)": round(duracao, 2)
                })
            except Exception as e:
                # Caso ocorra um erro
                print(f"Erro durante o experimento com {arquitetura}: {e}")
                resultados.append({
                    "Arquitetura": arquitetura,
                    "Fine-Tuning": fine_tuning,
                    "Test Loss": None,
                    "Test Accuracy": None,
                    "Erro": str(e)
                })

    # Converter os resultados para um DataFrame
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_excel("resultados_genero.xlsx", index=False)
    print("\nResultados salvos em 'resultados_genero.xlsx'\n")
    print(df_resultados.to_string(index=False))  # Exibe o DataFrame formatado no console


def main():
    executar_experimentos()

if __name__ == "__main__":
    main()
