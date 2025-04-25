import time
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

# Classe de métricas e gráficos
class ExibirTempoEMetricas(Callback):
    """
    Callback para exibir o tempo de execução e métricas ao final de cada época.
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.tempo_inicio = time.time()
        print(f"\nIniciando época {epoch + 1}...")

    def on_epoch_end(self, epoch, logs=None):
        tempo_epoca = time.time() - self.tempo_inicio
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)

        print(f"Época {epoch + 1} concluída em {tempo_epoca:.2f} segundos")
        print(f"Treinamento -> Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        print(f"Validação  -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

def plotar_metricas_treinamento(historico):

    acc = historico.history["accuracy"]
    val_acc = historico.history["val_accuracy"]
    loss = historico.history["loss"]
    val_loss = historico.history["val_loss"]

    plt.figure(figsize=(12, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Treino")
    plt.plot(val_acc, label="Validação")
    plt.title("Acurácia por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True)

    # Gráfico de Perda
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Treino")
    plt.plot(val_loss, label="Validação")
    plt.title("Perda por Época")
    plt.xlabel("Épocas")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def salvar_graficos_treinamento(historico, output_dir, arquitetura, fine_tuning):
    """
    Salva os gráficos de acurácia e perda durante o treinamento.

    Args:
        historico: Histórico do treinamento.
        output_dir: Diretório para salvar os gráficos.
        arquitetura: Nome da arquitetura.
        fine_tuning: Indica se o fine-tuning foi utilizado.
    """
    titulo = f"{arquitetura} {'Fine-Tuning' if fine_tuning else 'Aprendizado do Zero'}"
    prefixo_arquivo = f"{arquitetura}_{'fine_tuning' if fine_tuning else 'zero'}"

    # Gráfico de Acurácia
    plt.figure()
    plt.plot(historico.history['accuracy'], label='Treinamento')
    plt.plot(historico.history['val_accuracy'], label='Validação')
    plt.title(f'Acurácia - {titulo}')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{prefixo_arquivo}_acuracia.png"))
    plt.close()

    # Gráfico de Perda
    plt.figure()
    plt.plot(historico.history['loss'], label='Treinamento')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title(f'Perda - {titulo}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{prefixo_arquivo}_perda.png"))
    plt.close()
