import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
  
def avaliar_e_plotar_roc(modelo, test_data, num_classes):
    """
    Avalia e plota a curva ROC para classificação binária e multi-classes.

    Args:
        modelo: O modelo treinado.
        test_data: O conjunto de dados de teste (imgs, labels).
        num_classes: Número de classes no problema.
    """
    y_true, y_pred = [], []
    for imgs, labels in test_data:
        preds = modelo.predict(imgs)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Caso Binário
    if num_classes == 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred.ravel())
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC (Binária)')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    # Caso Multi-Classe
    elif num_classes > 2:
        # Binarizar os rótulos verdadeiros
        y_true_binarizado = label_binarize(y_true, classes=np.arange(num_classes))

        # Calcular FPR, TPR e AUC para cada classe
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarizado[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Curva Micro-Average
        fpr_micro, tpr_micro, _ = roc_curve(y_true_binarizado.ravel(), y_pred.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # Curva Macro-Average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)

        # Plotar todas as curvas ROC
        plt.figure(figsize=(10, 8))

        # Curvas individuais para cada classe
        for i in range(num_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

        # Curvas Micro e Macro
        plt.plot(fpr_micro, tpr_micro, linestyle='--', color='gray', label=f'Micro-Average (AUC = {roc_auc_micro:.2f})', lw=2)
        plt.plot(all_fpr, mean_tpr, linestyle='--', color='blue', label=f'Macro-Average (AUC = {roc_auc_macro:.2f})', lw=2)

        # Linha diagonal (aleatório)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC (Multi-Classe)')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

    else:
        raise ValueError("Número de classes inválido para cálculo da curva ROC.")

def exportar_resultados_para_excel(historico, caminho_saida):
    """
    Exporta as métricas de acurácia e perda para um arquivo Excel.

    Args:
        historico: Objeto de histórico retornado pelo model.fit.
        caminho_saida: Caminho para salvar o arquivo Excel.
    """
    # Coletar os dados do histórico
    epochs = range(1, len(historico.history['accuracy']) + 1)
    dados = {
        "Época": epochs,
        "Acurácia Treinamento": historico.history['accuracy'],
        "Acurácia Validação": historico.history['val_accuracy'],
        "Perda Treinamento": historico.history['loss'],
        "Perda Validação": historico.history['val_loss']
    }

    # Criar o DataFrame
    df = pd.DataFrame(dados)

    # Salvar como Excel
    caminho_excel = os.path.join(caminho_saida, "resultados_treinamento.xlsx")
    df.to_excel(caminho_excel, index=False)
    print(f"Resultados do treinamento exportados para {caminho_excel}")

def avaliar_modelo_completo(modelo, test_dataset, test_data, num_classes):
    """
    Avalia o modelo no conjunto de teste com métricas adicionais.
    """
    print("\n### Avaliação no conjunto de teste ###")
    test_loss, test_acc = modelo.evaluate(test_dataset)
    print(f"Teste - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Predições para métricas detalhadas
    y_true = [label for _, label in test_data]
    y_pred_probs = modelo.predict(test_dataset)
    y_pred_classes = np.argmax(y_pred_probs, axis=1) if num_classes > 2 else (y_pred_probs > 0.5).astype(int)

    print("\n### Relatório de Classificação ###")
    print(classification_report(y_true, y_pred_classes))

    # Curva ROC
    avaliar_e_plotar_roc(modelo, test_dataset, num_classes)

def salvar_curva_roc(modelo, test_dataset, test_data, num_classes, output_dir, arquitetura, fine_tuning):
    """
    Salva a curva ROC do modelo.

    Args:
        modelo: O modelo treinado.
        test_dataset: Dataset para avaliação.
        test_data: Dados de teste originais.
        num_classes: Número de classes.
        output_dir: Diretório para salvar os gráficos.
        arquitetura: Nome da arquitetura.
        fine_tuning: Indica se o fine-tuning foi utilizado.
    """
    # Avaliar e calcular a curva ROC
    fpr, tpr, roc_auc = avaliar_e_plotar_roc(modelo, test_dataset, num_classes)

    # Configurar o título e o nome do arquivo
    titulo = f"Curva ROC - {arquitetura} {'Fine-Tuning' if fine_tuning else 'Aprendizado do Zero'}"
    prefixo_arquivo = f"{arquitetura}_{'fine_tuning' if fine_tuning else 'zero'}"

    # Plotar a curva ROC
    plt.figure(figsize=(8, 6))
    if num_classes == 1:
        # Gráfico para problemas binários
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", color='darkorange', lw=2)
    else:
        # Gráfico para problemas multi-classe
        for i, classe in enumerate(fpr.keys()):
            plt.plot(fpr[classe], tpr[classe], label=f"Classe {classe} (AUC = {roc_auc[classe]:.4f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(titulo)  # Configurar o título aqui
    plt.legend(loc="lower right")
    plt.grid()

    # Salvar o gráfico diretamente no arquivo
    arquivo_grafico = os.path.join(output_dir, f"{prefixo_arquivo}_roc.png")
    plt.savefig(arquivo_grafico)
    plt.close()
    print(f"Curva ROC salva em: {arquivo_grafico}")
