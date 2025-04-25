# 🧠 Classificação de Dígitos com MNIST

Este projeto implementa uma rede neuronal para classificação dos dígitos manuscritos do dataset MNIST, utilizando Keras + TensorFlow. Inclui:

- Teste com topologia configurável
- Avaliação com validação cruzada k-fold
- Otimização de hiperparâmetros
- Visualização de métricas de treino


---

## 📁 Estrutura

- `classificacao_mnist.py`: Script principal
- `modelo.py`: Define a arquitetura da rede
- `treino_validacao.py`: Treino com K-Fold e gráficos
- `avaliacao_parametrizada.py`: Avaliação final com parâmetros definidos

---

## 📊 Resultados

Os gráficos são exibidos durante o treino. Resultados médios por configuração são salvos no ficheiro:

- `resultados_validacao.csv`

---
## 🖍️ Teste Manual com Desenho
É possível testar a rede com um dígito desenhado pelo utilizador. Para isso:

- Executa o script prever_digito.py.
- Será aberta uma janela onde podes desenhar o dígito com o rato (mouse).
- Após clicar em Salvar, o modelo preverá o valor do dígito.
- É exibida a imagem processada e a predição no terminal.
- O modelo usa os pesos treinados previamente salvos em modelo_treinado_parametrizado.h5.

---
## 👩‍💻 Autoria

Projeto desenvolvido por Tassia Carvalho no âmbito da disciplina de *Aprendizagem Automática (Mestrado)*.
