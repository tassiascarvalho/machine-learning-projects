# 🧠 Autoencoder com K-Means para Clustering de Imagens

Este projeto utiliza um autoencoder para aprender representações latentes de imagens do conjunto de dados MNIST e, em seguida, aplica o algoritmo de clustering K-Means para agrupar as representações latentes em clusters. O código explora diferentes moléculas de dimensões do espaço latente e pesos para a perda de residência e clustering.
---

## 📁 Requisitos

- Antes de rodar o projeto, você deve garantir que os seguintes requisitos estão instalados:
- Python 3.x (Recomendado: versão 3.7 ou superior)
- TensorFlow (Recomendado: versão 2.x)
- Keras
- NumPy
- Matplotlib (para visualização dos resultados)
- Pandas

---

## 📊 Descrição do Projeto
O projeto é composto por:

- Carregamento dos dados : O dataset MNIST é carregado e as imagens são normalizadas e achatadas para uso no modelo.
- Autoencoder : Um autoencoder é construído para aprender representações latentes das imagens. O modelo é composto por um codificador e um decodificador.
- K-Means : O algoritmo K-Means é aplicado para realizar o clustering das representações latentes aprendidas pelo autoencoder.
- Visualizações : São visualizações feitas para avaliar a distribuição das representações latentes utilizando t-SNE e para comparar as imagens originais e reconstruídas.
- Avaliação : O desempenho do modelo é avaliado com detalhes como a perda de profundidade e a pontuação de clustering.

---

## 👩‍💻 Autoria

Projeto desenvolvido por Tassia Carvalho no âmbito da disciplina de *Aprendizagem Automática (Mestrado)*.