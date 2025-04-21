# 📈 Projeto de Regressão com Gradiente Descendente

Este repositório contém a implementação de um modelo de regressão supervisionada - linear e polinomial - com otimização via descida do gradiente, análise exploratória e validação cruzada.

---

## 📁 Estrutura

- `regressao_linear.py`: Script principal
- `analise_exploratoria.py`: Geração de histogramas, densidades e correlações (salvos em PDF)
- `gradient_descent.py`: Algoritmo de otimização com critério de parada (e salvamento opcional da curva de custo)
- `validacao.py`: Validação cruzada (k-fold e LOOCV) com gráficos de erro
- `regressao_polinomial.py`: Expande as features com PolynomialFeatures e aplica os mesmos métodos de treino e validação
- `dataset/insurance.csv`: Ficheiro de dados 

---

## 📊 Saídas geradas

- `histogramas.pdf`: Gráficos das distribuições das features (histograma e densidade)
- `correlacoes.pdf`: Gráficos de correlação entre as features e com o target
  
---

## 🚀 Como executar

1. **Crie o ambiente virtual (recomendado):**
   ```bash
   python -m venv venv-ml
   ```

2. **Ative o ambiente:**

   - **Windows:**
     ```bash
     .\venv-ml\Scripts\activate
     ```
   - **Linux/macOS:**
     ```bash
     source venv-ml/bin/activate
     ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Coloque o ficheiro `insurance.csv` na pasta `dataset/`**  
   (crie a pasta `dataset` se necessário)

5. **Execute o script principal:**
   ```bash
   python regressao_linear.py
   python regressao_polinomial.py
   ```

---
📚 Considerações
A regressão polinomial tende a demorar mais para convergir devido ao maior número de features.

Os valores de learning_rate e tolerance podem precisar ser ajustados conforme o grau do polinómio.

## 👩‍💻 Autoria

Projeto desenvolvido por Tassia Carvalho no âmbito da disciplina de *Aprendizagem Automática (Mestrado)*.
