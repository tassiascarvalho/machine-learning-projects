# 📈 Projeto de Regressão Linear com Gradiente Descendente

Este repositório contém a implementação de um modelo de regressão linear multivariada com otimização via descida do gradiente, análise exploratória e validação cruzada.

---

## 📁 Estrutura

- `regressao_linear.py`: Script principal
- `analise_exploratoria.py`: Geração de histogramas, densidades e correlações (salvos em PDF)
- `gradient_descent.py`: Algoritmo de otimização com critério de parada (e salvamento opcional da curva de custo)
- `validacao.py`: Validação cruzada (k-fold e LOOCV) com gráficos de erro
- `dataset/insurance.csv`: Ficheiro de dados (não incluído, deve ser fornecido)

---

## 📊 Saídas geradas

- `histogramas.pdf`: Gráficos das distribuições das features (histograma e densidade)
- `correlacoes.pdf`: Gráficos de correlação entre as features e com o target
- `evolucao_custo.png` (opcional): Curva de convergência da função de custo

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
   ```

---

## 👩‍💻 Autoria

Projeto desenvolvido por Tassia Carvalho no âmbito da disciplina de *Aprendizagem Automática (Mestrado)*.