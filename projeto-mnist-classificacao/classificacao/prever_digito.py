import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from modelo import build_model
from tensorflow.keras.optimizers import Adam
from desenhar_digito import desenhar_digito, carregar_imagem

# === Parâmetros da arquitetura ===
input_neurons = 512
hidden_neurons = 256
l2_reg = 0.001
dropout_rate = 0.25
learning_rate = 0.0001

# === Carrega o modelo treinado ===
model = build_model(input_neurons, hidden_neurons, l2_reg, dropout_rate)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate), metrics=["accuracy"])
model.load_weights("modelo_treinado_parametrizado.h5")

# === Passo 1: desenhar o dígito ===
desenhar_digito("meu_digito.png")

imagem = carregar_imagem("meu_digito.png")
pred = model.predict(imagem)
classe = np.argmax(pred)

print(f"\n🧠 Dígito previsto: {classe}")
