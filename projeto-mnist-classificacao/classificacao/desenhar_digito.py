import tkinter as tk
from PIL import Image, ImageOps, ImageGrab, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

import tkinter as tk
from PIL import Image, ImageDraw

def desenhar_digito(salvar_como="meu_digito.png"):
    tamanho = 280
    janela = tk.Tk()
    janela.title("Desenhe um dígito")
    canvas = tk.Canvas(janela, width=tamanho, height=tamanho, bg='black')  # fundo preto
    canvas.pack()

    # Criar imagem PIL: fundo preto, traço branco
    img_pil = Image.new("L", (tamanho, tamanho), 0)
    draw = ImageDraw.Draw(img_pil)

    def draw_line(event):
        x, y = event.x, event.y
        r = 12
        canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)  # branco

    def salvar():
        img = img_pil.resize((28, 28), Image.LANCZOS)
        img.save(salvar_como)
        print(f"🖼️ Imagem salva como: {salvar_como}")
        janela.destroy()

    canvas.bind("<B1-Motion>", draw_line)
    tk.Button(janela, text="Salvar", command=salvar).pack()
    janela.mainloop()




def carregar_imagem(caminho):
    imagem = Image.open(caminho).convert('L')

    # Redimensiona diretamente para (28,28) — já vem assim!
    imagem = imagem.resize((28, 28), Image.LANCZOS)

    # Converte para array e normaliza
    img_array = np.array(imagem).astype("float32") / 255.0

    # Visualiza
    plt.imshow(img_array, cmap='gray')
    plt.title("Pré-processado (28x28)")
    plt.axis('off')
    plt.show()

    return img_array.reshape(1, 784)
