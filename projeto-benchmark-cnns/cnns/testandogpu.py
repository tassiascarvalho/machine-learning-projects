import tensorflow as tf

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