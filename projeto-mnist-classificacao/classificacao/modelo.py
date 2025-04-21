from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers


# Função para criar o modelo de rede neuronal com regularização L2 e Dropout
def build_model(input_neurons, hidden_neurons, l2_reg=0.001, dropout_rate=0.25, hidden_neurons2=None):
    inputs = Input(shape=(784,))
    x = Dense(input_neurons, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = Dropout(dropout_rate)(x)

    if hidden_neurons2:
        x = Dense(hidden_neurons2, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Dropout(dropout_rate)(x)

    x = Dense(hidden_neurons, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(10, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)
