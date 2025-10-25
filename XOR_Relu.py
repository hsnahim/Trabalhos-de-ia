import numpy as np

# --- Novas Funções ---
def relu(x):
    """Funçao de ativaçao RELU"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivada da RELU (1 para x > 0, 0 caso contrario)"""
    # Note que aplicamos a derivada na *ativação* (saída da relu), 
    # então x aqui já será >= 0.
    return (x > 0) * 1

# --- Funções da Sigmoide (mantidas para a camada de saída) ---
def sigmoid(x):
    """Funçao sigmoide"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada da funçao sigmoide"""
    return x * (1 - x)

# Dados do problema XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Parâmetros da rede
np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Inicialização dos pesos
weights_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
weights_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Treinamento com Backpropagation
for epoch in range(epochs):
    # --- Feedforward ---
    # Camada oculta (AGORA COM RELU)
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_activation = relu(hidden_layer_input)
    
    # Camada de saída (MANTÉM SIGMOID)
    output_layer_input = np.dot(hidden_layer_activation, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # --- Backpropagation ---
    
    # Calcular o erro (MSE)
    error = y - predicted_output
    
    # 1. Gradiente da camada de saída (USA DERIVADA DA SIGMOID)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # 2. Erro propagado para a camada oculta
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    
    # 3. Gradiente da camada oculta (AGORA COM DERIVADA DA RELU)
    #    Aplicada na *ativação* da camada oculta
    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_activation)
    
    # --- Atualização dos Pesos (Equação de Ajuste) ---
    # Camada de saída
    weights_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    # Camada oculta
    weights_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    if (epoch + 1) % 1000 == 0:
        mse = np.mean(np.square(error))
        print(f"Epoca {epoch+1}, Erro MSE: {mse:.4f}")

# Resultados do treinamento
print("\n--- Saida apos o treinamento (Problema XOR com RELU) ---")
print("Entradas:")
print(X)
print("Saidas previstas:")
print(predicted_output.round(2))
print("Saidas esperadas:")
print(y)