import numpy as np

# Função de ativação Sigmoide e sua derivada
def sigmoid(x):
    """Funçao sigmoide [cite: 38]"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivada da funçao sigmoide"""
    return x * (1 - x)

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

# Treinamento 
for epoch in range(epochs):
    # Camada oculta
    hidden_layer_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_layer_activation = sigmoid(hidden_layer_input)
    
    # Camada de saída
    output_layer_input = np.dot(hidden_layer_activation, weights_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # Backpropagation
    
    # Calcula o erro 
    error = y - predicted_output
    
    # 1. Gradiente da camada de saída
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # 2. Erro propagado para a camada oculta
    error_hidden_layer = d_predicted_output.dot(weights_output.T)
    
    # 3. Gradiente da camada oculta
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activation)
    
    # Atualização dos Pesos (Equação de Ajuste) 
    # Camada de saída
    weights_output += hidden_layer_activation.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    # Camada oculta
    weights_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    if (epoch + 1) % 1000 == 0:
        mse = np.mean(np.square(error)) # 
        print(f"Epoca {epoch+1}, Erro MSE: {mse:.4f}")

# Resultados do treinamento
print("\n--- Saida apos o treinamento (Problema XOR) ---")
print("Entradas:")
print(X)
print("Saidas previstas:")
print(predicted_output.round(2))
print("Saidas esperadas:")
print(y)