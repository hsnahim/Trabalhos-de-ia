import numpy as np

# A classe NeuralNetwork modificada para usar RELU na camada oculta
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Estrutura
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos
        self.weights_hidden = np.random.uniform(size=(self.input_size, self.hidden_size))
        self.bias_hidden = np.random.uniform(size=(1, self.hidden_size))
        self.weights_output = np.random.uniform(size=(self.hidden_size, self.output_size))
        self.bias_output = np.random.uniform(size=(1, self.output_size))

    # --- FUNÇÕES DE ATIVAÇÃO ---
    def _relu(self, x):
        """Funçao de ativaçao RELU (para camada oculta)"""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivada da RELU (para camada oculta)"""
        return (x > 0) * 1

    def _sigmoid(self, x):
        """Funçao sigmoide (para camada de saida)"""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivada da sigmoide (para camada de saida)"""
        return x * (1 - x)

    def _mse(self, y_true, y_pred):
        """Funçao de erro quadratico medio (MSE)"""
        return np.mean(np.square(y_true - y_pred))

    def feedforward(self, X):
        """Propagaçao direta (feedforward)"""
        # Camada oculta (AGORA COM RELU)
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_activation = self._relu(self.hidden_input)
        
        # Camada de saída (MANTÉM SIGMOID)
        self.output_input = np.dot(self.hidden_activation, self.weights_output) + self.bias_output
        self.predicted_output = self._sigmoid(self.output_input)
        
        return self.predicted_output

    def backpropagate(self, X, y):
        """Propagaçao reversa (backpropagation)"""
        
        # Calcular o erro
        error = y - self.predicted_output
        
        # 1. Gradiente da camada de saída (USA DERIVADA DA SIGMOID)
        d_predicted_output = error * self._sigmoid_derivative(self.predicted_output)
        
        # 2. Erro propagado para a camada oculta
        error_hidden_layer = d_predicted_output.dot(self.weights_output.T)
        
        # 3. Gradiente da camada oculta (AGORA COM DERIVADA DA RELU)
        d_hidden_layer = error_hidden_layer * self._relu_derivative(self.hidden_activation)
        
        # 4. Atualização dos Pesos (Equação de Ajuste)
        # Camada de saída
        self.weights_output += self.hidden_activation.T.dot(d_predicted_output) * self.learning_rate
        self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * self.learning_rate
        
        # Camada oculta
        self.weights_hidden += X.T.dot(d_hidden_layer) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        """Funçao de treinamento"""
        for epoch in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y)
            
            if (epoch + 1) % 1000 == 0:
                mse = self._mse(y, self.predicted_output)
                print(f"Epoca {epoch+1}/{epochs}, Erro MSE: {mse:.6f}")

    def predict(self, X):
        """Faz uma previsao apos o treinamento"""
        return self.feedforward(X)

# --- Dados do Problema 2 (Display 7 Segmentos) ---
X_display = np.array([
    [1, 1, 1, 1, 1, 1, 0], # 0
    [0, 1, 1, 0, 0, 0, 0], # 1
    [1, 1, 0, 1, 1, 0, 1], # 2
    [1, 1, 1, 1, 0, 0, 1], # 3
    [0, 1, 1, 0, 0, 1, 1], # 4
    [1, 0, 1, 1, 0, 1, 1], # 5
    [1, 0, 1, 1, 1, 1, 1], # 6
    [1, 1, 1, 0, 0, 0, 0], # 7
    [1, 1, 1, 1, 1, 1, 1], # 8
    [1, 1, 1, 1, 0, 1, 1]  # 9
])

y_display = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 1
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 2
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # 6
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 9
])

# --- Treinamento da Rede para 7 Segmentos (com RELU) ---
# Estrutura: 7 entradas, 5 ocultos, 10 saídas
nn_display_relu = NeuralNetwork(input_size=7, hidden_size=5, output_size=10, learning_rate=0.1)
print("Treinando rede de 7 segmentos (RELU na camada oculta)...")
# A taxa de aprendizado ou o número de épocas podem precisar de ajuste
# A RELU pode ser sensível à inicialização de pesos e taxa de aprendizado
nn_display_relu.train(X_display, y_display, epochs=15000)

print("\n--- Teste com dados originais (Rede com RELU) ---")
predictions = nn_display_relu.predict(X_display)

for i in range(len(X_display)):
    pred_digit = np.argmax(predictions[i])
    true_digit = np.argmax(y_display[i])
    status = "ACERTO" if pred_digit == true_digit else "ERRO"
    print(f"Entrada: {X_display[i]}, Previsto: {pred_digit} (Conf: {predictions[i][pred_digit]:.2f}), Esperado: {true_digit} -> {status}")

# --- Teste com Ruído ---
def add_noise(data, noise_level=0.1):
    noisy_data = data.copy()
    for i in range(noisy_data.shape[0]):
        for j in range(noisy_data.shape[1]):
            if np.random.rand() < noise_level:
                noisy_data[i, j] = 1 - noisy_data[i, j]
    return noisy_data

print("\n--- Teste com Ruido (Rede com RELU) ---")
for digit in range(10):
    original_input = X_display[digit:digit+1]
    noisy_input = add_noise(original_input, noise_level=0.15)
    
    noisy_prediction = nn_display_relu.predict(noisy_input)
    pred_digit = np.argmax(noisy_prediction[0])
    true_digit = digit
    status = "ACERTO" if pred_digit == true_digit else "ERRO"
    
    print(f"Entrada Original: {original_input[0]}")
    print(f"Entrada Ruidosa:  {noisy_input[0]}")
    print(f" -> Previsto: {pred_digit}, Esperado: {true_digit} -> {status}\n")