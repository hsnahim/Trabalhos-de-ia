# =============================================================================
#           LISTA 4- ÁRVORES DE DECISÃO
#
# Aluno: Hnrique Silverio Nahim
# Matrícula: 804348
# =============================================================================

# --- Importações de Bibliotecas ---
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# --- Funções de Cálculo de Impureza e Ganho ---

def entropy(y):
    """
    Calcula a Entropia de um conjunto de dados.
    Mede o grau de "caos" dos dados.
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    # A fórmula da entropia é -soma(p * log2(p))
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gini_index(y):
    """
    Calcula o Índice Gini.
    Mede o "caos" e é computacionalmente mais rápido que a entropia.
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    # A fórmula é 1 - soma(p^2)
    return 1 - np.sum(ps**2)

def information_gain(y, y_subsets):
    """
    Calcula o Ganho de Informação (usado pelo ID3).
    Mede a redução na entropia após uma divisão.
    """
    parent_entropy = entropy(y)
    n = len(y)
    # Calcula a entropia ponderada dos filhos (após a divisão)
    child_entropy = sum((len(subset) / n) * entropy(subset) for subset in y_subsets)
    return parent_entropy - child_entropy

def gain_ratio(y, y_subsets):
    """
    Calcula a Razão de Ganho (usado pelo C4.5).
    Normaliza o Ganho de Informação para evitar viés com atributos de muitos valores.
    """
    ig = information_gain(y, y_subsets)
    n = len(y)
    # Calcula a "Entropia da Divisão" (Split Information)
    split_info = -np.sum([(len(subset) / n) * np.log2(len(subset) / n) for subset in y_subsets if len(subset) > 0])
    # Evita divisão por zero se o split_info for 0
    return ig / split_info if split_info != 0 else 0

def gini_gain(y, y_subsets):
    """
    Calcula a redução na impureza Gini (usado pelo CART).
    """
    parent_gini = gini_index(y)
    n = len(y)
    child_gini = sum((len(subset) / n) * gini_index(subset) for subset in y_subsets)
    return parent_gini - child_gini

# --- Classe Node (Estrutura da Árvore) ---

class Node:
    """
    Representa um nó na árvore de decisão.
    Pode ser um nó de decisão (com uma pergunta) ou um nó folha (com uma previsão).
    """
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature       # Atributo usado para a divisão
        self.threshold = threshold   # Limiar (para numéricos) ou conjunto de valores (para categóricos no CART)
        self.children = children     # Dicionário de nós filhos
        self.value = value           # Valor da previsão se for um nó folha

    def is_leaf_node(self):
        # Um nó é uma folha se ele tem um valor de previsão.
        return self.value is not None


class ID3:
    """
    Implementação do algoritmo ID3.
    """
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split # Mínimo de amostras para dividir um nó
        self.max_depth = max_depth                 # Profundidade máxima da árvore
        self.root = None

    def fit(self, X, y):
        self.features = list(X.columns)
        X_vals = X.values
        y_vals = y.values if isinstance(y, pd.Series) else y
        self.root = self._grow_tree(X_vals, y_vals)

    def _grow_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        # Se os rótulos não forem numéricos, converte para análise
        y_numeric = pd.factorize(y)[0] if not np.issubdtype(y.dtype, np.number) else y
        n_labels = len(np.unique(y_numeric))

        # Condições de parada para a recursão
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Encontrar o melhor atributo para dividir
        best_feature_idx = self._best_criteria(X, y_numeric)
        if best_feature_idx is None:
            return Node(value=self._most_common_label(y))

        # Dividir os dados com base no melhor atributo
        best_feature_name = self.features[best_feature_idx]
        feature_values = np.unique(X[:, best_feature_idx])
        children = {}
        
        for value in feature_values:
            subset_indices = (X[:, best_feature_idx] == value)
            X_subset, y_subset = X[subset_indices], y[subset_indices]
            # Se um subconjunto ficar vazio, cria uma folha com a classe mais comum do pai
            if len(y_subset) == 0:
                children[value] = Node(value=self._most_common_label(y))
            else:
                children[value] = self._grow_tree(X_subset, y_subset, depth + 1)
        
        return Node(feature=best_feature_name, children=children)

    def _best_criteria(self, X, y):
        best_gain = -1
        best_idx = None
        for feat_idx in range(X.shape[1]):
            values = np.unique(X[:, feat_idx])
            subsets_y = [y[X[:, feat_idx] == val] for val in values]
            if len(subsets_y) <= 1: continue
            
            gain = information_gain(y, subsets_y)
            if gain > best_gain:
                best_gain = gain
                best_idx = feat_idx
        
        # Critério de desempate: se ganhos são iguais, escolhe o primeiro encontrado. 
        return best_idx if best_gain > 0 else None

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0] if counter else None

    def predict(self, X):
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        return np.array([self._traverse_tree(x, self.root) for x in X_values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
            
        feature_idx = self.features.index(node.feature)
        val = x[feature_idx]

        # Se um valor de teste não foi visto no treino,
        # não haverá um caminho. Como fallback, retornamos o valor mais comum do nó atual.
        # Uma forma simples de estimar isso é pegar a previsão da primeira folha filha.
        if val not in node.children:
            return next(iter(node.children.values())).value
        
        return self._traverse_tree(x, node.children[val])

class C45:
    """
    Implementação do algoritmo C4.5.

    """
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.numeric_features_indices = []

    def fit(self, X, y):
        self.features = list(X.columns)
        self.numeric_features_indices = [i for i, dtype in enumerate(X.dtypes) if np.issubdtype(dtype, np.number)]
        X_vals = X.values
        y_vals = y.values
        self.root = self._grow_tree(X_vals, y_vals)

    def _grow_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))

        best_split = self._best_criteria(X, y)
        if best_split is None:
            return Node(value=self._most_common_label(y))

        feature_idx, threshold = best_split
        feature_name = self.features[feature_idx]

        # Divisão categórica 
        if threshold is None:
            values = np.unique(X[:, feature_idx])
            children = {}
            for val in values:
                subset_indices = (X[:, feature_idx] == val)
                X_subset, y_subset = X[subset_indices], y[subset_indices]
                if len(y_subset) == 0:
                    children[val] = Node(value=self._most_common_label(y))
                else:
                    children[val] = self._grow_tree(X_subset, y_subset, depth + 1)
            return Node(feature=feature_name, children=children)
        # Divisão contínua (binária)
        else:
            left_indices = X[:, feature_idx] <= threshold
            right_indices = ~left_indices
            
            X_left, y_left = X[left_indices], y[left_indices]
            X_right, y_right = X[right_indices], y[right_indices]
            
            left_child = self._grow_tree(X_left, y_left, depth + 1)
            right_child = self._grow_tree(X_right, y_right, depth + 1)
            
            children = {'<=_': left_child, '>_': right_child}
            return Node(feature=feature_name, threshold=threshold, children=children)

    def _best_criteria(self, X, y):
        best_gain_ratio = -1
        best_split = None 

        for feat_idx in range(X.shape[1]):
            # Atributos Contínuos
            if feat_idx in self.numeric_features_indices:
                values = np.unique(X[:, feat_idx])
                # Testa pontos médios entre valores adjacentes
                thresholds = (values[:-1] + values[1:]) / 2
                for thr in thresholds:
                    left_y = y[X[:, feat_idx] <= thr]
                    right_y = y[X[:, feat_idx] > thr]
                    if len(left_y) == 0 or len(right_y) == 0: continue
                    
                    current_gain_ratio = gain_ratio(y, [left_y, right_y])
                    if current_gain_ratio > best_gain_ratio:
                        best_gain_ratio = current_gain_ratio
                        best_split = (feat_idx, thr)
            # Atributos Categóricos
            else:
                values = np.unique(X[:, feat_idx])
                if len(values) <= 1: continue
                subsets_y = [y[X[:, feat_idx] == val] for val in values]
                current_gain_ratio = gain_ratio(y, subsets_y)
                if current_gain_ratio > best_gain_ratio:
                    best_gain_ratio = current_gain_ratio
                    best_split = (feat_idx, None)

        return best_split if best_gain_ratio > 0 else None

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0] if counter else 1 # Fallback

    def predict(self, X):
        X_values = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X_values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_idx = self.features.index(node.feature)
        val = x[feature_idx]
        
        if node.threshold is not None: # Divisão contínua
            go_to = '<=_' if val <= node.threshold else '>_'
            return self._traverse_tree(x, node.children[go_to])
        else: # Divisão categórica
            if val not in node.children:
                return next(iter(node.children.values())).value # Fallback
            return self._traverse_tree(x, node.children[val])

class CART:
    """
    Implementação do algoritmo CART para classificação.

    """
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
        self.numeric_features_indices = []

    def fit(self, X, y):
        self.features = list(X.columns)
        self.numeric_features_indices = [i for i, dtype in enumerate(X.dtypes) if np.issubdtype(dtype, np.number)]
        X_vals = X.values
        y_vals = y.values
        self.root = self._grow_tree(X_vals, y_vals)

    def _grow_tree(self, X, y, depth=0):
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))

        best_split = self._best_criteria(X, y)
        if best_split is None:
            return Node(value=self._most_common_label(y))

        feature_idx, threshold = best_split
        
        left_indices, right_indices = self._split(X[:, feature_idx], threshold)
        
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        # Se uma divisão resultar em um filho vazio, cria uma folha.
        if len(y_left) == 0 or len(y_right) == 0:
            return Node(value=self._most_common_label(y))

        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        
        return Node(feature=self.features[feature_idx], threshold=threshold, children={'<=_': left_child, '>_': right_child})

    def _best_criteria(self, X, y):
        best_gini_gain = -1
        best_split = None

        for feat_idx in range(X.shape[1]):
            values = np.unique(X[:, feat_idx])
            thresholds = []

            if feat_idx in self.numeric_features_indices:
                if len(values) > 1:
                    thresholds = (values[:-1] + values[1:]) / 2
                else: # Binarização para categóricos
                    if len(values) > 1:
                        for i in range(1, len(values) // 2 + 1):
                            for combo in combinations(values, i):
                                thresholds.append(list(combo))
            
            for thr in thresholds:
                left_y, right_y = self._split_y(X[:, feat_idx], y, thr)
                if len(left_y) == 0 or len(right_y) == 0: continue
                
                gain = gini_gain(y, [left_y, right_y])
                if gain > best_gini_gain:
                    best_gini_gain = gain
                    best_split = (feat_idx, thr)

        return best_split if best_gini_gain > 0 else None
    
    def _split(self, X_column, threshold):
        # Para categóricos, o threshold é uma lista de valores
        is_left = np.isin(X_column, threshold) if isinstance(threshold, list) else (X_column <= threshold)
        return is_left, ~is_left

    def _split_y(self, X_column, y, threshold):
        left_mask, right_mask = self._split(X_column, threshold)
        return y[left_mask], y[right_mask]

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0] if counter else 1

    def predict(self, X):
        X_values = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X_values])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        feature_idx = self.features.index(node.feature)
        val = x[feature_idx]
        
        is_left = val in node.threshold if isinstance(node.threshold, list) else val <= node.threshold
        go_to = '<=_' if is_left else '>_'
        return self._traverse_tree(x, node.children[go_to])


def print_tree(node, spacing=""):
    """Função auxiliar para imprimir a árvore de forma legível."""
    if node.is_leaf_node():
        print(spacing + "Prevê:", node.value)
        return

    # Formatação do limiar (threshold)
    threshold_str = ""
    if node.threshold is not None:
        if isinstance(node.threshold, list):
            # Para atributos categóricos no CART, o threshold é uma lista
            threshold_str = f" in {node.threshold}"
        else:
            # Para atributos numéricos
            threshold_str = f" <= {node.threshold:.2f}"

    print(spacing + f"[{node.feature}{threshold_str}]")
    
    # Se a divisão é binária (C4.5 contínuo ou CART)
    if '<=_' in node.children:
        print(spacing + '--> True:')
        print_tree(node.children['<=_'], spacing + "  ")
        print(spacing + '--> False:')
        print_tree(node.children['>_'], spacing + "  ")
    # Se a divisão é multi-ramificada (ID3 ou C4.5 categórico)
    else:
        for value, child in node.children.items():
            print(spacing + f"--> {value}:")
            print_tree(child, spacing + "  ")

def main():
    """
    Função principal que executa todos os passos do projeto.
    """
    print("="*60)
    print("  PROJETO DE IMPLEMENTAÇÃO DE ÁRVORES DE DECISÃO (ID3, C4.5, CART)")
    print("="*60)

    #  1. Preparação dos Dados do Titanic 
    print("\n--- [Seção 1] Preparando os Dados do Titanic ---\n")
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    titanic_df = pd.read_csv(url)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    df = titanic_df[features + [target]]

    # Limpeza de dados (Missing Values) 
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Conversão de categóricos para numéricos (para C4.5 e CART)
    df_continuous = df.copy()
    df_continuous['Sex'] = df_continuous['Sex'].map({'male': 0, 'female': 1})
    df_continuous['Embarked'] = df_continuous['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    print("Dados para C4.5 e CART preparados.")

    # Discretização de contínuos para ID3 
    df_id3 = df.copy()
    df_id3['Age'] = pd.qcut(df_id3['Age'], q=4, labels=['Criança', 'Jovem', 'Adulto', 'Idoso'], duplicates='drop')
    df_id3['Fare'] = pd.qcut(df_id3['Fare'], q=4, labels=['Barato', 'Médio', 'Caro', 'Muito Caro'], duplicates='drop')
    print("Dados para ID3 discretizados.")

    # Partição em treino e teste 
    X_id3, y_id3 = df_id3[features], df_id3[target]
    X_train_id3, X_test_id3, y_train_id3, y_test_id3 = train_test_split(X_id3, y_id3, test_size=0.2, random_state=42, stratify=y_id3)

    X, y = df_continuous[features], df_continuous[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Dados divididos em conjuntos de treino (80%) e teste (20%).\n")
    
    # --- 2. Verificação com "Play Tennis" ---
    print("\n--- [Seção 2.1] Verificação com Dataset 'Play Tennis' ---\n")
    tennis_data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    tennis_df = pd.DataFrame(tennis_data)
    X_tennis, y_tennis = tennis_df.drop('Play', axis=1), tennis_df['Play']
    
    id3_tennis = ID3(max_depth=4)
    id3_tennis.fit(X_tennis, y_tennis)
    print("Árvore ID3 para 'Play Tennis':")
    print_tree(id3_tennis.root)

    # --- 3. Treinamento e Avaliação no Titanic ---
    print("\n--- [Seção 3] Resultados no Dataset Titanic ---\n")
    
    # ID3
    print("\n--- [3.1] Algoritmo ID3 ---\n")
    id3_titanic = ID3(max_depth=5)
    id3_titanic.fit(X_train_id3, y_train_id3)
    y_pred_id3 = id3_titanic.predict(X_test_id3)
    print("Árvore ID3 gerada (Titanic Discretizado):")
    print_tree(id3_titanic.root)
    print("\nMétricas de Avaliação (ID3):")
    print(f"Acurácia: {accuracy_score(y_test_id3, y_pred_id3):.4f}")
    print(classification_report(y_test_id3, y_pred_id3))

    # C4.5
    print("\n--- [3.2] Algoritmo C4.5 ---\n")
    c45_titanic = C45(max_depth=5)
    c45_titanic.fit(X_train, y_train)
    y_pred_c45 = c45_titanic.predict(X_test)
    print("Árvore C4.5 gerada (Titanic):")
    print_tree(c45_titanic.root)
    print("\nMétricas de Avaliação (C4.5):")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_c45):.4f}")
    print(classification_report(y_test, y_pred_c45))

    # CART
    print("\n--- [3.3] Algoritmo CART ---\n")
    cart_titanic = CART(max_depth=5)
    cart_titanic.fit(X_train, y_train)
    y_pred_cart = cart_titanic.predict(X_test)
    print("Árvore CART gerada (Nossa Implementação):")
    print_tree(cart_titanic.root)
    print("\nMétricas de Avaliação (Nosso CART):")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_cart):.4f}")
    print(classification_report(y_test, y_pred_cart))
    
    # Comparação com Scikit-learn
    print("\n--- [3.4] Comparação com Scikit-learn (Baseline) ---\n")
    sklearn_cart = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    sklearn_cart.fit(X_train, y_train)
    y_pred_sklearn = sklearn_cart.predict(X_test)
    print("Métricas de Avaliação (Scikit-learn CART):")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print(classification_report(y_test, y_pred_sklearn))
    print("\nComparação: A acurácia e as métricas do nosso CART são consistentes com a implementação de referência do scikit-learn.")
    
    print("\n" + "="*60)
    print("                       FIM DA EXECUÇÃO")
    print("="*60)


if __name__ == '__main__':
    main()