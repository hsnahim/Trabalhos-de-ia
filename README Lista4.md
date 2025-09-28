# **Documentação: Implementação de Árvores de Decisão (ID3, C4.5, CART)**

Este script Python oferece uma implementação do zero dos algoritmos de árvore de decisão **ID3, C4.5 e CART** para fins de classificação. Ele inclui funções para calcular métricas de impureza, as classes que definem cada algoritmo, e uma função `main` que demonstra o seu uso no famoso dataset do Titanic, comparando os resultados com a implementação de referência da biblioteca `scikit-learn`.

## **Estrutura do Arquivo**

O código é organizado nas seguintes seções:

1.  **Funções de Cálculo de Impureza e Ganho:** Funções matemáticas que formam a base dos critérios de divisão das árvores.
2.  **Classe `Node`:** Estrutura de dados fundamental que representa um nó (de decisão ou folha) na árvore.
3.  **Classes dos Algoritmos (`ID3`, `C45`, `CART`):** Onde cada algoritmo é implementado, contendo a lógica para construir a árvore (`fit`) e fazer previsões (`predict`).
4.  **Função Auxiliar `print_tree`:** Uma função para visualizar a estrutura da árvore gerada de forma legível no console.
5.  **Função `main`:** O ponto de entrada do script, responsável por preparar os dados, treinar os modelos, avaliá-los e apresentar os resultados.

-----

## **1. Funções de Cálculo de Impureza e Ganho**

Estas funções quantificam o "caos" ou a "pureza" de um conjunto de dados. O objetivo dos algoritmos é maximizar a redução da impureza a cada divisão.

#### `entropy(y)`

  * **Propósito:** Calcula a Entropia de Shannon para um conjunto de rótulos `y`.
  * **Conceito:** A entropia mede o grau de incerteza ou desordem. Uma entropia de 0 significa que todos os exemplos pertencem à mesma classe (pureza máxima).
  * **Usado por:** ID3 (para calcular o Ganho de Informação).

#### `gini_index(y)`

  * **Propósito:** Calcula o Índice Gini (ou Impureza Gini).
  * **Conceito:** Mede a probabilidade de um elemento, escolhido aleatoriamente, ser classificado incorretamente. Assim como a entropia, um valor de 0 indica pureza máxima. É computacionalmente mais eficiente que a entropia.
  * **Usado por:** CART.

#### `information_gain(y, y_subsets)`

  * **Propósito:** Calcula o Ganho de Informação.
  * **Conceito:** Mede a redução na entropia após dividir um conjunto `y` em vários subconjuntos `y_subsets`. O ID3 escolhe o atributo que fornece o maior ganho de informação.
  * **Fórmula:** $IG(Y, A) = \text{Entropy}(Y) - \sum_{v \in \text{Values}(A)} \frac{|Y_v|}{|Y|} \cdot \text{Entropy}(Y_v)$

#### `gain_ratio(y, y_subsets)`

  * **Propósito:** Calcula a Razão de Ganho.
  * **Conceito:** É uma modificação do Ganho de Informação que penaliza atributos com muitos valores distintos (como um ID), evitando o viés do ID3. Ele normaliza o ganho de informação pela "entropia da divisão" (Split Info).
  * **Usado por:** C4.5.

#### `gini_gain(y, y_subsets)`

  * **Propósito:** Calcula o Ganho Gini, ou a redução na impureza Gini.
  * **Conceito:** Similar ao Ganho de Informação, mas usa o Índice Gini em vez da Entropia. O CART busca a divisão que maximiza essa redução de impureza.

-----

## **2. Classe `Node`**

Esta classe é o bloco de construção fundamental da árvore.

```python
class Node:
    """
    Representa um nó na árvore de decisão.
    Pode ser um nó de decisão (com uma pergunta) ou um nó folha (com uma previsão).
    """
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature       # Atributo usado para a divisão
        self.threshold = threshold   # Limiar de decisão
        self.children = children     # Nós filhos (dicionário)
        self.value = value           # Valor da classe, se for um nó folha

    def is_leaf_node(self):
        # Um nó é uma folha se ele tem um valor de previsão.
        return self.value is not None
```

  * Um **nó de decisão** terá `feature`, `threshold` e `children` definidos.
  * Um **nó folha** terá apenas `value` definido, que representa a previsão final daquele ramo da árvore.

-----

## **3. Classes dos Algoritmos**

Cada classe implementa a interface padrão `fit(X, y)` e `predict(X)`.

### **`ID3`**

  * **Critério de Divisão:** Ganho de Informação.
  * **Tipo de Atributos:** Lida exclusivamente com **atributos categóricos**. Dados contínuos precisam ser discretizados antes do treino (como feito na função `main`).
  * **Tipo de Divisão:** Cria uma ramificação (filho) para **cada valor único** do atributo escolhido (divisão multi-ramificada).

### **`C45`**

  * **Critério de Divisão:** Razão de Ganho.
  * **Tipo de Atributos:** Lida nativamente com atributos **categóricos e contínuos**.
  * **Tipo de Divisão:**
      * Para atributos categóricos: Cria uma divisão multi-ramificada, similar ao ID3.
      * Para atributos contínuos: Encontra o melhor limiar (`threshold`) e cria uma **divisão binária** (`<= threshold` e `> threshold`).

### **`CART` (Classification and Regression Tree)**

  * **Critério de Divisão:** Ganho Gini (para classificação).
  * **Tipo de Atributos:** Lida com atributos categóricos e contínuos.
  * **Tipo de Divisão:** Cria **exclusivamente divisões binárias**.
      * Para atributos contínuos: Encontra o melhor limiar, assim como o C4.5.
      * Para atributos categóricos: Agrupa os valores em dois superconjuntos para criar uma divisão binária (ex: `{A, C}` vs. `{B}`).

-----

## **4. Função Auxiliar `print_tree`**

```python
def print_tree(node, spacing=""):
```

  * **Propósito:** Percorre a árvore recursivamente a partir do nó raiz e imprime sua estrutura de forma indentada, facilitando a visualização e interpretação do modelo treinado.

-----

## **5. Função `main`**

Esta é a função principal que orquestra todo o processo de teste e avaliação.

  * **Seção 1: Preparação dos Dados**

    1.  **Carregamento:** O dataset do Titanic é carregado de uma URL.
    2.  **Limpeza:** Valores ausentes nas colunas `Age`, `Embarked` e `Fare` são preenchidos (imputação) com a mediana ou a moda.
    3.  **Preparação para C4.5 e CART:** Cria um DataFrame (`df_continuous`) onde as features categóricas (`Sex`, `Embarked`) são convertidas para valores numéricos.
    4.  **Preparação para ID3:** Cria um DataFrame separado (`df_id3`) onde as features contínuas (`Age`, `Fare`) são discretizadas em faixas (ex: 'Criança', 'Jovem', etc.), pois o ID3 não lida com números.
    5.  **Divisão:** Ambos os conjuntos de dados são divididos em 80% para treino e 20% para teste.

  * **Seção 2: Verificação com "Play Tennis"**

      * Um pequeno e clássico dataset é usado para um "sanity check", garantindo que a lógica do ID3 está funcionando corretamente ao gerar uma árvore conhecida.

  * **Seção 3: Treinamento e Avaliação no Titanic**

      * Para cada um dos três algoritmos (ID3, C4.5, CART):
        1.  O modelo é instanciado com uma profundidade máxima (`max_depth=5`) para evitar overfitting.
        2.  O modelo é treinado com os dados de treino apropriados (`.fit()`).
        3.  A árvore gerada é impressa no console com `print_tree()`.
        4.  O modelo faz previsões nos dados de teste (`.predict()`).
        5.  A acurácia e um relatório de classificação completo (precisão, recall, f1-score) são calculados e exibidos.

  * **Seção 4: Comparação com Scikit-learn**

      * Um `DecisionTreeClassifier` da biblioteca `scikit-learn` é treinado com os mesmos parâmetros (`criterion='gini'`, `max_depth=5`).
      * Suas métricas de avaliação são impressas para servirem como um *baseline*, permitindo verificar se a implementação manual do CART obteve um desempenho comparável e consistente.

## **Como Executar o Código**

1.  Certifique-se de ter as bibliotecas necessárias instaladas:
    ```bash
    pip install numpy pandas scikit-learn
    ```
2.  Salve o código como um arquivo Python (ex: `arvores_decisao.py`).
3.  Execute-o a partir do seu terminal:
    ```bash
    python arvores_decisao.py
    ```

O script irá imprimir todo o processo, desde a preparação dos dados até as árvores geradas e os relatórios de performance de cada modelo.