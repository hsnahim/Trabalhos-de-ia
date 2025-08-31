import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

base = pd.read_csv('restaurante.csv', sep=';')

mapeamento_cliente = {'Nenhum': 0, 'Alguns': 1, 'Cheio': 2}
base['Cliente'] = base['Cliente'].map(mapeamento_cliente)

X = base.drop('Conclusao', axis=1)
y = base['Conclusao']

X_treino, X_teste, y_treino, y_teste = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

with open('restaurante.pkl', 'wb') as f:
    pickle.dump([X_treino, X_teste, y_treino, y_teste], f)
