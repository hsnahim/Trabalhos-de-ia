from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

modelo = DecisionTreeClassifier(criterion='entropy',
                                max_depth=4,
                                random_state=42)
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
print("Acurácia:", accuracy_score(y_teste, previsoes))

regras = export_text(modelo, feature_names=list(X.columns))
print(regras)
