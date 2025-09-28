# Questão 07
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = {
    'Leite':    ['Não','Sim','Não','Sim','Não','Não','Não','Não','Não','Não'],
    'Café':     ['Sim','Não','Sim','Sim','Não','Não','Não','Sim','Não','Não'],
    'Cerveja':  ['Não','Sim','Sim','Não','Sim','Não','Sim','Não','Não','Não'],
    'Pão':      ['Sim','Sim','Sim','Não','Sim','Sim','Sim','Não','Não','Não'],
    'Manteiga': ['Sim','Não','Não','Sim','Não','Sim','Não','Não','Não','Não'],
    'Arroz':    ['Não','Não','Sim','Não','Não','Não','Não','Não','Sim','Sim'],
    'Feijão':   ['Não','Não','Não','Não','Não','Não','Não','Sim','Sim','Não'],
}

df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df)
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

print("Regras de Associação considerando ausência de itens (Questão 07):")
for idx, row in rules.iterrows():
    antecedent = ', '.join(row['antecedents'])
    consequent = ', '.join(row['consequents'])
    print(f"Se {antecedent} então {consequent} (confiança: {row['confidence']:.2f})")