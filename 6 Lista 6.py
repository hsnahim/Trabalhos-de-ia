# Questão 06
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
df = df.applymap(lambda x: 1 if x=="Sim" else 0)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

print("Itemsets Frequentes com Suporte (Questão 06):")
for index, row in frequent_itemsets.iterrows():
    print(f"{set(row['itemsets'])} -> suporte: {row['support']:.2f}")
