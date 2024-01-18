import pandas as pd

data = {'PraticaExercicio': [True, False, False, True],
        'AssisteEsportesTV': [True, True, False, True],
        'GostaEsporte': ['Sim', 'Não', 'Não', 'Sim']}

df = pd.DataFrame(data)

from sklearn.naive_bayes import BernoulliNB

# Separar as características (X) da classe alvo (y)
X = df[['PraticaExercicio', 'AssisteEsportesTV']]
y = df['GostaEsporte']

# Criar um classificador Naive Bayes na forma Bernoulli
classifier = BernoulliNB()

# Treinar o classificador
classifier.fit(X, y)

# Fazer previsões para novos dados
novo_dado = [[True, False]]
previsao = classifier.predict(novo_dado)

print(previsao)