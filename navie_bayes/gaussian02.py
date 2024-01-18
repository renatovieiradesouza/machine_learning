import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Criando o dataframe de exemplo
dados = {
    'texto': [
        'oferta incrível! compre agora',
        'ganhe dinheiro rápido e fácil',
        'promoção exclusiva por tempo limitado',
        'trabalhe em casa e tenha renda extra',
        'sua encomenda foi entregue',
        'confirmação do seu pedido'
    ],
    'spam': ['spam', 'spam', 'spam', 'spam', 'não spam', 'não spam']
}

df = pd.DataFrame(dados)

# Convertendo o texto em recursos numéricos usando Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['texto'])
y = df['spam']

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Criando o modelo Naive Bayes Gaussiano
modelo = GaussianNB()

# Treinando o modelo
modelo.fit(X_train.toarray(), y_train)

# Fazendo previsões
y_pred = modelo.predict(X_test.toarray())

# Calculando a acurácia do modelo
acuracia = accuracy_score(y_test, y_pred)
print('Acurácia:', acuracia)