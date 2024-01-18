import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Criando um dataframe de exemplo
data = {
    'Mensagem': ['Promoção incrível! Aproveite nossas ofertas', 'Olá, como você está?', 'Ganhe dinheiro rápido!', 'Convite para a festa de aniversário'],
    'Classe': ['spam', 'não spam', 'spam', 'não spam']
}

df = pd.DataFrame(data)

# Criando um vetorizador para converter texto em vetores de contagem de palavras
vectorizer = CountVectorizer()

# Vetorizando as mensagens
X = vectorizer.fit_transform(df['Mensagem'])

# Criando o modelo Naive Bayes
model = MultinomialNB()

# Treinando o modelo com os dados
model.fit(X, df['Classe'])

# Exemplo de classificação de uma nova mensagem
#nova_mensagem = ['Venha, últimas vagas!']
nova_mensagem = ['Aproveite nossas ofertas, estão imperdíveis!']
# Vetorizando a nova mensagem
nova_mensagem_vetorizada = vectorizer.transform(nova_mensagem)

# Classificando a nova mensagem
resultado = model.predict(nova_mensagem_vetorizada)

print(resultado)  # Saída: ['spam']