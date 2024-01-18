import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Carregar dados de saúde
data = pd.read_csv('./dados/dados_saude.csv')

# Pré-processamento dos dados
X = data.drop('Diagnostico', axis=1)  # Características (removendo a coluna de diagnóstico)
y = data['Diagnostico']  # Classe (diagnóstico)

# Vetorizar os sintomas
vectorizer = CountVectorizer()
X_sintomas = vectorizer.fit_transform(X['Sintomas'])

# Padronizar outras características numéricas (exemplo: idade, pressão arterial)
scaler = StandardScaler()
X_numerico = scaler.fit_transform(X[['Idade', 'PressaoArterial']])

# Concatenar as características vetorizadas e numéricas
X = pd.DataFrame.sparse.from_spmatrix(X_sintomas)
X = pd.concat([X, pd.DataFrame(X_numerico, columns=['Idade', 'PressaoArterial'])], axis=1)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo Naive Bayes
model = MultinomialNB()

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo: {:.2f}%".format(accuracy * 100))