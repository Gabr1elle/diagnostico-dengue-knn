# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Leitura dos Dados

df = pd.read_csv("dados_dengue.csv")


# Transformação dos Dados

# Convertendo valores 'sim' para 1 e 'nao' para 0
df_binario = df.replace({'sim': 1, 'nao': 0})

# Exibindo os primeiros registros transformados
print("Dados transformados:\n")
print(df_binario.head())  # Mostra os primeiros 5 registros


# Mineração de Dados


# Separando as colunas de entrada (X) e o alvo (y)
X = df_binario.drop('tem_dengue', axis=1)  # Todas menos a coluna alvo
y = df_binario['tem_dengue']               # Coluna alvo

# Dividindo em conjunto de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo KNN com k=3
modelo = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
modelo.fit(X_train, y_train)

# Realizando predições com os dados de teste
y_pred = modelo.predict(X_test)


# Avaliação do Modelo

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nAcurácia do Modelo:")
print(accuracy_score(y_test, y_pred))
