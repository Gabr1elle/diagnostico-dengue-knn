# Diagnóstico de Dengue com KNN 🦟🧠

Este projeto tem como objetivo prever casos de dengue com base em sintomas clínicos usando técnicas de Mineração de Dados e o algoritmo KNeighborsClassifier (KNN).

## 📌 Descrição

Aplicação prática de aprendizado supervisionado com Python para classificação binária (`dengue` ou `não dengue`) a partir de uma base de dados com sintomas. Segui o fluxo básico de um processo de Mineração de Dados:

- Transformação dos dados (valores 'sim' e 'nao' para 1 e 0)
- Separação em dados de treino e teste
- Treinamento com KNN
- Avaliação com matriz de confusão e acurácia

## 🛠 Tecnologias Utilizadas

- Python 3
- pandas
- scikit-learn

## 📂 Estrutura

integrated-project-3/
├── index.py ← Código principal
├── dados_dengue.csv ← Base de dados
└── venv/ ← Ambiente virtual


## ▶️ Como Executar

1. Clone o repositório:
   
git clone https://github.com/Gabr1elle/diagnostico-dengue-knn.git
cd diagnostico-dengue-knn

2. Crie o ambiente virtual:
   
python -m venv venv
source venv/bin/activate  # No Linux/Mac

3. Instale as dependências:

pip install pandas scikit-learn

4. Execute o código:

python index.py

# 📊 Resultado

Após treinar e testar o modelo, foram exibidos:

✅ A matriz de confusão

📈 A acurácia do modelo de classificação

Essas métricas permitem avaliar o desempenho do classificador na detecção de casos de dengue.

# 📚 Aprendizados

- Uso do ambiente virtual (venv)

- Transformação de dados categóricos para numéricos

- Aplicação do algoritmo KNN

- Avaliação de modelos com confusion_matrix e accuracy_score

Feito como parte do Projeto Integrado III para a disciplina de IA e Ciência de Dados. 💻✨
