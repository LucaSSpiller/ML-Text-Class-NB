# Classificação de Textos com Naive Bayes

Passos para classificar textos usando o algoritmo Naive Bayes multinomial com `scikit-learn`.

## 1. Importações Necessárias

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
```


## Bibliotecas
- `CountVectorizer:` Converte textos em uma matriz de contagem de tokens.

- `train_test_split:` Divide dados em treinamento e teste.

- `MultinomialNB:` Algoritmo de classificação Naive Bayes multinomial.

- `accuracy_score:` Calcula a acurácia das predições.

## 2. Dados de Exemplo
```python
textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]
```
- `textos:` Lista de textos a serem classificados.

- `categorias:` Rótulos correspondentes a cada texto.

## 3. Vectorização dos Textos
```python
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)
```
- `CountVectorizer():` Cria um objeto CountVectorizer.

- `fit_transform(textos):` Transforma textos em uma matriz de contagem de tokens (X).

## 4. Divisão dos Dados
```python
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)
```
`train_test_split:` Divide dados em treinamento (50%) e teste (50%).
- `X_train, X_test:` Dados de texto para treinamento e teste.
- `y_train, y_test:` Rótulos correspondentes para treinamento e teste.

## 5. Treinamento do Classificador
```python
clf = MultinomialNB()
clf.fit(X_train, y_train)
```
- `MultinomialNB():` Cria o classificador Naive Bayes.
- `fit(X_train, y_train):` Treina o classificador com os dados de treinamento.

 ## 6. Predição e Avaliação
 ```python
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")
```
- `predict(X_test):` Prediz os rótulos dos dados de teste.
- `accuracy_score(y_test, y_pred):` Calcula a acurácia comparando rótulos reais e preditos.
- `print():` Exibe a acurácia do classificador.

### Resumo
1. Importar bibliotecas necessárias.
2. Definir dados de exemplo.
3. Vectorizar os textos.
4. Dividir dados em treinamento e teste.
5. Treinar o classificador.
6. Fazer predições e avaliar a acurácia.
