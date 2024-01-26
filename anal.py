import os
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Função para carregar e preprocessar os dados
def load_data(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"): # ajuste para o formato do seu arquivo
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

# Função para preprocessar os textos
def preprocess(documents):
    stop_words = set(stopwords.words('english'))
    texts = [
        [word for word in word_tokenize(document.lower()) if word.isalpha() and word not in stop_words]
        for document in documents
    ]
    return texts

# Carrega e preprocessa os dados
directory = 'txts' # substitua pelo caminho real
documents = load_data(directory)
texts = preprocess(documents)

# Cria um dicionário e corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Modelo LDA para Análise de Tópicos
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Exibe os tópicos
topics = lda_model.print_topics(num_words=4)
for topic in topics:
    print(topic)
