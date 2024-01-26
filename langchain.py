import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Função para carregar documentos de uma pasta
def carregar_documentos(diretorio):
    documentos = []
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith(".txt"):
            with open(os.path.join(diretorio, arquivo), 'r', encoding='utf-8') as file:
                texto = file.read()
                # Dividir documentos longos, se necessário
                # Por exemplo, dividir em partes de 5000 palavras
                partes = [texto[i:i+5000] for i in range(0, len(texto), 5000)]
                documentos.extend(partes)
    return documentos

# Carregar documentos da pasta "txt"
documentos = carregar_documentos("txts")

# Pré-processamento
nlp = spacy.load("en_core_web_sm")
stop_words_adicionais = {"company", "report"}  # Adicione palavras adicionais aqui
for word in stop_words_adicionais:
    nlp.Defaults.stop_words.add(word)

docs_processados = [" ".join([token.lemma_.lower() for token in nlp(doc) if not token.is_stop and not token.is_digit and not token.like_num]) for doc in documentos]

# Extração de Temas com LDA
vectorizer = TfidfVectorizer(min_df=3, max_df=0.85)  # Ajuste min_df e max_df conforme necessário
X = vectorizer.fit_transform(docs_processados)
lda = LatentDirichletAllocation(n_components=10)  # Experimente com diferentes números de temas
lda.fit(X)

# Visualização dos temas
palavras = vectorizer.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    tema_palavras = [palavras[i] for i in topic.argsort()[-10:]]
    # Filtrar termos específicos ou irrelevantes, se necessário
    tema_palavras = [palavra for palavra in tema_palavras if palavra not in stop_words_adicionais]
    print(f"Tema {i+1}: {' '.join(tema_palavras)}")
