import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import nltk

# Baixar stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Função para ler todos os arquivos de texto
def read_text_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Função de limpeza e pré-processamento de texto
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    return " ".join([word for word in words if word.isalpha() and word not in stop_words])

# Ler dados
folder_path = 'txts'  # Substitua pelo caminho da sua pasta
texts = read_text_files(folder_path)
df = pd.DataFrame({'text': texts})

# Limpeza e pré-processamento
df['cleaned_text'] = df['text'].apply(clean_text)

# Análise TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
tfidf_scores = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_scores = tfidf_scores.sum().sort_values(ascending=False)

# Visualização dos top N palavras
top_n = 20
word_scores.head(top_n).plot(kind='bar')
plt.show()

# Opcional: Visualização em Nuvem de Palavras
wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      min_font_size = 10).generate_from_frequencies(word_scores)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show()
