import nltk
from collections import Counter
import math

# Fonction pour lire les fichiers texte
def lire_fichiers_textes(chemins_fichiers):
    docs = []
    for fichier in chemins_fichiers:
        with open(fichier, 'r', encoding='utf-8') as f:
            docs.append(f.read())
    return docs

# Fonction pour compter combien de documents contiennent chaque terme (pour calculer n_i plus tard)
def calculeNbDocument_Terme(docs, stop_words, mode):
    term_doc_count = Counter()  # Dictionnaire global pour compter les documents contenant chaque terme
    
    for doc in docs:
        seen_terms = set()
        if mode == "Split":
            tokens = [token.lower() for token in doc.split() if token.lower() not in stop_words]
        elif mode == "RegularExpression":
            ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
            tokens = [token.lower() for token in ExpReg.tokenize(doc) if token.lower() not in stop_words]
        
        for token in tokens:
            if token not in seen_terms:
                term_doc_count[token] += 1
                seen_terms.add(token)
    
    return term_doc_count

# Fonction pour calculer la fréquence et le poids des tokens dans chaque document
def calculer_frequences_poids(docs, stop_words, mode, term_doc_count):
    tokens_NumDoc = []
    N = len(docs)

    for doc_num, doc in enumerate(docs, 1):
        if mode == "Split":
            tokens = [token.lower() for token in doc.split() if token.lower() not in stop_words]
        elif mode == "RegularExpression":
            ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
            tokens = [token.lower() for token in ExpReg.tokenize(doc) if token.lower() not in stop_words]
        
        freq_dist = Counter(tokens)
        max_freq = max(freq_dist.values())
        
        for token, freq in freq_dist.items():
            n_i = term_doc_count[token]
            poids = (freq / max_freq) * math.log10(N / n_i +1)
            poids_arrondi = round(poids, 4)
            tokens_NumDoc.append((doc_num, token, freq, poids_arrondi))
    
    return tokens_NumDoc

# Fonction pour appliquer un algorithme de stemming et recalculer les fréquences et poids
from collections import defaultdict

def appliquer_stemming_et_calculer_frequences(tokens, stemmer, N):
    # Dictionnaire pour regrouper les fréquences des termes après stemming dans chaque document
    stemmed_token_freq = defaultdict(lambda: defaultdict(int))
    
    # Remplir le dictionnaire avec les fréquences après stemming
    for doc_num, token, freq, _ in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_token_freq[doc_num][stemmed_token] += freq  # Additionne les fréquences pour le même terme stemmé dans le même document

    # Recalcul de term_doc_count pour le nombre de documents contenant chaque terme stemmé
    term_doc_count = Counter()
    for terms in stemmed_token_freq.values():
        for term in terms:
            term_doc_count[term] += 1

    # Calcul de la distribution des fréquences maximales dans chaque document
    stemmed_tokens = []
    for doc_num, term_freqs in stemmed_token_freq.items():
        max_freq = max(term_freqs.values())  # Fréquence maximale pour normalisation dans ce document
        for token, freq in term_freqs.items():
            n_i = term_doc_count[token]
            poids = (freq / max_freq) * math.log10((N / n_i) + 1)
            poids_arrondi = round(poids, 4)
            stemmed_tokens.append((doc_num, token, freq, poids_arrondi))
    
    return stemmed_tokens  # La liste finale sans doublons, prête pour l'export


# Fonction pour sauvegarder les tokens dans un fichier
def sauvegarder_fichier(fichier, data, mode):
    with open(fichier, 'w', encoding='utf-8') as f:
        for doc_num, token, frequency, poid in data:
            if mode == "Inverse":
                f.write(f'{token} {doc_num} {frequency} {poid}\n')
            elif mode == "Descripteur":
                f.write(f'{doc_num} {token} {frequency} {poid}\n')

# Liste des fichiers texte à traiter
fichiers = [f'./Collections/D{i}.txt' for i in range(1, 1034)]
docs = lire_fichiers_textes(fichiers)
stop_words = nltk.corpus.stopwords.words('english')

# Tokenisation initiale et calcul des fréquences et poids
term_doc_count = calculeNbDocument_Terme(docs, stop_words, mode="Split")
tokens_with_docs = calculer_frequences_poids(docs, stop_words, mode="Split", term_doc_count=term_doc_count)
tokens_tries = sorted(tokens_with_docs, key=lambda x: (x[0], x[1]))
sauvegarder_fichier('DescripteurSplit.txt', tokens_tries, mode="Descripteur")
tokens_tries_inverse = sorted(tokens_with_docs, key=lambda x: (x[1]))
sauvegarder_fichier('InverseSplit.txt', tokens_tries_inverse, mode="Inverse")

# Calcul des fréquences et poids après PorterStemmer
Porter = nltk.PorterStemmer()
# Appel de la fonction avec trois arguments uniquement, sans `term_doc_count`
tokens_porter = appliquer_stemming_et_calculer_frequences(tokens_tries, Porter, len(docs))
tokens_tries_porter = sorted(tokens_porter, key=lambda x: (x[1]))
sauvegarder_fichier('DescripteurSplitPorter.txt', tokens_porter,  mode="Descripteur")
sauvegarder_fichier('InverseSplitPorter.txt', tokens_tries_porter, mode="Inverse")

# Calcul des fréquences et poids après LancasterStemmer
Lancaster = nltk.LancasterStemmer()
# Appel de la fonction avec trois arguments uniquement, sans `term_doc_count`
tokens_lancaster = appliquer_stemming_et_calculer_frequences(tokens_tries, Lancaster, len(docs))
tokens_tries_lancaster = sorted(tokens_lancaster, key=lambda x: (x[1]))
sauvegarder_fichier('DescripteurSplitLancaster.txt', tokens_lancaster, mode="Descripteur")
sauvegarder_fichier('InverseSplitLancaster.txt', tokens_tries_lancaster, mode="Inverse")

# Tokenisation initiale et calcul des fréquences et poids
term_doc_count = calculeNbDocument_Terme(docs, stop_words, mode="RegularExpression")
tokens_with_docs = calculer_frequences_poids(docs, stop_words, mode="RegularExpression", term_doc_count=term_doc_count)
tokens_tries = sorted(tokens_with_docs, key=lambda x: (x[0], x[1]))
sauvegarder_fichier('DescripteurToken.txt', tokens_tries, mode="Descripteur")
tokens_tries_inverse = sorted(tokens_with_docs, key=lambda x: (x[1]))
sauvegarder_fichier('InverseToken.txt', tokens_tries_inverse, mode="Inverse")

# Calcul des fréquences et poids après PorterStemmer
Porter = nltk.PorterStemmer()
# Appel de la fonction avec trois arguments uniquement, sans `term_doc_count`
tokens_porter = appliquer_stemming_et_calculer_frequences(tokens_tries, Porter, len(docs))
tokens_tries_porter = sorted(tokens_porter, key=lambda x: (x[1]))
sauvegarder_fichier('DescripteurTokenPorter.txt', tokens_porter,  mode="Descripteur")
sauvegarder_fichier('InverseTokenPorter.txt', tokens_tries_porter, mode="Inverse")

# Calcul des fréquences et poids après LancasterStemmer
Lancaster = nltk.LancasterStemmer()
# Appel de la fonction avec trois arguments uniquement, sans `term_doc_count`
tokens_lancaster = appliquer_stemming_et_calculer_frequences(tokens_tries, Lancaster, len(docs))
tokens_tries_lancaster = sorted(tokens_lancaster, key=lambda x: (x[1]))
sauvegarder_fichier('DescripteurTokenLancaster.txt', tokens_lancaster, mode="Descripteur")
sauvegarder_fichier('InverseTokenLancaster.txt', tokens_tries_lancaster, mode="Inverse")
