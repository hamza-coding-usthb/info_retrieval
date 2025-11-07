from tkinter import *
from tkinter import ttk
import tkinter as tk
import nltk
from collections import defaultdict
from collections import Counter
import math
from math import sqrt
import re 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def preprocessed_requete(query):

    query = query.lower()
    stemmed_tokens = []
    stop_words = nltk.corpus.stopwords.words('english')
    if(x.get()==0):
        tokens = [token.lower() for token in query.split() if token.lower() not in stop_words]
    elif(x.get()==1):
        ExpReg = nltk.RegexpTokenizer(r"(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,\-]\d+)?%?|\w+(?:[\-/]\w+)*")
        tokens = [token.lower() for token in ExpReg.tokenize(query) if token.lower() not in stop_words]
    if(y.get()==0):
        return tokens
    else:
        if(y.get()==1) :
            Porter = nltk.PorterStemmer()
            for token in tokens:
                t = Porter.stem(token)
                stemmed_tokens.append(t)
        elif(y.get()==2) :
            Lancaster = nltk.LancasterStemmer()
            for token in tokens:
                t = Lancaster.stem(token)
                stemmed_tokens.append(t)
        return stemmed_tokens
    
def search():

    print("model choice",k.get())

    print("measures choice",dq.get())

    results = []

    # Dictionnaire pour associer les options aux noms de fichiers
    file_mapping_descripteur = {
        (0, 0, 1): 'DescripteurSplit.txt',
        (1, 0, 1): 'DescripteurToken.txt',
        (0, 1, 1): 'DescripteurSplitPorter.txt',
        (1, 1, 1): 'DescripteurTokenPorter.txt',
        (0, 2, 1): 'DescripteurSplitLancaster.txt',
        (1, 2, 1): 'DescripteurTokenLancaster.txt',
    }

    file_mapping_inverse = {
        (0, 0, 0): 'InverseSplit.txt',
        (1, 0, 0): 'InverseToken.txt',
        (0, 1, 0): 'InverseSplitPorter.txt',
        (1, 1, 0): 'InverseTokenPorter.txt',
        (0, 2, 0): 'InverseSplitLancaster.txt',
        (1, 2, 0): 'InverseTokenLancaster.txt',
    }

    nettoyer_frame_plot()
    
    # Si un fichier est sélectionné, ouvre-le et recherche le terme
    if (z.get() == 0):
        # Récupère le fichier en fonction des valeurs x, y, z
        selected_file = file_mapping_inverse.get((x.get(), y.get(), z.get()))
    elif (z.get() == 1):
        # Récupère le fichier en fonction des valeurs x, y, z
        selected_file = file_mapping_descripteur.get((x.get(), y.get(), z.get()))
    if(dq.get()==0):
        update_results_evaluation("", "", "", "", "")
        query = search_entry.get()
        print(f"Searching for: {query}")
        if(t.get()==1):
            query = preprocessed_requete(query)
            print(query)
            if selected_file:
                with open(selected_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        words = line.split()
                        for q in query:
                            if q == words[0]:
                                results.append(words)
            print(results)
            return results
        elif(t.get()==0):
            if(k.get()==0):#vector space model
                query = preprocessed_requete(query)
                print(query)
                poids_docs = getPoidDocs(selected_file)
                if(selected_option=="Scalar Product"):
                    scores = calculer_RSV_requete(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de pertinence : {score}")
                elif(selected_option=="Cosine Measure"):
                    scores = calculer_RSV_cosinus(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de Cosine : {score}")
                elif(selected_option=="Jaccard Measure"):
                    scores = calculer_RSV_formule_Jaccard(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de Jaccard : {score}")
                return scores
            elif(k.get()==1):#probalistic model
                query = preprocessed_requete(query)
                print(query)
                freq_docs = getFreqDocs(selected_file)
                term_doc_count = calculeNbDocument_Terme(selected_file, query)
                avl = calcule_avdl(selected_file)
                scores = calculer_RSV_probabiliste(float(entry_k.get()), float(entry_b.get()), freq_docs, query, term_doc_count, avl)
                for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de proabilité : {score}")
                return scores
            elif(k.get()==2):#Boolean model
                print(query)
                if(validate_boolean_query(query)):
                    #query = preprocessed_requete(query)
                    freq_docs = getFreqDocs(selected_file)
                    rst = boolean_model(freq_docs, query)
                    print("rst=",rst)
                    print(validate_boolean_query(query))
                    return rst
    elif(dq.get()==1):  
        update_results_evaluation("", "", "", "", "")
        fichier_path = 'Queries'
        ligne = lire_ligne_i(fichier_path, int(spinbox.get())-1) 
        # Insérer la ligne dans le champ de texte 'search_entry'
        search_entry.delete(0, 'end')  # Effacer tout texte existant dans l'Entry
        search_entry.insert(0, ligne)  # Insérer la ligne lue dans l'Entry
        fichier_jugement = "Judgements"
        jugement = get_from_judgment_doc_req(fichier_jugement, int(spinbox.get()))
        query = search_entry.get()
        print(f"Searching for: {query}")
        if(t.get()==1):
            query = preprocessed_requete(query)
            print(query)
            if selected_file:
                with open(selected_file, 'r', encoding='utf-8') as file:
                    for line in file:
                        words = line.split()
                        for q in query:
                            if q == words[0]:
                                results.append(words)
            return results
        elif(t.get()==0):
            if(k.get()==0):#vector space model
                query = preprocessed_requete(query)
                print(query)
                poids_docs = getPoidDocs(selected_file)
                if(selected_option=="Scalar Product"):
                    scores = calculer_RSV_requete(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de pertinence : {score}")
                elif(selected_option=="Cosine Measure"):
                    scores = calculer_RSV_cosinus(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de Cosine : {score}")
                elif(selected_option=="Jaccard Measure"):
                    scores = calculer_RSV_formule_Jaccard(poids_docs,query)
                    for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de Jaccard : {score}")
                precision, rappel, f_score, p5, p10 = calculer_precision_recall_fscore(scores, jugement)
                update_results_evaluation(precision, p5, p10, rappel, f_score)
                calculer_precision_rappel_reel(scores, jugement)
                # Étape 3 : Calcul des deux tableaux
                precision_reelle, rappel_reel = calculer_precision_rappel_reel(scores, jugement)
                rappel_interpole, precision_interpole = interpolation_precision(precision_reelle, rappel_reel)
                tracer_courbe(rappel_interpole, precision_interpole)
                # Affichage des deux tableaux
                print("Tableau des valeurs réelles :")
                print("Model Results | Précision | Rappel")
                retrieved_docs = extraire_numeros(scores)
                for doc, p, r in zip(retrieved_docs, precision_reelle, rappel_reel):
                    print(f"{doc:<13} | {p:<9} | {r}")

                print("\nTableau des valeurs interpolées :")
                print("Rappel | Précision")
                for r, p in zip(rappel_interpole, precision_interpole):
                    print(f"{r:<6} | {p}")
                return scores
            elif(k.get()==1):#probalistic model
                query = preprocessed_requete(query)
                print(query)
                freq_docs = getFreqDocs(selected_file)
                term_doc_count = calculeNbDocument_Terme(selected_file, query)
                avl = calcule_avdl(selected_file)
                scores = calculer_RSV_probabiliste(float(entry_k.get()), float(entry_b.get()), freq_docs, query, term_doc_count, avl)
                for doc_num, score in scores:
                        print(f"Document {doc_num} - Score de proabilité : {score}")
                precision, rappel, f_score, p5, p10 = calculer_precision_recall_fscore(scores, jugement)
                update_results_evaluation(precision, p5, p10, rappel, f_score)
                calculer_precision_rappel_reel(scores, jugement)
                # Étape 3 : Calcul des deux tableaux
                precision_reelle, rappel_reel = calculer_precision_rappel_reel(scores, jugement)
                rappel_interpole, precision_interpole = interpolation_precision(precision_reelle, rappel_reel)
                tracer_courbe(rappel_interpole, precision_interpole)
                # Affichage des deux tableaux
                print("Tableau des valeurs réelles :")
                print("Model Results | Précision | Rappel")
                retrieved_docs = extraire_numeros(scores)
                for doc, p, r in zip(retrieved_docs, precision_reelle, rappel_reel):
                    print(f"{doc:<13} | {p:<9} | {r}")

                print("\nTableau des valeurs interpolées :")
                print("Rappel | Précision")
                for r, p in zip(rappel_interpole, precision_interpole):
                    print(f"{r:<6} | {p}")
                return scores
            elif(k.get()==2):#Boolean model
                print(query)
                if(validate_boolean_query(query)):
                    #query = preprocessed_requete(query)
                    freq_docs = getFreqDocs(selected_file)
                    rst = boolean_model(freq_docs, query)
                    print("rst=",rst)
                    print(validate_boolean_query(query))
                    return rst   

        
def validate_boolean_query(query):
    # Supprime les espaces en trop
    query = query.strip()

    # Vérifie les caractères interdits (parenthèses, caractères spéciaux, etc.)
    if re.search(r'[()<>]', query):
        return False

    # Définir les termes autorisés : tout mot sauf les mots-clés réservés AND, OR, NOT
    terme = r"(?!AND|OR|NOT\b)[a-zA-Z_][a-zA-Z0-9_]*"

    # Construire la regex pour une formule logique valide
    # Syntaxe : Terme (opérateur Terme)* où l'opérateur peut être AND, OR, ou NOT au bon endroit
    pattern = fr"^(NOT\s+)?{terme}(\s+(AND|OR)\s+(NOT\s+)?{terme})*$"

    # Vérifie si la formule correspond au pattern
    if not re.fullmatch(pattern, query.upper()):
        return False

    # Vérification supplémentaire de structure logique
    tokens = query.upper().split()
    prev = None
    operators = {"AND", "OR", "NOT"}
    for token in tokens:
        if token in operators:
            if prev in operators and token != "NOT":  # Deux opérateurs consécutifs, sauf NOT
                return False
        else:  # Si c'est un terme
            if prev and prev not in operators:  # Deux termes consécutifs sans opérateur
                return False
        prev = token

    # Si tout passe, la requête est valide
    return True

def boolean_model(freq_docs, query):
    """
    Implémente un modèle booléen pour une requête donnée avec gestion explicite des opérateurs composés.
    Les opérateurs ont la priorité suivante : NOT > AND > OR, avec traitement particulier de ANDNOT et ORNOT.

    Args:
        freq_docs (dict): Dictionnaire {doc_num: {token: freq}}.
        query (str): Requête booléenne sous forme de chaîne de caractères.

    Returns:
        list: Liste des numéros de documents pertinents.
    """

    # Priorité des opérateurs
    operator_priority = {"not": 3, "and": 2, "or": 1, "andnot": 2, "ornot": 1}

    # Créer un index inverse des termes vers les documents
    inverted_index = defaultdict(set)  # {term: {doc_num}}
    for doc_num, terms_freq in freq_docs.items():
        for term in terms_freq:
            inverted_index[term].add(doc_num)

    # Préparer la requête : remplacer les combinaisons complexes
    query = query.replace("AND NOT", "ANDNOT").replace("OR NOT", "ORNOT")
    query_terms = parse_boolean_query(query)
    query_terms = [term.lower() for term in query_terms]

    # Initialisation des documents pertinents
    relevant_docs = set(freq_docs.keys())  # Commence avec tous les documents

    # Stacks pour les opérateurs et les valeurs
    operator_stack = []
    value_stack = []

    def apply_operator(operator, operand1, operand2=None):
        """Applique un opérateur logique à des ensembles."""
        if operator == "not":
            return relevant_docs - operand1
        elif operator == "and":
            return operand1 & operand2
        elif operator == "or":
            return operand1 | operand2
        elif operator == "andnot":
            return operand1 & (relevant_docs - operand2)
        elif operator == "ornot":
            return operand1 | (relevant_docs - operand2)
        else:
            raise ValueError(f"Opérateur inconnu : {operator}")

    def resolve_stack():
        """Résout l'opérateur au sommet de la pile d'opérateurs."""
        operator = operator_stack.pop()
        if operator == "not":
            operand = value_stack.pop()
            value_stack.append(apply_operator(operator, operand))
        else:
            operand2 = value_stack.pop()
            operand1 = value_stack.pop()
            value_stack.append(apply_operator(operator, operand1, operand2))

    # Analyse de la requête avec respect des priorités
    i = 0
    while i < len(query_terms):
        term = query_terms[i]
        if term in operator_priority:
            if term == "not":  # Évaluer immédiatement NOT
                i += 1
                next_term = query_terms[i]
                next_term = preprocessed_requete(next_term)[0]
                value_stack.append(relevant_docs - inverted_index.get(next_term, set()))
            else:
                # Traiter les priorités des opérateurs
                while (
                    operator_stack
                    and operator_priority[operator_stack[-1]] >= operator_priority[term]
                ):
                    resolve_stack()
                operator_stack.append(term)
        else:
            # Pré-traitement du terme et ajout à la pile de valeurs
            term = preprocessed_requete(term)[0]
            value_stack.append(inverted_index.get(term, set()))
        i += 1

    # Résolution finale de la pile
    while operator_stack:
        resolve_stack()

    # Résultat final
    result_docs = value_stack.pop() if value_stack else set()
    return list(result_docs)  # Retourner la liste des documents pertinents

def parse_boolean_query(query):
    # On se débarrasse des parenthèses et des espaces en trop
    query = query.replace('(', ' ( ').replace(')', ' ) ').strip()
    # Remplacer les opérateurs par des espaces
    query = re.sub(r'\s+(AND|OR|NOT)\s+', r' \1 ', query.upper())
    return query.split()
  
def getPoidDocs(selected_file):
    poids_docs = defaultdict(dict)  # Structure {doc_num: {token: poids}}
    
    with open(selected_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, doc_num, _, poids = line.strip().split()
            doc_num = int(doc_num)  # Convertir le numéro de document en entier
            poids = float(poids)    # Convertir le poids en flottant
            
            poids_docs[doc_num][token] = poids  # Ajouter le poids du token dans le document
    
    return poids_docs

# Fonction pour récupérer les fréquences des termes dans les documents
def getFreqDocs(selected_file):
    freq_docs = defaultdict(dict)  # Structure {doc_num: {token: freq}}
    with open(selected_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, doc_num, freq, _ = line.strip().split()
            doc_num = int(doc_num)  # Convertir en entier
            freq = int(freq)  # Convertir la fréquence en entier
            freq_docs[doc_num][token] = freq
    return freq_docs

# Fonction pour créer et configurer le tableau
def create_treeview():
    global tree  # Rendre `tree` accessible dans la fonction `display_results`
    
    # Détruire un tableau existant pour éviter des doublons
    for widget in Window.place_slaves():
        if isinstance(widget, ttk.Treeview):
            widget.destroy()

    # Vérifier la valeur de `selected_option` et configurer le tableau
    if t.get() == 1:
        tree = ttk.Treeview(Window, columns=("N°", "Terme", "N°doc", "Freq", "Poids"), show="headings", height=12)
        tree.heading("N°", text="N°")
        tree.heading("Terme", text="Terme")
        tree.heading("N°doc", text="N°doc")
        tree.heading("Freq", text="Freq")
        tree.heading("Poids", text="Poids")
        tree.column("N°", width=40)
        tree.column("Terme", width=120)
        tree.column("N°doc", width=80)
        tree.column("Freq", width=80)
        tree.column("Poids", width=80)
    else:
        tree = ttk.Treeview(Window, columns=("N°", "Relevance"), show="headings", height=12)
        tree.heading("N°", text="N°")
        tree.heading("Relevance", text="Relevance")
        tree.column("N°", width=20)
        tree.column("Relevance", width=20)
    
    # Placer le tableau dans la fenêtre
    tree.place(x=20, y=210, width=516)

# Fonction pour afficher les résultats dans le tableau
def display_results(results):
    # Effacer les anciennes lignes dans la table
    for i in tree.get_children():
        tree.delete(i)
    
    # Ajouter les nouvelles lignes dans la table
    if t.get() == 1:
        for idx, row in enumerate(results):
            tree.insert('', 'end', values=(idx + 1, row[0], row[1], row[2], row[3]))
    else:
        if k.get()==2 :
            for doc_num in results:
                tree.insert('', 'end', values=(doc_num, "YES"))
        else:
            for doc_num, score in results:
                if(k.get()==1):
                    formatted_score = f"{score:.4f}"
                    tree.insert('', 'end', values=(doc_num, formatted_score))
                else:
                    if(score != 0):
                        formatted_score = f"{score:.4f}"
                        tree.insert('', 'end', values=(doc_num, formatted_score))

def on_search_click():
    # Appeler la fonction de recherche et afficher les résultats
    results = search()
    create_treeview()
    display_results(results)

# Fonction pour calculer le score RSV d'une requête pour chaque document
def calculer_RSV_requete(poids_docs, requete):
    # Calcul du score RSV pour chaque document
    scores = {}
    for doc_num, terms_poids in poids_docs.items():
        score = sum(terms_poids.get(term, 0) for term in requete)

        if(score != 0):
            score = round(score, 4)
            scores[doc_num] = score
    
    scores_RSV_trie = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return scores_RSV_trie

# Fonction pour calculer le score RSV avec la formule cosinus
def calculer_RSV_cosinus(poids_docs, requete):

    scores = {}
    
    # Préparer les poids de la requête
    poids_requete = defaultdict(float)
    for terme in requete:
        poids_requete[terme] += 1.0  # On peut pondérer différemment si besoin

    # Calcul pour chaque document
    for doc_num, termes_poids in poids_docs.items():
        # Calcul de ∑ vi * wi
        numerateur = sum(
            poids_requete[terme] * termes_poids.get(terme, 0)
            for terme in requete
        )
        
        # Calcul des normes (√∑ vi² et √∑ wi²)
        norme_requete = sqrt(sum(v ** 2 for v in poids_requete.values()))
        norme_document = sqrt(sum(w ** 2 for w in termes_poids.values()))
        
        # Calcul final du score RSV
        if norme_requete > 0 and norme_document > 0:
            score = numerateur / (norme_requete * norme_document)
        else:
            score = 0.0
        
        if(score != 0):
            score = round(score, 4)
            scores[doc_num] = score
    
    # Trier les scores par ordre décroissant
    scores_RSV_trie = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return scores_RSV_trie

def calculer_RSV_formule_Jaccard(poids_docs, requete):
    
    scores = {}
    
    # Préparer les poids de la requête
    poids_requete = defaultdict(float)
    for terme in requete:
        poids_requete[terme] += 1.0  # Pondération simple des termes de la requête (TF simple)

    # Calcul pour chaque document
    for doc_num, termes_poids in poids_docs.items():
        # Calcul de ∑ vi * wi
        numerateur = sum(
            poids_requete[terme] * termes_poids.get(terme, 0)
            for terme in requete
        )
        
        # Calcul de ∑ vi^2 et ∑ wi^2
        somme_v_carres = sum(v ** 2 for v in poids_requete.values())
        somme_w_carres = sum(w ** 2 for w in termes_poids.values())
        
        # Calcul du dénominateur
        denominateur = somme_v_carres + somme_w_carres - numerateur
        
        # Calcul final du score RSV
        if denominateur != 0:
            score = numerateur / denominateur
        else:
            score = 0.0
        
        if(score != 0.0):
            # Arrondir le score à 4 chiffres après la virgule
            scores[doc_num] = round(score, 4)
    
    # Trier les scores par ordre décroissant
    scores_RSV_trie = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return scores_RSV_trie

def calcule_avdl(selected_file):
    total_terms = 0  # Nombre total de termes dans tous les documents
    doc_count = 0    # Nombre total de documents

    freq_docs = getFreqDocs(selected_file)  # Récupérer les fréquences des termes par document
    for doc_num, terms_freq in freq_docs.items():
        total_terms += sum(terms_freq.values())  # Longueur totale du document (somme des fréquences des termes)
        doc_count += 1  # Compter le document

    # Calculer et retourner la taille moyenne des documents
    return total_terms / doc_count if doc_count > 0 else 0

# Fonction pour calculer le nombre de documents contenant chaque terme
def calculeNbDocument_Terme(selected_file, requete):
    term_doc_count = Counter()
    with open(selected_file, 'r', encoding='utf-8') as f:
        for line in f:
            token, doc_num, _, _ = line.strip().split()
            if token in requete:
                term_doc_count[token] += 1
    return term_doc_count

def calculer_RSV_probabiliste(K, B, freq_docs, requete, term_doc_count, avdl):
    """
    Calcule le score RSV probabiliste pour une requête donnée.

    Args:
        K (float): Constante BM25.
        B (float): Constante BM25 pour la normalisation de la longueur.
        freq_docs (dict): Dictionnaire {doc_num: {term: freq}} pour les documents.
        requete (list): Liste des termes de la requête.
        term_doc_count (dict): Dictionnaire {term: nombre de documents contenant le terme}.
        avdl (float): Longueur moyenne des documents.

    Returns:
        list: Liste triée des scores (doc_num, score).
    """
    scores = {}  # Dictionnaire pour stocker les scores de chaque document
    N = len(freq_docs)  # Nombre total de documents

    # Calculer le score pour chaque document
    for doc_num, terms_freq in freq_docs.items():
        dl = sum(terms_freq.values())  # Longueur du document (somme des fréquences des termes)
        score_doc = 0  # Score du document courant
        term_present = False  # Indicateur pour vérifier si un terme de la requête est présent

        for term in requete:
            if term in terms_freq:
                term_present = True  # Au moins un terme de la requête est présent
                freq = terms_freq[term]  # Fréquence du terme dans le document
                n_i = term_doc_count.get(term, 0)  # Nombre de documents contenant le terme
                if n_i > 0:  # Éviter la division par zéro
                    # Calculer les deux parties de l'équation BM25
                    part1 = freq / (K * ((1 - B) + B * (dl / avdl)) + freq)
                    part2 = math.log10((N - n_i + 0.5) / (n_i + 0.5))
                    score_doc += part1 * part2  # Ajouter au score du document

        # Ajouter le score uniquement si un terme de la requête est présent
        if term_present:
            scores[doc_num] = round(score_doc, 4)  # Arrondir le score pour plus de lisibilité

    # Trier les scores par ordre décroissant
    scores_trie = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return scores_trie

# Fonction déclenchée lorsque la sélection dans le Combobox change
def check_box_click(event):
    global selected_option
    selected_option = combo.get()  # Récupérer la valeur sélectionnée
   
def lire_ligne_i(fichier_path, i):
    try:
        with open(fichier_path, 'r', encoding='utf-8') as fichier:
            # Lire tout le contenu et diviser en phrases en fonction des lignes vides
            contenu = fichier.read().strip()  # Lire tout le contenu et supprimer les espaces inutiles
            phrases = [phrase.strip() for phrase in contenu.split("\n\n") if phrase.strip()]  # Diviser par double saut de ligne
            
            # Vérifier si l'indice i est valide
            if 0 <= i < len(phrases):
                return phrases[i]  # Retourner la i-ème phrase
            else:
                return f"Erreur : Le fichier ne contient que {len(phrases)} phrases."
    except FileNotFoundError:
        return f"Erreur : Le fichier '{fichier_path}' n'a pas été trouvé."
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {e}"


def get_from_judgment_doc_req(fichier_jugement, num_requet):
    # Charger les jugements pertinents depuis le fichier
    jugement = []
    with open(fichier_jugement, 'r') as file:
        for line in file:
            req, doc = line.split()
            print(int(req))
            print(num_requet)
            if int(req) == num_requet:
                print("yes")
                jugement.append(int(doc))  # Convertir le document en entier
    return jugement  # Retourner après avoir parcouru tout le fichier

#Fonction pour calculer les measures
def calculer_precision_recall_fscore(model_results, jugement):
    # Extraire les numéros de documents des résultats
    retrieved_docs = {doc_num for doc_num, score in model_results}

    # Calculer l'intersection des documents pertinents et des documents récupérés
    intersection = set(jugement) & retrieved_docs

    precision = len(intersection) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
    precision = round(precision, 4)

    rappel = len(intersection) / len(jugement) if len(jugement) > 0 else 0
    rappel = round(rappel, 4)

    # Calcul du F-score
    if precision + rappel > 0:
        f_score = 2 * (precision * rappel) / (precision + rappel)
    else:
        f_score = 0

    f_score = round(f_score, 4)

    # Calcul de P@5 (documents pertinents dans les 5 premiers ou tous les documents si < 5)
    p_at_5 = len(set(jugement) & {doc_num for doc_num, _ in model_results[:min(5, len(model_results))]}) / 5 if len(model_results) > 0 else 0
    # Calcul de P@10 (documents pertinents dans les 10 premiers ou tous les documents si < 10)
    p_at_10 = len(set(jugement) & {doc_num for doc_num, _ in model_results[:min(10, len(model_results))]}) / 10 if len(model_results) > 0 else 0

    # Arrondir P@5 et P@10 à 4 décimales
    p_at_5 = round(p_at_5, 4)
    p_at_10 = round(p_at_10, 4)

    # # Afficher les résultats
    # print("Precision:", precision)
    # print("Rappel:", rappel)
    # print("F-score:", f_score)
    # print("P@5:", p_at_5)
    # print("P@10:", p_at_10)

    return precision, rappel, f_score, p_at_5, p_at_10

# Fonction pour mettre à jour les résultats
def update_results_evaluation(precision, p5, p10, recall, fscore):
    precision_label_results.config(text=precision)
    p5_label_results.config(text=p5)
    p10_label_results.config(text=p10)
    recall_label_results.config(text=recall)
    fscore_label_results.config(text=fscore)

# Fonction pour extraire les numéros de documents
def extraire_numeros(model_results):
    return [doc_num for doc_num, _ in model_results]

# Fonction pour calculer la précision et le rappel réels
def calculer_precision_rappel_reel(model_results, jugement):
    retrieved_docs = extraire_numeros(model_results)  # Extraire les numéros de documents
    precision = []  # Liste des précisions
    rappel = []     # Liste des rappels
    pertinents_cumules = 0  # Compteur des documents pertinents trouvés

    for i, doc in enumerate(retrieved_docs):
        if doc in jugement:
            pertinents_cumules += 1
        precision_courante = pertinents_cumules / (i + 1)
        
        # Condition pour la longueur de jugement
        if len(jugement) > 10:
            rappel_courant = pertinents_cumules / 10  # Diviser par 10 si > 10 documents pertinents
        else:
            rappel_courant = pertinents_cumules / len(jugement)

        # Troncature sans arrondi
        precision.append(int(precision_courante * 100) / 100)
        rappel.append(int(rappel_courant * 100) / 100)

        # Si le rappel atteint 1, on arrête les calculs
        if rappel_courant == 1:
            break

    print(precision)
    print(rappel)
    return precision, rappel

# Étape 2 : Calcul des valeurs interpolées
def interpolation_precision(precision, rappel):
    rappel_interpole = [i / 10 for i in range(11)]  # Rappel de 0.0 à 1.0 par pas de 0.1
    precision_interpole = []
    
    # Pour chaque rappel interpolé, on prend la précision maximale des points où rappel >= rappel_interpolé
    for r in rappel_interpole:
        precision_max = max([p for p, r_real in zip(precision, rappel) if r_real >= r], default=0)
        precision_interpole.append(round(precision_max, 2))
    return rappel_interpole, precision_interpole

# Fonction pour tracer la courbe Recall/Precision
def tracer_courbe(rappel_interpole, precision_interpole):
    # Création de la figure
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)  # Ajout d'un sous-graphique

    ax.set_facecolor('white')  # Fond blanc

    # Tracer les lignes manuellement
    for i in range(1, len(rappel_interpole)):
        ax.plot(
            [rappel_interpole[i-1], rappel_interpole[i]],
            [precision_interpole[i-1], precision_interpole[i]],
            color='black', linewidth=1
        )

    # Tracer les points manuellement avec barres d'erreur
    for i in range(len(rappel_interpole)):
        ax.scatter(rappel_interpole[i], precision_interpole[i], color='red', s=50, zorder=5)
        ax.errorbar(rappel_interpole[i], precision_interpole[i], yerr=0.02, 
                    fmt='o', color='red', ecolor='red', capsize=3)

    # Personnalisation des axes avec petites tailles de police
    ax.set_xlabel("Recall", fontsize=8, color='yellowgreen')  # Taille réduite
    ax.set_ylabel("Precision", fontsize=8, color='yellowgreen')  # Taille réduite
    ax.set_title("Curve Recall/Precision", fontsize=9, fontweight='bold')  # Taille réduite
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.6, 1.05)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.arange(0.6, 1.05, 0.05))  # Pas de 0.05 pour l'axe y
    ax.grid(True, linestyle='-', alpha=0.5)

    # Réduction de la taille des ticks (graduations)
    ax.tick_params(axis='both', which='major', labelsize=7)  # Taille des labels des axes réduite

    # Intégration dans Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)  # Création du canvas
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Placement du canvas

def nettoyer_frame_plot():
    for widget in frame_plot.winfo_children():  # Parcours des widgets enfants
        widget.destroy()  # Supprime chaque widget
    frame_plot.config(text="Graphique")  # Mettre à jour le titre du cadre à vide

#######################Interface###########################

# Configuration de la fenêtre
Window = Tk()
Window.geometry("900x500")
Window.title("Search Interface")
Window.config(background="#d4d3d2")

# Variables des options de radio
x = IntVar()
y = IntVar()
z = IntVar()
t = IntVar()
k = IntVar()
dq = IntVar()

# Interface de saisie de la requête
label_query = Label(Window, text="Query", font=('Arial', 10, 'bold'), bg='#d4d3d2')
label_query.place(x=20, y=10)
search_entry = Entry(Window, font=('Arial', 10), width=55)
search_entry.place(x=70, y=10)
search_button = Button(Window, text='Search', font=('Arial', 10, 'bold'), command=on_search_click)
search_button.place(x=480, y=7)
index_checkbox = Checkbutton(Window, bg="#d4d3d2", variable=dq)
index_checkbox.place(x=570, y=8)
label_query = Label(Window, text="Queries Dataset", font=('Arial', 8), bg='#d4d3d2')
label_query.place(x=590, y=9)
# Add Spinbox
spinbox = Spinbox(Window, from_=1, to=100, bg="white", font=('Arial', 8), width=4)
spinbox.place(x=690, y=10)


# Cadre pour les options de traitement
processing_frame = LabelFrame(Window, text="Processing", font=('Arial', 9, 'bold'), bg="#d4d3d2", padx=3, pady=1)
processing_frame.place(x=20, y=50, width=250, height=120)
tokenization_frame = LabelFrame(processing_frame, text="Tokenization", font=('Arial', 8, 'bold'), bg="#d4d3d2")
tokenization_frame.grid(row=0, column=0, padx=5)
token_option1 = Radiobutton(tokenization_frame, text="Split", font=('Arial', 8), bg="#d4d3d2", variable=x, value=0)
token_option1.grid(row=0, column=0, sticky="w")
token_option2 = Radiobutton(tokenization_frame, text="RegExp", font=('Arial', 8), bg="#d4d3d2", variable=x, value=1)
token_option2.grid(row=1, column=0, sticky="w")
normalization_frame = LabelFrame(processing_frame, text="Normalization", font=('Arial', 8, 'bold'), bg="#d4d3d2")
normalization_frame.grid(row=0, column=1, padx=10)
norm_option1 = Radiobutton(normalization_frame, text="No stem", font=('Arial', 8), bg="#d4d3d2", variable=y, value=0)
norm_option1.grid(row=0, column=0, sticky="w")
norm_option2 = Radiobutton(normalization_frame, text="Porter Stemmer", font=('Arial', 8), bg="#d4d3d2", variable=y, value=1)
norm_option2.grid(row=1, column=0, sticky="w")
norm_option3 = Radiobutton(normalization_frame, text="Lancaster Stemmer", font=('Arial', 8), bg="#d4d3d2", variable=y, value=2)
norm_option3.grid(row=2, column=0, sticky="w")

# Cadre pour l'index
index_frame = LabelFrame(Window, text="Index", font=('Arial', 9, 'bold'), bg="#d4d3d2", padx=10, pady=10)
index_frame.place(x=280, y=70, width=260, height=80)
index_checkbox = Checkbutton(index_frame, bg="#d4d3d2", variable=t)
index_checkbox.grid(row=0, column=0, sticky="w")
index_option1 = Radiobutton(index_frame, text="DOCS per TERM", font=('Arial', 8), bg="#d4d3d2", variable=z, value=0)
index_option1.grid(row=0, column=1, sticky="w")
index_option2 = Radiobutton(index_frame, text="TERMS per DOC", font=('Arial', 8), bg="#d4d3d2", variable=z, value=1)
index_option2.grid(row=0, column=2, sticky="w")

# Cadre pour les options de matching
processing_frame = LabelFrame(Window, text="Matching", font=('Arial', 8, 'bold'), bg="#d4d3d2", padx=3, pady=1)
processing_frame.place(x=550, y=70, width=300, height=140)
token_option1 = Radiobutton(processing_frame, text="Vector Space Model", font=('Arial', 8), bg="#d4d3d2", variable=k, value=0)
token_option1.grid(row=0, column=0, sticky="w")
# Liste déroulante (Combobox) pour les méthodes de calcul
options = ["Scalar Product", "Cosine Measure", "Jaccard Measure"]  # Vous pouvez ajouter d'autres options
combo = ttk.Combobox(processing_frame, values=options, state="readonly", font=('Arial', 8), width=16)
combo.set("Select Method")  # Définir la valeur par défaut
combo.grid(row=1, column=0, padx=5, pady=(10, 0))

# Associer l'événement <<ComboboxSelected>> à la fonction `check_box_click`
combo.bind("<<ComboboxSelected>>", check_box_click)

# Option pour le modèle BM25
bm25_radio = Radiobutton(processing_frame, text="Probabilistic Model (BM25)", font=('Arial', 8), bg="#d4d3d2", variable=k, value=1)
bm25_radio.grid(row=0, column=1, padx=5)

# Champ K
entry_k = Entry(processing_frame, font=('Arial', 8), width=8)
entry_k.grid(row=1, column=1, pady=(10, 0))  # Alignement à côté du label K

# Champ B
entry_b = Entry(processing_frame, font=('Arial', 8), width=8)
entry_b.grid(row=2, column=1,  pady=(10, 0))

# Option pour le modèle boolean
boolean_Model_radio = Radiobutton(processing_frame, text="Boolean Model", font=('Arial', 8), bg="#d4d3d2", variable=k, value=2)
boolean_Model_radio.grid(row=3, column=0, pady=15, sticky="w")

# Option pour le modèle datamining
dataMining_Model_radio = Radiobutton(processing_frame, text="Data Mining Model", font=('Arial', 8), bg="#d4d3d2", variable=k, value=3)
dataMining_Model_radio.grid(row=3, column=1, padx=5, pady=15, sticky="w")

# Affichage des résultats dans une table
results_label = Label(Window, text="Results", font=('Arial', 10, 'bold'), bg='#d4d3d2')
results_label.place(x=20, y=180)
# Créer un Treeview sans colonnes ni données
tree = ttk.Treeview(Window, show="tree", height=12)  # `show="tree"` masque les en-têtes de colonnes
tree.place(x=20, y=210, width=516)

# Cadre pour afficher le graphique avec fond
frame_plot = tk.LabelFrame(Window, text="Graphique", padx=5, pady=5, bg='#d4d3d2')
frame_plot.place(x=550, y=219, width=300, height=260)

# Création des labels pour chaque métrique
precision_label = Label(Window, text="Precision:", font=('Arial', 8, 'bold'), bg='#d4d3d2')
precision_label.place(x=20, y=480)
precision_label_results = Label(Window, text="", font=('Arial', 8), bg='#d4d3d2')
precision_label_results.place(x=80, y=480)

p5_label = Label(Window, text="P@5:", font=('Arial', 8, 'bold'), bg='#d4d3d2')
p5_label.place(x=140, y=480)
p5_label_results = Label(Window, text="", font=('Arial', 8), bg='#d4d3d2')
p5_label_results.place(x=170, y=480)

p10_label = Label(Window, text="P@10:", font=('Arial', 8, 'bold'), bg='#d4d3d2')
p10_label.place(x=230, y=480)
p10_label_results = Label(Window, text="", font=('Arial', 8), bg='#d4d3d2')
p10_label_results.place(x=270, y=480)

recall_label = Label(Window, text="Recall:", font=('Arial', 8, 'bold'), bg='#d4d3d2')
recall_label.place(x=330, y=480)
recall_label_results = Label(Window, text="", font=('Arial', 8), bg='#d4d3d2')
recall_label_results.place(x=370, y=480)

fscore_label = Label(Window, text="F-score:", font=('Arial', 8, 'bold'), bg='#d4d3d2')
fscore_label.place(x=430, y=480)
fscore_label_results = Label(Window, text="", font=('Arial', 8), bg='#d4d3d2')
fscore_label_results.place(x=480, y=480)


Window.mainloop()
