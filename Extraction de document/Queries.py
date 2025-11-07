# Ouvrir le fichier d'entrée en lecture et le fichier de sortie en écriture
with open("med/MED.QRY", "r") as infile, open("Queries", "w") as outfile:
    for line in infile:
        # Supprimer les espaces inutiles au début/fin de la ligne
        line = line.strip()
        # Ignorer les lignes commençant par .I ou .W
        if not (line.startswith(".I") or line.startswith(".W")):
            # Écrire les phrases dans le fichier Queries
            outfile.write(line + "\n")  # Ajouter une ligne vide après chaque phrase
        elif(line.startswith(".I")):
            outfile.write("\n")
