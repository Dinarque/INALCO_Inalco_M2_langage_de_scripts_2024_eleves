import time
import threading
import asyncio

# Liste des noms des étudiants
etudiants = ["Alice", "Bob", "Chloé", "David", "Emma",
             "Félix", "Gabriel", "Hannah", "Isabelle", "Julien"]

# 1) la classe séquentielle

def entree_classe():
    print("Les étudiants entrent en classe, un par un...\n")
    for etudiant in etudiants:
        print(f"{etudiant} : \"Présent !\"")
        time.sleep(3)  # Pause de 3 secondes pour chaque étudiant
    print("\nTous les étudiants sont assis.")

def debut_cours():
    print("Le professeur : \"Bien, tout le monde est là. Commençons le cours !\"")

"""
start = time.time()
entree_classe()
debut_cours()
end = time.time()
print(f"La classe a mis {round(end-start)} secondes à s'installer")
"""



# 2) la classe asynchrone

# Fonction asynchrone pour simuler l'entrée d'un étudiant
async def etudiant_entree(etudiant):
    print(f"{etudiant} : \"Présent !\"")
    await asyncio.sleep(3)  # Pause asynchrone de 3 secondes
    print(f"{etudiant} s'assoit.")

# Fonction asynchrone principale pour gérer la classe
async def entree_classe_async():
    print("Les étudiants entrent en classe, un par un...\n")
    tasks = [etudiant_entree(etudiant) for etudiant in etudiants]
    await asyncio.gather(*tasks)  # Lance toutes les tâches en parallèle
    print("\nTous les étudiants sont assis.")

"""
start = time.time()
asyncio.run(entree_classe_async())  # Lancer la fonction asynchrone principale
debut_cours()
end = time.time()

print(f"La classe a mis {round(end-start)} secondes à s'installer.")
"""

# 3) le professeur pressé

# Fonction asynchrone pour simuler l'entrée d'un étudiant
async def etudiant_entree(etudiant):
    print(f"{etudiant} : \"Présent !\"")
    await asyncio.sleep(3)  # Pause asynchrone de 3 secondes
    print(f"{etudiant} s'assoit.")

# il faut que cette fonction soit async pour pouvoir être exécutée en //
async def debut_cours_presse():
    print("Le professeur : \"Tant pis, je commence le cours !\"")

# Fonction asynchrone principale pour gérer la classe
async def entree_classe_presse():
    print("Les étudiants entrent en classe, un par un...\n")
    tasks = [asyncio.create_task(etudiant_entree(etudiant)) for etudiant in etudiants]
    tasks.append(asyncio.create_task(debut_cours_presse()))  # Ajouter la tâche du professeur
    print("\nCertains étudiants ne sont peut-être pas encore assis...")
    await asyncio.gather(*tasks)
# Fonction pour débuter le cours


# Lancer le programme
start = time.time()
asyncio.run(entree_classe_presse())  # Lancer la fonction asynchrone principale
end = time.time()
print(f"La classe a mis {round(end-start)} secondes à s'installer (mais le professeur n'a pas attendu !).")
time.sleep(34)


# Lock pour accéder au tableau
tableau_lock = threading.Lock()

def ecrire_nom(nom):
    print(f"{nom} veut écrire sur le tableau.")
    with tableau_lock:
        print(f"{nom} commence à écrire...")
        time.sleep(1)  # Simule le temps d'écrire
        print(f"{nom} a fini d'écrire son nom.")
    print(f"{nom} retourne s'asseoir.\n")

# Création et lancement des threads
threads = [threading.Thread(target=ecrire_nom, args=(nom,)) for nom in etudiants]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

print("Tous les élèves ont écrit leur nom sur le tableau.")
