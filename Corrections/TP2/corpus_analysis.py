import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from transformers import pipeline
from tqdm import tqdm
import difflib
import ast
import unicodedata
import networkx as nx
from unidecode import unidecode
from sklearn.metrics import f1_score


""" 4) Chargement et nettoyage du dataset """

# Observer le dataset. Que dire des 4 premières lignes ?  => Il y a une erreur de formattage et pas de texte capturé
# Que faut-il faire ?
# Il faut donc les supprimer ! Le paramètre skiprows permet de le faire . Ici on veut garder la ligne 0 (index) mais
# enlever les lignes 2 à 5. On fait donc appel à une fonction lambda :)
df = pd.read_csv("papyrus_corpus.csv", skiprows=lambda x: x in range(1, 5))
print(df.columns)

# Combien de textes n'ont pas été capturés pendant le scraping ? Comment le voit-on ? Enlevez-les
# Cela revient  à Compter le nombre de valeurs NaN dans la colonne 'Full Text' ?  Enlevez-les du corpus
nan_count = df["Full Text"].isna().sum()
print(f"Il y a {nan_count} papyri dont le texte est indisponible.")

# combien la collection compte-t elle de papyrus après nettoyage ?
df = df[df["Full Text"].notna()]
papyri_count = df.shape[0]
print(f"Il y a {papyri_count} papyri avec un texte.")
print(df.head())

# Trier selon l'ID de Papyrus si pas fait avant
df = df.sort_values(by="ID")

""" 5)  Etude de corpus : genre, lieu et date  """

""" Genre"""
# Quelles sont les différents genre de texte que l'on a collecté ? La nomenclature de la colonne "Content (beta!) est
# un peu trop précise. Basez-vous sur le premier mot et créez un graphique
df["Content"] = (
    df["Content (beta!)"]
    .str.split()
    .str[0]
    .str.replace(":", "")
    .str.replace("See", "")
    .str.lower()
)
df["Content"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(6, 6))
print(df["Content"].unique())
plt.ylabel("")
plt.show()

# Combien de papyri ont-il été réutilisés ? (c'est à dire ont un recto et un verso...)
reused_count = df["Reuse type"].dropna().sum()
print(f"Il y a {reused_count} papyri qui ont été réutilisés.")


""" lieu"""
# D'où viennent les papyri ?  De même ne retenez que le nom de la ville. Faites un diagramme en barre cette fois.
# Qu'en concluez-vous ?
df["Provenance"] = df["Provenance"].str.split().str[0]
df["Provenance"].value_counts().plot(kind="bar", figsize=(6, 6))
print(df["Content"].unique())
plt.ylabel("")
plt.show()
# => L'essentiel  de la collection vient du même endroit


"""Date"""
# Identifier la date d'écriture. Observez la catégorie des dates et nettoyez le texte pour regrouper ensemble les catégories pertinente
# NB quand il y a plusieurs dates mentionnées on ne prendra en compte que la première


def clean_date(text):
    # identification des intervalles dans la donnée de dates
    match_range = re.search(r"AD (\d{3}) - (\d{3})", text)
    if match_range:
        return f"AD {match_range.group(1)} - {match_range.group(2)}"

    # si échec on recherche une date simple
    match_single = re.search(r"AD (\d{3})", text)
    if match_single:
        return f"AD {match_single.group(1)}"
    return None


df["Date"] = df["Date"].apply(
    clean_date
)  # pour appliquer la fonction précédente à tout le dataset

# Un diagramme en barre n'est pas très adapté pour des intervalles discontinus.
# De ce fait on va transfomer les intervalles en une liste des dates entre les deux bornes


def extract_years(date_str):
    if not isinstance(date_str, str):
        return []
    matches = re.findall(r"AD (\d{3}) - (\d{3})", date_str)
    years = []
    for start, end in matches:
        years.extend(
            range(int(start), int(end) + 1)
        )  # NB on pourrait faire avec un range
    return years


# On crée une colonne qui associe chaque papyrus à la liste des dates possibles
df["Years"] = df["Date"].apply(lambda x: extract_years(x))
# On applatit pour avoir un histogramme
all_years = [year for sublist in df["Years"] for year in sublist]

# On crée un graphique représentant la densité du nombre d'années qui apparaissent.
# NB : cette idée n'a pas de vraie valeur scientifique car une date certaine a autant de valeur que la potentialité
# d'être dans un créneau large (siècle...) . on pourrait pondérer en divisant par le nombre d'années dans l'intervalle
# ce type de représentation n'aurait du sens que si tout le corpus était précisément daté, ici le but était juste de
# jouer un peu avec les données.
years_df = pd.DataFrame(all_years, columns=["Year"])
plt.figure(figsize=(12, 6))
sns.kdeplot(years_df["Year"], fill=True)
plt.title("Density of papyrus per years")
plt.xlabel("Year")
# On réajuste l'axe des abcisses pour ne couvrir que la période pour laquelle on a des textes, et hop !
plt.xlim(years_df["Year"].min(), years_df["Year"].max())
plt.grid(True)
plt.show()


""" 6) Nettoyage du texte des papyri """


# écrivez une première fonction de nettoyage du texte qui retire les chiffres arabes, les lignes perdues | gap | ainsi
# que les caractères spéciaux "†" et "⳨". Appliquez là au texte.
def remove_arabic_numbers(text):
    pattern = r"[0-9]"
    cleaned_text = re.sub(pattern, "", text)
    return " ".join(cleaned_text.split())


def clean_papyri_text(text):
    text = remove_arabic_numbers(text)
    text = re.sub(r"\|gap=\d+_lines\|", ".\n", text)
    text = text.replace("†", ".\n").replace("⳨", "")
    return text.strip()


# Ecrivez une fonction qui prend en entrée un texte de papyrus nettoyé et renvoie la proportion de lettres incertaines
# dans le texte. Stockez pour chaque texte cette valeur dans la colonne "Uncertain Portion"]


def count_dot_below(text):
    dot_below = "\u0323"  # Unicode du point bizarre en bas d'une lettre
    count = 0
    for char in text:
        if char == dot_below:
            count += 1
    return count


def count_letters_in_brackets(text):
    # On somme le nombre de caractères entre les parenthèse et les crochets
    square_bracket_matches = re.findall(r"\[(.*?)\]", text)
    square_bracket_count = sum(len(match) for match in square_bracket_matches)

    parentheses_matches = re.findall(r"\((.*?)\)", text)
    parentheses_count = sum(len(match) for match in parentheses_matches)

    return square_bracket_count + parentheses_count


def count_uncertain_proportion(text):
    uncertain_letter_counts = count_letters_in_brackets(text) + count_dot_below(text)
    total_letter = len(unedit_text(text).replace(" ", ""))
    return uncertain_letter_counts / total_letter


# Ecrivez une fonction qui enlève les parenthèses et les crochets mais converve leur contenu
def unedit_text(text):
    text = text.replace("[", "").replace("]", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("/", "").replace("\\", "")
    text = text.replace(" -", "").replace("- ", "")
    text = text.replace("\u0323", "")
    return text


df["Clean Text"] = df["Full Text"].apply(clean_papyri_text)
df["Uncertain Portion"] = df["Clean Text"].apply(count_uncertain_proportion)
print(
    f" Il y a {df[df['Uncertain Portion'] > 1/3]['ID'].nunique()} papyrus dont plus d un tiers des lettres sont incertaines"
)
df["Unedited Text"] = df["Clean Text"].apply(unedit_text)


""" 7) Utiliser un système de NER sur les textes du corpus """

# Observez le contenu des cellules de la case "people-list". Que remarquez-vous ? Réglez le problème

# 1) Il y a des chiffres arabes parfois au début des noms dont on a pas besoin

# 2) parfois des contenus indésirables ont été collectés


# On doit donc créer une fonction de nettoyage adaptée
def clean_people_list(str_people_list):
    str_people_list = str_people_list.replace(
        r"['\r\n \t\t\t\t\tWe currently do not have any people attestations for this text.']",
        "[]",
    )
    if (
        "We currently do not have any people attestations for this text"
        in str_people_list
    ):
        str_people_list = "[]"
    str_people_list = str_people_list.replace(", 'Subscribe to export the table'", "")
    str_people_list = (
        remove_arabic_numbers(str_people_list).replace("' ", "'").replace("- ", "")
    )
    return str_people_list


df["People List"] = df["People List"].apply(clean_people_list)


# La liste de lieux ressemble à un dictionnaire donc on doit retrouver les clés
def clean_places_list(str_places_list):
    return str([k for k in eval(str_places_list)])


df["Places List"] = df["Places List"].apply(clean_places_list)

# Création d'un pipeline de NER avec Hugging Face
ner = pipeline("ner", model="UGARIT/grc-ner-bert", aggregation_strategy="first")

#  Téléchargez le modèle de NER suivant sur Hugging Face https://huggingface.co/UGARIT/grc-ner-bert
tqdm.pandas()  # Enable tqdm for pandas


def apply_ugarit_ner(text):
    text = text.replace("-", "").replace(".", "")
    locations = []
    persons = []
    others = []
    entities = ner(text)
    for entity in entities:
        if entity["entity_group"] == "PER":
            persons.append(entity["word"])
        if entity["entity_group"] == "LOC":
            locations.append(entity["word"])
        else:
            others.append(entity["word"])
    return persons, locations, others


df["People Ugarit"], df["Places Ugarit"], df["Others Ugarit"] = zip(
    *df["Unedited Text"].progress_apply(apply_ugarit_ner)
)
df.to_csv("clean_papyrus-corpus_NER.csv")

"""Evaluation du système de NER"""


def remove_diacritics(text):
    normalized_text = unicodedata.normalize("NFD", text)
    filtered_text = "".join(
        [char for char in normalized_text if not unicodedata.combining(char)]
    )
    return unicodedata.normalize("NFC", filtered_text)


def normalize_list(liste):
    if type(liste) == str:
        liste = eval(liste)
    return set([remove_diacritics(unedit_text(el.lower().strip())) for el in liste])


def compute_f1_scores(df):
    # On normalise pour éviter que les signes diacritiques ne créent des soucis

    # On calcule les résultats ligne à ligne puis on calculera la moyenne
    def evaluate(row, entity):
        true = normalize_list(row[f"{entity} List"])
        detected_loose = (
            normalize_list(row[f"People Ugarit"])
            | normalize_list(row[f"Places Ugarit"])
            | normalize_list(row["Others Ugarit"])
        )
        detected_severe = normalize_list(row[f"{entity} Ugarit"])

        print(entity)
        print(detected_loose)
        print(true)

        loose_tp = len(true & detected_loose)
        loose_fp = len(detected_loose - true)
        loose_fn = len(true - detected_loose)

        severe_tp = len(true & detected_severe)
        severe_fp = len(detected_severe - true)
        severe_fn = len(true - detected_severe)

        return loose_tp, loose_fp, loose_fn, severe_tp, severe_fp, severe_fn

    metrics = {
        "loose_tp": 0,
        "loose_fp": 0,
        "loose_fn": 0,
        "severe_tp": 0,
        "severe_fp": 0,
        "severe_fn": 0,
    }
    for _, row in df.iterrows():
        for entity in ["People", "Places"]:
            lt, lf, lf_n, st, sf, sf_n = evaluate(row, entity)
            metrics["loose_tp"] += lt
            metrics["loose_fp"] += lf
            metrics["loose_fn"] += lf_n
            metrics["severe_tp"] += st
            metrics["severe_fp"] += sf
            metrics["severe_fn"] += sf_n

    loose_f1 = f1_score(
        [1] * metrics["loose_tp"]
        + [0] * metrics["loose_fp"]
        + [1] * metrics["loose_fn"],
        [1] * metrics["loose_tp"]
        + [1] * metrics["loose_fp"]
        + [0] * metrics["loose_fn"],
        average="binary",
    )
    severe_f1 = f1_score(
        [1] * metrics["severe_tp"]
        + [0] * metrics["severe_fp"]
        + [1] * metrics["severe_fn"],
        [1] * metrics["severe_tp"]
        + [1] * metrics["severe_fp"]
        + [0] * metrics["severe_fn"],
        average="binary",
    )

    return loose_f1, severe_f1


loose_f1, severe_f1 = compute_f1_scores(df)
print("Résultats de l'évaluation du système de NER Ugarit sur notre corpus")
print(f"Loose F1: {loose_f1}, Severe F1: {severe_f1}")


""" 8) Etude des variations graphiques """

# Créez un nouveau DataFrame nommé "sound_change_df"  qui aura pour colonne "old" (forme correcte en grec classique) "new" (forme trouvée dans le papyrus) et remplissez le en lisant les erreurs relevées dans la colonne Irrtex
# Normalisez le texte en amont et enlevez les signes diacritiques


# En utilisant la librairie difflib qui permet d'étudier les différences (insertions, délétions, modifications...)
# entre deux chaînes de caractère,
def extract_irregularities(line):

    parts = line.split(":")
    if len(parts) != 2:
        return None

    old_form = parts[0].strip()
    new_form = parts[1].strip().replace("read", "").replace(":", "").strip()
    # On analyse la sortie de difflib pour repérer les caractères changés
    d = list(difflib.ndiff(old_form, new_form))
    old_segment = (
        "".join([char[2] for char in d if char.startswith("-")])
        .strip()
        .replace(">", "")
        .replace("<", "")
    )
    new_segment = (
        "".join([char[2] for char in d if char.startswith("+")])
        .strip()
        .replace(">", "")
        .replace("<", "")
    )
    return {"old": new_segment, "new": old_segment}


# Il faut convertir la colonne Text Irregularities en liste pour pouvoir extraire ses éléments
df["Text Irregularities"] = df["Text Irregularities"].apply(ast.literal_eval)

flattened_irregularities = [
    extract_irregularities(el)
    for ir_list in df["Text Irregularities"]
    for el in ir_list
]
flattened_irregularities = [ir for ir in flattened_irregularities if ir is not None]
sound_change_df = pd.DataFrame(flattened_irregularities)
sound_change_df.to_csv("sound_change.csv", index=False)
print("sound_change_csv saved")
sound_change_df = sound_change_df[sound_change_df["old"] != "{}"]
sound_change_df = sound_change_df[sound_change_df["old"] != "<>"]

# Quels sont les  10 changements les plus fréquents ?
sound_change_df["change"] = sound_change_df["old"] + " -> " + sound_change_df["new"]
top_10_changes = sound_change_df["change"].value_counts().head(10)
print("10 changements de sons les plus fréquents")
print(top_10_changes)
old_sound_counts = sound_change_df["old"].value_counts()
most_frequent = sound_change_df[
    sound_change_df["old"].isin(old_sound_counts[old_sound_counts > 29].index)
]
# Quels graphèmes du grecs classiques ont été modifiés plus de 30 fois dans le dataset ?
most_frequent_sounds = most_frequent["old"].unique()
most_frequent_sounds_count = most_frequent["old"].nunique()
print(f"Il y a {most_frequent_sounds_count} sons qui ont été modifiés plus de 30 fois.")
print(most_frequent_sounds)


# Créez un graphique unique qui représente pour chacun de ces 8 graphèmes la nouvelle forme
# qu'il va prendre sous la forme d'un pie chart (utilisez les subplots de matplotlib)
fig, axes = plt.subplots(4, 2, figsize=(15, 15))  # 4 rows, 2 columns
axes = axes.flatten()

for i, old_sound in enumerate(most_frequent_sounds):
    new_transformations = most_frequent[most_frequent["old"] == old_sound][
        "new"
    ].value_counts()
    axes[i].pie(
        new_transformations,
        labels=new_transformations.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[i].set_title(f"Old Sound: {old_sound}")

plt.tight_layout()
plt.show()

# BONUS : représentez le graphe de conversion de sons (chaque noeud représente un son, le poids d'une arrete le nombre
# de changements constatés dans le corpus)


def create_translation_graph(df):

    G = nx.DiGraph()
    for index, row in df.iterrows():
        old_sound = row["old"]
        new_sound = row["new"]
        weight = df[(df["old"] == old_sound) & (df["new"] == new_sound)].shape[0]  #

        if not G.has_edge(old_sound, new_sound):
            G.add_edge(old_sound, new_sound, weight=weight)
    return G


def plot_translation_graph(G):
    pos = nx.circular_layout(G)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    print(edge_weights)
    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
    )  # TODO add weights as labels
    plt.title("Sound Translation Graph")
    plt.show()


G = create_translation_graph(most_frequent)
plot_translation_graph(G)


# Ne conservez que les colonnes utiles à notre analyse.
# Quelles sont les colonnes du df ? Eliminez toutes celles qui vous paraissent inutiles.
print(df.columns)
columns_to_delete = [
    "Authors / works",
    "Book form",
    "Content (beta!)",
    "Culture & genre",
    "Language/script",
    "Material",
    "Note",
    "Recto/Verso",
    "Reuse note",
    "Reuse type",
    "Ro",
    "Years",
    "Clean Text",
]
# Drop the specified columns
clean_df = df.drop(columns=columns_to_delete)
clean_df.to_csv("clean_papyrus-corpus.csv")
df = clean_df
print(clean_df.columns)


""" Vous y avez échappé je n'ai pas eu le temps de finaliser :


7) étude de la subordination : l'emploi de ws et wste 

from flair.models import SequenceTagger
from flair.data import Sentence
from transformers import AutoTokenizer

# Correct the model identifier from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")

destination = './Ancient-Greek-BERT/SuperPeitho-FLAIR-v2/final-model.pt'

model_url = "https://github.com/pranaydeeps/Ancient-Greek-BERT/raw/main/SuperPeitho-FLAIR-v2/final-model.pt"


#urllib.request.urlretrieve(model_url, destination)


def replace_string_in_pt_file(file_path, old_string, new_string):
    # Read the entire file as binary
    with open(file_path, 'rb') as file:
        file_content = file.read()

    # Replace the old string with the new string in byte format
    file_content_modified = file_content.replace(old_string.encode('utf-8'), new_string.encode('utf-8'))

    # Write the modified content back to the file
    with open(file_path, 'wb') as file:
        file.write(file_content_modified)

# Define the file path and strings to replace
file_path = destination  # Update with your correct path
old_string = '../LM/SuperPeitho-v1'  # The string to be replaced
new_string = 'pranaydeeps/Ancient-Greek-BERT'  # The new string to replace with

# Call the function to perform the replacement
replace_string_in_pt_file(file_path, old_string, new_string)


tagger = SequenceTagger.load(destination)

def POS_TAG_text(text):

    sentence = Sentence(text)
    tagger.predict(sentence)
    print(sentence)

df["Tagged Text"] = df["Unedited Text"].progress_apply(POS_TAG_text)
df.to_csv("clean_papyrus-corpus_POS.csv") """
