import unicodedata

import streamlit as st
import pandas as pd
import re
from annotated_text import annotated_text
from collections import defaultdict


# Fonction pour extraire l'année de début et de fin de la chaîne de date
def extract_years(date_str):
    match = re.search(r"AD (\d+)(?:\s*-\s*(\d+))?", date_str)
    if match:
        start_year = int(match.group(1))  # Première année (début de l'intervalle)
        end_year = (
            int(match.group(2)) if match.group(2) else start_year
        )  # Si pas d'intervalle, end_year = start_year
        return start_year, end_year
    return None, None


# création d'un index inversé des noms et des lieux
def annotate_text_with_corrections(text, annotations):
    annotated_content = []
    start_index = 0

    for original, correction, color in annotations:

        position = text.find(original, start_index)
        if position != -1:
            if start_index < position:
                annotated_content.append(text[start_index:position])
            annotated_content.append((original, correction, color))
            start_index = position + len(original)

    if start_index < len(text):
        annotated_content.append(text[start_index:])

    return annotated_content


def create_reverted_index(csv_file):
    # Création d'un index inversé pour retrouver la liste des ids depuis le texte
    df = pd.read_csv(csv_file)
    reverted_index = defaultdict(list)
    for _, row in df.iterrows():
        entity_id = row["ID"]
        people = (
            eval(row.get("People List", "")) if pd.notna(row.get("People List")) else []
        )
        places = (
            eval(row.get("Places List", "")) if pd.notna(row.get("Places List")) else []
        )

        for entity in people + places:
            entity = entity.strip()
            if entity:
                reverted_index[entity].append(entity_id)

    return dict(reverted_index)


# Chargement du DataFrame à partir du fichier CSV
df = pd.read_csv("clean_papyrus-corpus.csv")
reverted_index = create_reverted_index("clean_papyrus-corpus.csv")

# Appliquer l'extraction d'années pour créer deux colonnes dans le DataFrame
df["Year_Beginning"], df["Year_End"] = zip(*df["Date"].apply(extract_years))

# Titre de l'application
st.title("La Chasse aux Papyrus")

# Image en haut de la page
st.image(
    "https://st.depositphotos.com/1534011/2186/i/450/depositphotos_21867323-stock-photo-egyptian-papyrus.jpg",
    caption="Un voyage à travers les anciens manuscrits",
)

# Description de l'application
st.write(
    "Bienvenue dans l'exploration des papyrus anciens. Vous pouvez naviguer à travers notre collection et filtrer par provenance ou date de découverte pour en savoir plus sur chaque papyrus."
)

# Fonctionnalité Bonus
if st.button("Déclenchez la malédiction de la momie"):
    st.balloons()
    # NB : cette fonctionnalité sert à égayer votre quotidien et à vous rappeler qu'en tant que professeur j'ai le droit
    # de faire de l'humour

### SIDEBAR GÉRANT LA SÉLECTION ###

# Barre latérale - Menu de sélection
st.sidebar.header("Filtres")

# Filtre par provenance (selectbox)
lieux = df["Provenance"].unique().tolist()
lieu_selection = st.sidebar.selectbox(
    "Filtrer par provenance", options=["Toutes"] + lieux
)

# Filtre par date (slider) basé sur les colonnes 'Year_Beginning' et 'Year_End'
date_min = int(df["Year_Beginning"].min())
date_max = int(df["Year_End"].max())
date_selection = st.sidebar.slider(
    "Filtrer par date",
    min_value=date_min,
    max_value=date_max,
    value=(date_min, date_max),
)

# Filtrage des données selon les critères choisis
if lieu_selection != "Toutes":
    df_filtered = df[
        (df["Provenance"] == lieu_selection)
        & (
            (df["Year_Beginning"] <= date_selection[1])
            & (df["Year_End"] >= date_selection[0])
        )
    ]
else:
    df_filtered = df[
        (df["Year_Beginning"] <= date_selection[1])
        & (df["Year_End"] >= date_selection[0])
    ]

# Menu déroulant pour choisir un papyrus après filtrage
papyrus_list = df_filtered["ID"].tolist()

# Condition pour afficher les papyri seulement s'il y en a après filtrage
if papyrus_list:
    selection = st.sidebar.selectbox("Choisissez un papyrus (ID)", options=papyrus_list)

    ### FIN sidebar

    # Affichage des informations sur le papyrus sélectionné
    if selection:
        papyrus_info = df[df["ID"] == selection].iloc[0]

        # Numéro du papyrus
        st.header(f"Papyrus N° {papyrus_info['ID']}")

        # Utilisation des colonnes pour présenter les informations sous forme clé/valeur
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Date")
            st.write(papyrus_info["Date"])

            st.subheader("Provenance")
            st.write(papyrus_info["Provenance"])

            st.subheader("Type de texte")
            st.write(papyrus_info["Content"])

        with col2:
            st.subheader("Personnes")
            if pd.notna(papyrus_info["People List"]):
                for i, person in enumerate(set(eval(papyrus_info["People List"]))):
                    person = person.strip()
                    if person:
                        if st.button(person, key=person + str(i)):
                            st.write(
                                f"Papyrus IDs **{person}**: {reverted_index.get(person, [])}"
                            )

            st.subheader("Lieux")
            if pd.notna(papyrus_info["Places List"]):
                for i, place in enumerate(set(eval(papyrus_info["Places List"]))):
                    place = place.strip()
                    if place:
                        if st.button(place, key=place + str(i)):
                            st.write(
                                f"Papyrus IDs **{place}**: {reverted_index.get(place, [])}"
                            )

            st.subheader("Corrections apportées au texte")
            st.write(papyrus_info["Text Irregularities"])

        st.subheader("Contenu")
        # Affichage du contenu avec annotations pour les irrégularités textuelles

        text_irregularities = papyrus_info["Text Irregularities"]
        text = papyrus_info["Unedited Text"]
        text = unicodedata.normalize("NFC", text)

        if pd.notna(text_irregularities):  # Check for irregularities
            annotations = []
            # On crée une liste d'annotations

            for irregularity in eval(text_irregularities):

                original, correction = irregularity.split("read")
                original = original.strip().replace(":", "")
                correction = correction.strip().replace(":", "")
                # on les ajoute
                annotations.append(
                    (
                        unicodedata.normalize("NFC", original),
                        unicodedata.normalize("NFC", correction),
                        "#faa",
                    )
                )

                st.write(irregularity)
                st.write(unicodedata.normalize("NFC", original) in text)
            # Use annotated_text to display the text with highlights

            annotated_text(annotate_text_with_corrections(text, annotations))

        st.subheader("Texte non nettoyé")
        st.write(papyrus_info["Full Text"])

    else:
        st.write("Aucun papyrus ne correspond aux filtres sélectionnés.")
