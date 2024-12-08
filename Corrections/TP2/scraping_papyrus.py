import requests
from bs4 import BeautifulSoup
import csv
import time
import random
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

""" 1) Obtenir la liste des url à scrapper"""

# Il faut pour cela extraire la liste du fichier


def get_url_list(path):
    """
    :param path: chemin menant au fichier contenant les infos sur le corpus à scraper
    :return: liste des urls à scaper.
    """
    df = pd.read_csv(path)
    # On peut extraire la liste des ids de papyrus du corpus dans la colonne ID du csv fourni en transformant un peu le
    # texte. Une étude de la structure du site permet de comprendre comment dériver l'url depuis l'ID du papyrus.
    ids = df["ID"]
    return [f"https://www.trismegistos.org/text/{i.split('TM ')[1]}" for i in ids]


""" 2) écrire une fonction qui permette d'extraire d'un lien web toutes les informations qui nous intéressent"""

# NB : si on étudie la structure de la page, les informations qui nous intéressent se trouvent dans la section TOP
# (dictionnaire des métadonnées) et dans la section all_tabs où chaque tab stocke une information particulière


def extract_greek_text(tag):
    """
    Recompose le texte grec éparpillé dans plusieurs balises
    """
    text_parts = []
    for element in tag.children:
        if element.name == "span" and "tooltiptext" in element.get("class", []):
            continue
        if element.name == "a" and "info-tooltip" in element.get("class", []):
            tooltip_span = element.find("span", class_="tooltiptext")

            # Si on trouve tooltip dans le texte alors il faut nettoyer plus
            if tooltip_span:
                link_text = element.text.replace(tooltip_span.get_text(), "").strip()
            else:
                link_text = element.text.strip()
            if link_text:
                text_parts.append(link_text)

        elif isinstance(element, str):
            text_parts.append(element.strip())

    return " ".join(text_parts)


# Fonction pour scraper les métadonnées d'une page Web correspondant à un papyrus
def scrape_metadata(url):

    # récupérer le contenu de la page
    headers = {"User-agent": "student for a project on the syntax on some greek papyri"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:

        # création du parser
        soup = BeautifulSoup(response.content, "html.parser")

        # Trouver les divisions stockant les informations d'intérêt dans le bloc central
        metadata_section = soup.find("div", id="text-details", class_="text-info")
        metadata_dict = {}

        if metadata_section:

            divisions = metadata_section.find_all(
                "div", class_="division"
            ) + metadata_section.find_all("p", class_="division")

            for division in divisions:
                # Pour chaque sous section de la section text_details, on va capturer et stocker l'information
                division_text = division.get_text(strip=True)
                print(division_text)

                # Distinction entre la clé et la valeur de l'information
                if ":" in division_text:
                    label, value = division_text.split(":", 1)
                    label = label.strip()
                    value = value.strip()
                    metadata_dict[label] = value

                # Capture du texte du papyrus
                full_text_section = soup.find("div", id="words-full-text")
                if full_text_section:
                    combined_tooltips_text = extract_greek_text(
                        full_text_section
                    ).strip()
                    metadata_dict["Full Text"] = combined_tooltips_text

                # Capture des noms de personne
                people_list = soup.find("ul", id="people-list")
                if people_list:
                    names = []
                    for el in people_list.children:
                        if len(el.text) > 2:
                            names.append(el.text)
                    metadata_dict["People List"] = names

                # Capture des noms de lieux + lien de redirection vers geo JSON pour question bonus
                location_list = soup.find("ul", id="places-list")
                if location_list:
                    locations = {}

                    for element in location_list.find_all("li"):
                        place_name = element.get_text(strip=True)

                        onclick_value = element.get("onclick")
                        if onclick_value:
                            geo_number = onclick_value.split("getgeo(")[-1].rstrip(")")
                            locations[place_name] = geo_number

                    metadata_dict["Places List"] = locations

                # Capture des détails sur les lieux (pour question bonus)
                location_detail = soup.find("div", id="places-detail")
                if location_detail:
                    location_detail = location_detail.get_text(strip=True)
                    metadata_dict["Location Detail"] = location_detail

                # Capture des irrégularités
                texirr_list = soup.find("ul", id="texirr-list")
                if texirr_list:
                    texirr = []
                    for el in texirr_list.children:
                        if len(el.text) > 2:
                            texirr.append(el.text)
                    metadata_dict["Text Irregularities"] = texirr
                    metadata_dict["url"] = url

        return metadata_dict
    else:
        print(f"Failed to retrieve html page for {url}")
        return None


"""3) Réunir tout le travail en une seule fonction """
# Function to write the metadata to a CSV file


def write_data_to_csv(data, csv_filename="papyrus_corpus.csv"):
    """
    :param data:  un csv traite chaque ligne / row comme un dictionnaire
    :param csv_filename: path du fichier final
    """
    if data:
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())

        # Tri des Id en ordre croissant pour plus de lisibilité
        fieldnames = ["ID"] + sorted(all_keys - {"ID"})

        with open(csv_filename, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                filled_row = {key: row.get(key, np.nan) for key in fieldnames}
                writer.writerow(filled_row)
    else:
        print("No data to write to CSV")


# Fonction principale qui assure tout le travail
def scrape_all_papyri_from_corpus(path):

    url_list = get_url_list(path)

    # chaque ligne représente un papyrus. NB il y aurait sans doute une manière plus élégante de le faire avec une liste
    # par compréhension, j'ai fait vite...
    all_metadata = []
    for url in tqdm(url_list):
        metadata = scrape_metadata(url)
        if metadata:
            metadata["ID"] = (
                f'TM {url.replace("https://www.trismegistos.org/text/","")}'
            )
            all_metadata.append(metadata)

        # le serveur appréciera une petite pause entre deux pages. Je la rends aléatoire pour limiter les risques de le
        # saturer
        time_to_sleep = random.uniform(0, 2)
        time.sleep(time_to_sleep)

    write_data_to_csv(all_metadata)
    return all_metadata


if __name__ == "__main__":
    # Plus qu'à lancer le script ; rapide et efficace !
    path_to_initial_csv = "./corpus_basic_metadata.csv"
    scrape_all_papyri_from_corpus(path_to_initial_csv)
