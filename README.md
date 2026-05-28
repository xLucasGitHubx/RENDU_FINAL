# Projets Machine Learning — Analyse COVID-19

Deux projets de Machine Learning autour de la pandémie de COVID-19, livrés dans le cadre d'un rendu final.

| Projet                 | Type de ML                              | Interface livrée                  |
| ---------------------- | --------------------------------------- | --------------------------------- |
| **1. Chatbot médical** | Classification supervisée + Clustering  | App **Streamlit** (`app.py`)      |
| **2. Carte dynamique** | Clustering géographique + Visualisation | Dashboard **HTML** (`index.html`) |

Chaque projet possède son propre notebook Jupyter détaillé (EDA, modélisation, évaluation, conclusion).

---

## Structure du projet

```
RENDU_FINAL/
├── README.md
├── requirements.txt                       # Dépendances communes (notebooks)
│
├── Projet1_Chatbot_COVID/
│   ├── projet_chatbot.ipynb               # Notebook complet — 44 cellules
│   ├── app.py                             # App Streamlit (interface chat)
│   ├── train_model.py                     # Script d'entraînement du modèle
│   ├── requirements.txt                   # Dépendances spécifiques à l'app
│   ├── artifacts/
│   │   └── chatbot_artifacts.joblib       # Modèle Random Forest + encodeurs
│   └── data/
│       ├── Raw-Data.csv                   # Vocabulaire source (Kaggle)
│       └── Cleaned-Data.csv               # 316 800 lignes générées
│
└── Projet2_Carte_Dynamique/
    ├── carte-dynamique.ipynb              # Notebook complet — 34 cellules
    ├── index.html                         # Dashboard HTML (4 cartes intégrées)
    └── cartes/
        ├── map_regions_deces.html         # Choroplèthe régions
        ├── map_depts_hosp.html            # Choroplèthe départements
        ├── dual_map.html                  # Carte synchronisée (Dual Map)
        └── map_clusters.html              # Clusters géographiques
```

---

## Installation

### Prérequis

- Python **3.10+** (testé sur 3.12)
- pip
- Navigateur web moderne

### 1. Cloner le dépôt

```bash
git clone <url-du-repo>
cd machine-learning-covid-rendu-final
```

### 2. (Recommandé) Créer un environnement virtuel

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Installer les dépendances

Pour exécuter les notebooks et toutes les fonctionnalités :

```bash
pip install -r requirements.txt
```

Pour ne lancer **que l'app Streamlit du Projet 1** (installation plus légère) :

```bash
pip install -r Projet1_Chatbot_COVID/requirements.txt
```

### 4. (Une seule fois) Enregistrer le kernel Jupyter

```bash
python -m ipykernel install --user --name python3
```

---

## Lancement

### 🩺 Projet 1 — Chatbot COVID-19 (Streamlit)

```bash
cd Projet1_Chatbot_COVID

# (Optionnel — le fichier artifacts/chatbot_artifacts.joblib est déjà fourni)
# Si tu veux ré-entraîner le modèle from scratch :
python train_model.py

# Lancer l'interface chat
streamlit run app.py
```

L'app s'ouvre automatiquement dans le navigateur sur `http://localhost:8501`.
Interface conversationnelle : 7 questions guidées (pays, âge, genre, symptômes, etc.) puis affichage du diagnostic et du niveau de confiance.

### 🗺 Projet 2 — Carte dynamique (HTML statique)

Aucune installation n'est nécessaire pour la visualisation. Ouvrir simplement :

```bash
# Windows
start Projet2_Carte_Dynamique/index.html

# Linux / macOS
open Projet2_Carte_Dynamique/index.html
```

Ou double-cliquer sur `Projet2_Carte_Dynamique/index.html`.
Le dashboard regroupe les 4 cartes interactives dans une interface unique avec navigation par onglets.

### 📓 Notebooks Jupyter (exploration complète)

```bash
jupyter notebook
```

Puis ouvrir :

- `Projet1_Chatbot_COVID/projet_chatbot.ipynb`
- `Projet2_Carte_Dynamique/carte-dynamique.ipynb`

Les sorties (graphiques, tableaux, résultats) sont déjà visibles dans les fichiers livrés.

---

## Projet 1 — Chatbot médical COVID-19

### Objectif

Construire un chatbot sous forme d'interrogatoire médical aidant à pré-diagnostiquer la COVID-19 à partir des symptômes du patient.

### Pipeline

1. **Génération des données** : produit cartésien de 7 variables catégorielles → **316 800 combinaisons** (10 pays × 5 âges × 3 genres × 16 symptômes × 11 symptômes secondaires × 4 sévérités × 3 contacts).
2. **Labelisation** : règle à base de score clinique inspirée des directives OMS (pondérations sur Fever, Dry-Cough, Difficulty-in-Breathing, etc., multiplicateur Contact, ajout selon Severity et Age).
3. **Feature engineering** : multi-hot encoding pour les listes de symptômes, One-Hot Encoding pour les variables simples.
4. **Comparaison de 5 modèles supervisés** : Logistic Regression, Random Forest, SVM (LinearSVC calibré), KNN, Gradient Boosting.
5. **Chatbot interactif** : interface Streamlit avec interrogatoire conversationnel et prédiction en temps réel.
6. **Clustering non supervisé** : PCA + KMeans (elbow + silhouette) + DBSCAN pour segmenter les profils patients.

### Résultats clés

| Modèle                | F1-score   | AUC-ROC |
| --------------------- | ---------- | ------- |
| Gradient Boosting     | **0.9999** | ~1.000  |
| Random Forest         | 0.997      | ~0.999  |
| SVM (Linear, calibré) | 0.989      | ~0.998  |
| Logistic Regression   | 0.988      | ~0.997  |
| KNN                   | 0.908      | ~0.95   |

- Le **meilleur modèle** est Gradient Boosting (F1=0.9999) ; l'app Streamlit livrée utilise un **Random Forest** (compromis taille/performance, F1=0.997).
- Le chatbot prédit correctement les cas-tests typiques (ex : patient 60+, contact Yes, symptômes sévères → POSITIF 100%).
- Le clustering KMeans suggère 3 profils homogènes de patients.

---

## Projet 2 — Carte dynamique COVID-19

### Objectif

Visualiser sur des cartes interactives la progression de l'épidémie COVID-19 en France (par région et département), et identifier des clusters géographiques homogènes par apprentissage non supervisé.

### Pipeline

1. **Données géographiques** : contours GeoJSON régions (13) et départements (96) depuis [gregoiredavid/france-geojson](https://github.com/gregoiredavid/france-geojson).
2. **Données épidémiques** : hospitalisations / réanimations / décès / retours à domicile depuis Santé Publique France ([data.gouv.fr](https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/)).
3. **Jointure** : sur le code INSEE des départements, agrégation par région via table officielle de correspondance.
4. **Clustering non supervisé** : StandardScaler + KMeans (choix de k par silhouette) + clustering hiérarchique Ward.
5. **Cartes Folium interactives** : choroplèthes, dual map synchronisée, carte des clusters.
6. **Bonus** : analyse de corrélation entre couverture vaccinale et indicateurs hospitaliers.

### Résultats clés

- **4 cartes HTML interactives** générées dans `Projet2_Carte_Dynamique/cartes/`.
- **Dashboard `index.html`** : navigation entre les 4 cartes avec onglets, KPIs, descriptions méthodologiques et mode plein écran.
- **Clustering optimal** : k=2 pour les départements (silhouette = 0.690), traduisant la fracture urbain/rural.
- **Corrélation vaccination/COVID** : Pearson +0.74 à +0.79 (corrélation positive — biais de variable confondante via la densité de population, expliqué dans le notebook).

> La **dual map synchronisée** (`dual_map.html`) est la pièce maîtresse : deux panneaux côte à côte (Hospitalisations × Décès) avec zoom et déplacement liés.

---

## Données

### Projet 1

- **Source** : [Kaggle COVID-19 Symptoms Checker](https://www.kaggle.com/iamhungundji/covid19-symptoms-checker)
- `Raw-Data.csv` (fourni) — vocabulaire des labels
- `Cleaned-Data.csv` (généré par le notebook ou par `train_model.py`) — 316 800 combinaisons étiquetées

### Projet 2

- **Contours** : [gregoiredavid/france-geojson](https://github.com/gregoiredavid/france-geojson) (régions + départements, IGN)
- **Indicateurs hospitaliers** : [Santé Publique France via data.gouv.fr](https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/)
- **Vaccination** : [SPF / data.gouv.fr](https://www.data.gouv.fr/fr/datasets/donnees-relatives-aux-personnes-vaccinees-contre-la-covid-19-1/)

> Les notebooks téléchargent automatiquement les données SPF (connexion internet requise au premier lancement).

---

## Stack technique

- **Manipulation de données** : pandas, numpy
- **Machine Learning** : scikit-learn (Pipeline, MultiLabelBinarizer, StandardScaler, PCA, KMeans, DBSCAN, AgglomerativeClustering, modèles supervisés)
- **Visualisation** : matplotlib, seaborn
- **Géospatial** : geopandas, shapely, folium
- **Interface chatbot** : Streamlit
- **Statistiques** : scipy (dendrogramme)
- **Sérialisation modèle** : joblib

---

## Notes

- Les notebooks ont été **exécutés intégralement sans erreur** (44 cellules pour le Projet 1, 34 pour le Projet 2). Les sorties (graphiques, tableaux, résultats) sont déjà visibles dans les fichiers `.ipynb` livrés.
- L'app Streamlit utilise un modèle **Random Forest** pré-entraîné (`artifacts/chatbot_artifacts.joblib`, fourni). Pour ré-entraîner depuis zéro, lancer `python train_model.py` (~2 min).
- Pour ré-exécuter complètement les notebooks, prévoir ~5 minutes pour le Projet 1 (entraînement des 5 modèles) et ~2 minutes pour le Projet 2 (téléchargements + clustering).

---

## Auteur

**Lucas Madjinda**
**Junior Chimene**
