
import streamlit as st
import numpy as np
import joblib
import os

from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SmartSeg | Data Mining",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS PERSONNALISE
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101827, #172B4D);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #172B4D;
    margin-bottom: 5px;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #172B4D, #2563EB);
    padding: 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #DBEAFE;
    font-size: 16px;
}

.stat-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 12px rgba(0,0,0,0.03);
    min-height: 115px;
}

.stat-number {
    color: #2563EB;
    font-size: 27px;
    font-weight: 800;
}

.stat-label {
    color: #64748B;
    font-size: 13px;
}

.dataset-card {
    background: white;
    padding: 23px;
    border-radius: 17px;
    border: 1px solid #E2E8F0;
    min-height: 235px;
    margin-bottom: 12px;
}

.dataset-card h3 {
    color: #172B4D;
    font-size: 20px;
}

.dataset-card p {
    color: #64748B;
    font-size: 14px;
}

.tag {
    display: inline-block;
    background: #DBEAFE;
    color: #1D4ED8;
    padding: 5px 11px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #2563EB;
    margin-top: 20px;
}

.stButton > button {
    background: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1D4ED8;
    color: white;
}

div[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. INFORMATIONS DES DATASETS
# ==========================================

datasets = {

    "Glass Identification": {
        "fichier": "verre",
        "methode": "K-Means",
        "icone": "",
        "couleur": "#2563EB",
        "description": "Étude des caractéristiques physiques et chimiques des échantillons de verre afin de découvrir des groupes similaires.",
        "objectif": "Identifier des groupes d'échantillons de verre.",
        "variables": "Caractéristiques chimiques du verre.",
        "image": ""
    },

    "Wheat Seeds": {
        "fichier": "grain_ble",
        "methode": "K-Means",
        "icone": "",
        "couleur": "#D97706",
        "description": "Analyse des caractéristiques géométriques des graines de blé afin de former des groupes présentant des propriétés similaires.",
        "objectif": "Regrouper les graines selon leurs caractéristiques.",
        "variables": "Mesures physiques des graines.",
        "image": ""
    },

    "Wholesale Customers": {
        "fichier": "depense",
        "methode": "CAH",
        "icone": "",
        "couleur": "#059669",
        "description": "Analyse des habitudes de dépenses des clients grossistes pour identifier différents profils de consommation.",
        "objectif": "Identifier des profils de clients selon leurs dépenses.",
        "variables": "Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper et Delicassen.",
        "image": ""
    }
}


# ==========================================
# 4. BARRE LATERALE
# ==========================================

with st.sidebar:

    st.markdown("""
    <h1 style='font-size:28px;'> SmartSeg</h1>
    <p style='color:#CBD5E1;'>Plateforme de Data Mining</p>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            " Accueil",
            " Glass Identification",
            " Wheat Seeds",
            " Wholesale Customers",
            " À propos"
        ]
    )

    st.markdown("---")
    st.caption("Projet Machine Learning | 2026")


# ==========================================
# 5. PAGE ACCUEIL
# ==========================================

if page == " Accueil":

    st.markdown("""
    <div class="hero">
        <h1>Bienvenue sur SmartSeg </h1>
        <p>
        Découvrez une nouvelle manière d'explorer les données
        grâce aux algorithmes de segmentation non supervisée.
        </p>
        <p>
        Analysez, regroupez et interprétez vos données
        avec K-Means et la Classification Ascendante Hiérarchique.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">Notre plateforme</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="subtitle">
    Trois datasets, deux algorithmes et une seule plateforme d'analyse.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">03</div>
            <div class="stat-label">Datasets étudiés</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">02</div>
            <div class="stat-label">Algorithmes utilisés</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">03</div>
            <div class="stat-label">Modèles de segmentation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Découvrez nos datasets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="dataset-card">
            <h3> Glass Identification</h3>
            <span class="tag">K-MEANS</span>
            <p>
            Étude des propriétés chimiques du verre pour
            découvrir des groupes d'échantillons similaires.
            </p>
            <b>Objectif :</b>
            <p>Regrouper les échantillons de verre.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="dataset-card">
            <h3> Wheat Seeds</h3>
            <span class="tag">K-MEANS</span>
            <p>
            Analyse des dimensions et des propriétés
            physiques des graines de blé.
            </p>
            <b>Objectif :</b>
            <p>Identifier les groupes de graines similaires.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="dataset-card">
        <h3> Wholesale Customers</h3>
        <span class="tag">CAH</span>
        <p>
        Analyse des dépenses des clients grossistes dans
        différentes catégories de produits.
        </p>
        <b>Objectif :</b>
        <p>Identifier les profils de consommation des clients.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Les algorithmes utilisés")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <div class="dataset-card">
            <h3> K-Means</h3>
            <p>
            Algorithme de clustering qui répartit les
            observations en K groupes autour de centroïdes.
            </p>
            <b>Utilisé pour :</b>
            <p>Glass Identification et Wheat Seeds.</p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="dataset-card">
            <h3> CAH</h3>
            <p>
            Algorithme hiérarchique qui construit progressivement
            des groupes en fonction des distances entre observations.
            </p>
            <b>Utilisé pour :</b>
            <p>Wholesale Customers.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Utilisez le menu à gauche pour explorer un dataset et effectuer une segmentation.")


# ==========================================
# 6. PAGES DE PREDICTION
# ==========================================

elif page in [
    " Glass Identification",
    " Wheat Seeds",
    " Wholesale Customers"
]:

    nom = page.split(" ", 1)[1]

    info = datasets[nom]

    st.markdown(
        f'<div class="main-title">{info["icone"]} {nom}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="subtitle">{info["description"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="dataset-card">
        <span class="tag">{info["methode"]}</span>
        <h3>Présentation du dataset</h3>
        <p><b>Objectif :</b> {info["objectif"]}</p>
        <p><b>Variables :</b> {info["variables"]}</p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement du modèle
    chemin = f'modeles/{info["fichier"]}.pkl'

    if not os.path.exists(chemin):
        st.error(f"Fichier du modèle introuvable : {chemin}")
        st.stop()

    m = joblib.load(chemin)

    st.markdown("## Effectuer une segmentation")

    st.write(
        "Saisissez les caractéristiques de l'individu "
        "pour découvrir son groupe."
    )

    with st.form("prediction_form"):

        # Channel et Region uniquement pour Wholesale Customers
        if nom == "Wholesale Customers":

            st.markdown("### Informations du client")

            col1, col2 = st.columns(2)

            with col1:
                channel = st.selectbox(
                    "Channel",
                    [1, 2],
                    format_func=lambda x: "Horeca" if x == 1 else "Retail"
                )

            with col2:
                region = st.selectbox(
                    "Region",
                    [1, 2, 3],
                    format_func=lambda x: {
                        1: "Lisbonne",
                        2: "Porto",
                        3: "Autre"
                    }[x]
                )

        if nom == "Glass Identification":

            st.markdown("### Caractéristiques de l'échantillon de verre")

        elif nom == "Wheat Seeds":

            st.markdown("### Caractéristiques des graines de blé")

        elif nom == "Wholesale Customers":

            st.markdown("### Dépenses du client")

        valeurs = []

        col1, col2 = st.columns(2)

        for i, col in enumerate(m["colonnes"]):

            with (col1 if i % 2 == 0 else col2):

                valeur = st.number_input(
                    col,
                    min_value=0.0,
                    value=0.0,
                    key=f"{info['fichier']}_{col}"
                )

                valeurs.append(valeur)

        submitted = st.form_submit_button(
            "Lancer la segmentation",
            use_container_width=True
        )

    if submitted:

        try:

            # Création de l'individu
            client = np.array(
                valeurs,
                dtype=float
            ).reshape(1, -1)

            # Normalisation
            if m.get("normalisation") == "normalize":

                client = normalize(client)

            elif m.get("scaler") is not None:

                client = m["scaler"].transform(client)

            # Vérification des dimensions
            if client.shape[1] != m["centroides"].shape[1]:

                st.error(
                    "Les variables saisies ne correspondent pas au modèle."
                )
                st.stop()

            # Calcul des distances
            distances = cdist(
                client,
                m["centroides"]
            )

            indice = distances.argmin()

            g = int(m["classes"][indice])

            # Description
            descriptions = m["descriptions"]

            if isinstance(descriptions, dict):

                description = descriptions.get(
                    str(g),
                    descriptions.get(
                        g,
                        "Description non disponible"
                    )
                )

            elif isinstance(descriptions, (list, tuple, np.ndarray)):

                description = descriptions[g]

            else:

                description = str(descriptions)

            # Résultat
            st.markdown("---")
            st.markdown("## Résultat de la segmentation")

            st.markdown(f"""
            <div class="result-card">
                <p style="color:#64748B;">GROUPE IDENTIFIÉ</p>
                <h1 style="color:#2563EB;font-size:40px;">
                    Groupe {g}
                </h1>
                <p style="color:#334155;font-size:16px;">
                    {description}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success("Segmentation réalisée avec succès !")

            with st.expander("Voir les détails techniques"):

                st.write("Méthode :", info["methode"])
                st.write("Groupe identifié :", g)
                st.write("Distances aux centroïdes :")

                st.write(distances)

        except Exception as e:

            st.error(f"Une erreur est survenue : {e}")


# ==========================================
# 7. PAGE A PROPOS
# ==========================================

elif page == "À propos":

    st.markdown(
        '<div class="main-title">À propos de SmartSeg</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero">
        <h1>SmartSeg </h1>
        <p>
        Une application de segmentation de données développée
        dans le cadre d'un projet de Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Objectif du projet")

    st.write("""
    Ce projet consiste à mettre en pratique les méthodes de
    clustering afin de découvrir des structures et des groupes
    similaires dans différents jeux de données.
    """)

    st.markdown("## Technologies utilisées")

    st.markdown("""
    - Python
    - Streamlit
    - Scikit-learn
    - NumPy
    - SciPy
    - Joblib
    """)

    st.markdown("---")

    st.caption("SmartSeg | Projet de Machine Learning et Data Mining")
