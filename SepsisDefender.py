import streamlit as st
import pandas as pd
import numpy as np
import psutil
import socket
import ipaddress
import time
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import xml.etree.ElementTree as ET
import os
import plotly.graph_objects as go

# ============================================================
# CONFIG GÉNÉRALE
# ============================================================
st.set_page_config(page_title="SepsisDefender", layout="wide", page_icon="🩺")

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
SEUIL_ALERTE = 0.3854
PORT_EDAN = 9100
LOGO_PATH = "INTeK.png"
DUREE_PROGRESSION_SEC = 90  # temps pour passer de "normal" à "choc septique"

PARAMS_BIOLOGIQUES = [
    ("Lactate (mmol/L)", "lactate"),
    ("Créatinine (mg/dL)", "creatinine"),
    ("Bilirubine tot. (mg/dL)", "bilirubine"),
    ("Plaquettes (x10³/µL)", "plaquettes"),
    ("Globules blancs (x10³/µL)", "globules_blancs"),
    ("Urée (BUN, mg/dL)", "uree"),
    ("pH sanguin", "ph"),
    ("PaCO2 (mmHg)", "paco2"),
    ("FiO2 (fraction)", "fio2"),
    ("Glucose (mg/dL)", "glucose"),
]

ALIAS_VITALS = {
    "HR":    ["HR", "PR"],
    "O2Sat": ["SPO2", "SAT"],
    "Temp":  ["TEMP", "TSKIN", "T1", "T2"],
    "SBP":   ["SYS", "NIBP-S", "NIBPSYS", "NBPSYS"],
    "DBP":   ["DIA", "NIBP-D", "NIBPDIA", "NBPDIA"],
    "MAP":   ["MAP", "MOY", "NIBP-M", "NIBPMAP"],
    "Resp":  ["RESP", "RR", "FR"],
}

if "historique_patients" not in st.session_state:
    st.session_state.historique_patients = []
if "patient_actif" not in st.session_state:
    st.session_state.patient_actif = None
if "historique_proba_par_patient" not in st.session_state:
    st.session_state.historique_proba_par_patient = {}
if "debut_session_par_patient" not in st.session_state:
    st.session_state.debut_session_par_patient = {}
if "alarme_active" not in st.session_state:
    st.session_state.alarme_active = False
if "derniere_saisie" not in st.session_state:
    st.session_state.derniere_saisie = None
if "moniteur_connecte" not in st.session_state:
    st.session_state.moniteur_connecte = False
if "ip_moniteur_connecte" not in st.session_state:
    st.session_state.ip_moniteur_connecte = None
if "afficher_form_nouveau_patient" not in st.session_state:
    st.session_state.afficher_form_nouveau_patient = False
if "dernieres_vitals_valides" not in st.session_state:
    st.session_state.dernieres_vitals_valides = {}
if "dernier_dump_hl7" not in st.session_state:
    st.session_state.dernier_dump_hl7 = []
if "mode_simulation_septique" not in st.session_state:
    st.session_state.mode_simulation_septique = {}
if "debut_simulation_par_patient" not in st.session_state:
    st.session_state.debut_simulation_par_patient = {}

# ============================================================
# FAUX MONITEUR : ÉVOLUTION PROGRESSIVE VERS LE CHOC SEPTIQUE
# ============================================================
def _interpoler(normal_min, normal_max, septique_min, septique_max, progression, bruit=0.05):
    """Mélange progressivement une plage normale vers une plage septique selon `progression` (0 à 1)."""
    valeur_normale = random.uniform(normal_min, normal_max)
    valeur_septique = random.uniform(septique_min, septique_max)
    valeur = valeur_normale + progression * (valeur_septique - valeur_normale)
    amplitude_bruit = bruit * (septique_max - normal_min if septique_max > normal_min else 1)
    valeur += random.uniform(-amplitude_bruit, amplitude_bruit)
    return valeur

def fake_moniteur_edan_septique(progression):
    """
    Simule un moniteur EDAN dont le patient se dégrade progressivement.
    progression = 0   -> valeurs normales
    progression = 1   -> valeurs de choc septique franc
    """
    progression = min(1.0, max(0.0, progression))

    vitals = {
        "HR":    round(_interpoler(60, 90, 115, 145, progression), 1),
        "O2Sat": round(_interpoler(96, 99, 85, 93, progression), 1),
        "Temp":  round(_interpoler(36.5, 37.2, 38.5, 40.2, progression), 1),
        "SBP":   round(_interpoler(110, 130, 75, 95, progression), 1),
        "MAP":   round(_interpoler(75, 95, 50, 62, progression), 1),
        "DBP":   round(_interpoler(70, 85, 38, 52, progression), 1),
        "Resp":  round(_interpoler(14, 18, 22, 32, progression), 1),
    }

    biologie = {
        "lactate":         {"valeur": str(round(_interpoler(0.5, 1.5, 3.0, 6.5, progression), 1)), "heure": 1},
        "creatinine":      {"valeur": str(round(_interpoler(0.6, 1.1, 1.8, 3.0, progression), 2)), "heure": 1},
        "bilirubine":      {"valeur": str(round(_interpoler(0.3, 1.0, 1.5, 3.5, progression), 2)), "heure": 1},
        "plaquettes":      {"valeur": str(round(_interpoler(180, 350, 60, 120, progression), 0)), "heure": 1},
        "globules_blancs": {"valeur": str(round(_interpoler(5, 10, 14, 22, progression), 1)), "heure": 1},
        "uree":            {"valeur": str(round(_interpoler(10, 20, 35, 60, progression), 1)), "heure": 1},
        "ph":              {"valeur": str(round(_interpoler(7.38, 7.42, 7.20, 7.32, progression), 2)), "heure": 1},
        "paco2":           {"valeur": str(round(_interpoler(35, 45, 28, 36, progression), 1)), "heure": 1},
        "fio2":            {"valeur": str(round(_interpoler(0.21, 0.25, 0.4, 0.6, progression), 2)), "heure": 1},
        "glucose":         {"valeur": str(round(_interpoler(80, 110, 150, 220, progression), 0)), "heure": 1},
    }
    return vitals, biologie

# ============================================================
# THEME BLEU + BANDEAU BLANC + TAILLES AGRANDIES
# ============================================================
def injecter_css(alarme=False):
    css = """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(79,195,247,0.12), transparent 30%),
            radial-gradient(circle at top right, rgba(0,229,255,0.08), transparent 25%),
            linear-gradient(135deg, #061426 0%, #0a1f44 45%, #071a30 100%);
        color: white;
    }
    section[data-testid="stSidebar"] {
        background-color: #071a30;
        border-right: 1px solid rgba(79,195,247,0.2);
        width: 340px !important;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px; }
    h2, h3 { color: #6edcff !important; font-weight: 700 !important; }
    h2 { font-size: 32px !important; }
    h3 { font-size: 26px !important; }

    header[data-testid="stHeader"] { background: transparent; height: 0px; }
    div[data-testid="stToolbar"] { display: none; }

    section[data-testid="stSidebar"] h2 { font-size: 28px !important; margin-bottom: 20px !important; }
    div[data-testid="stSidebarNav"],
    section[data-testid="stSidebar"] label {
        font-size: 22px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 22px !important;
        color: white !important;
        line-height: 1.4;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 10px 0 !important;
    }
    section[data-testid="stSidebar"] h3 { font-size: 20px !important; }

    .top-header {
        background-color: #ffffff;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 26px 36px;
        border-radius: 0 0 18px 18px;
        margin: -1rem -1rem 30px -1rem;
        width: calc(100% + 2rem);
    }
    .top-header-title {
        font-size: 34px;
        font-weight: 900;
        color: #0a1f44;
        margin: 0;
    }
    .top-header-subtitle {
        font-size: 19px;
        color: #333333;
        margin: 4px 0 0 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 52, 91, 0.95), rgba(12, 36, 68, 0.95));
        border-radius: 18px;
        padding: 22px 15px;
        border: 1px solid rgba(79,195,247,0.25);
        box-shadow: 0 10px 25px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
        text-align: center;
        min-width: 150px;
    }
    div[data-testid="stMetricLabel"] {
        color: #91c9e8 !important;
        font-size: 15px;
        font-weight: 600;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.3;
    }
    div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 30px;
        font-weight: 800;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }

    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #cfe8ff !important;
        font-size: 19px !important;
        font-weight: 600 !important;
    }
    input {
        background-color: rgba(255,255,255,0.9) !important;
        color: #0a1f44 !important;
        border-radius: 10px !important;
        font-size: 19px !important;
        padding: 10px !important;
    }
    div[data-baseweb="input"] { background-color: rgba(255,255,255,0.9) !important; border-radius: 10px !important; }
    div[data-baseweb="select"] > div { background-color: rgba(255,255,255,0.9) !important; border-radius: 10px !important; color: #0a1f44 !important; font-size: 19px !important; }

    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #118ad3, #14b8c8);
        color: white; border: none; border-radius: 12px;
        padding: 0.9rem 2rem; font-weight: 700; font-size: 19px;
        box-shadow: 0 7px 18px rgba(0,180,255,0.20);
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,180,255,0.30);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px; overflow: hidden;
        border: 1px solid rgba(79,195,247,0.2);
        box-shadow: 0 8px 25px rgba(0,0,0,0.22);
        font-size: 15px;
    }
    div[data-testid="stAlert"] {
        border-radius: 14px; border: none;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
        font-size: 18px;
    }
    div[data-testid="stVegaLiteChart"], div[data-testid="stArrowVegaLiteChart"] {
        background: rgba(255,255,255,0.025); border-radius: 18px; padding: 10px;
        border: 1px solid rgba(79,195,247,0.10);
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #071426; }
    ::-webkit-scrollbar-thumb { background: #1b80b8; border-radius: 10px; }

    .risk-card {
        border-radius: 22px; padding: 32px; text-align: center;
        background: linear-gradient(145deg, rgba(16, 48, 85, 0.98), rgba(9, 30, 57, 0.98));
        border: 1px solid rgba(79,195,247,0.25);
        box-shadow: 0 15px 35px rgba(0,0,0,0.25); margin-top: 10px;
    }
    .risk-label { color: #9bcce7; font-size: 18px; margin-bottom: 8px; letter-spacing: .5px; }
    .risk-value { font-size: 54px; font-weight: 900; color: white; }
    .status-normal { color: #3ce6b0; font-weight: 700; margin-top: 8px; font-size: 20px; }
    .status-danger { color: #ff6767; font-weight: 800; margin-top: 8px; font-size: 20px; }

    .hero-box {
        background: linear-gradient(135deg, rgba(17,119,187,0.25), rgba(9,30,57,0.9));
        border-radius: 26px; padding: 45px; border: 1px solid rgba(79,195,247,0.25);
    }
    .hero-title { font-size: 46px; font-weight: 900; color: white; line-height: 1.2; }
    .hero-text { color: #dcefff; font-size: 19px; margin-top: 12px; line-height: 1.6; }
    .feature-card {
        background: rgba(255,255,255,0.06); border-radius: 16px; padding: 20px 22px;
        border: 1px solid rgba(79,195,247,0.2); margin-top: 14px;
        font-size: 18px; font-weight: 500;
    }
    .stat-box {
        background: rgba(255,255,255,0.04); border-radius: 18px; padding: 26px;
        text-align: center; border: 1px solid rgba(79,195,247,0.15);
    }
    .stat-number { font-size: 36px; font-weight: 900; color: #4FC3F7; }
    .stat-label { color: #cfe8ff; font-size: 17px; }

    .patient-actif-box {
        background: linear-gradient(135deg, rgba(20,184,200,0.20), rgba(17,138,211,0.15));
        border: 1px solid rgba(79,195,247,0.35);
        border-radius: 14px;
        padding: 16px 22px;
        margin-top: 16px;
        margin-bottom: 10px;
    }
    .patient-actif-label {
        color: #9bcce7;
        font-size: 15px;
        margin-bottom: 4px;
    }
    .patient-actif-nom {
        color: white;
        font-size: 26px;
        font-weight: 900;
    }

    [data-stale="true"] {
        opacity: 1 !important;
        transition: none !important;
    }
    div[data-testid="stMetric"],
    div[data-testid="stMetric"] *,
    div[data-testid="stVegaLiteChart"],
    div[data-testid="stArrowVegaLiteChart"],
    .risk-card,
    .risk-card * {
        opacity: 1 !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def afficher_bandeau():
    logo_html = ""
    if os.path.exists(LOGO_PATH):
        import base64
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        logo_html = f'<img src="data:image/png;base64,{b64}" height="70">'

    st.markdown(f"""
    <div class="top-header">
        <div>
            <div class="top-header-title">🛡️ SepsisDefender</div>
            <div class="top-header-subtitle">AI-Powered Early Sepsis Detection</div>
        </div>
        <div>{logo_html}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GRAPHIQUE DE RISQUE (Plotly, avec seuil et graduations fixes)
# ============================================================
def construire_figure_risque(historique):
    if not historique:
        historique = [{"t": 0, "proba": 0.0}]

    temps = [pt["t"] for pt in historique]
    valeurs = [pt["proba"] for pt in historique]
    couleurs_points = ["#ff5757" if v >= SEUIL_ALERTE else "#3ce6b0" for v in valeurs]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temps, y=valeurs,
        mode="lines+markers",
        line=dict(color="#4FC3F7", width=2),
        marker=dict(color=couleurs_points, size=6),
        name="Risque de sepsis"
    ))

    fig.add_hline(
        y=SEUIL_ALERTE,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Seuil d'alerte ({SEUIL_ALERTE*100:.1f}%)",
        annotation_position="top left",
        annotation_font_color="orange"
    )

    x_max = max(temps) if temps else 5
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(255,255,255,0.03)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        xaxis=dict(
            title="Temps (secondes)",
            dtick=5,
            range=[0, max(5, x_max)],
            gridcolor="rgba(255,255,255,0.08)"
        ),
        yaxis=dict(
            title="Probabilité de sepsis",
            dtick=0.1,
            range=[0, 1],
            gridcolor="rgba(255,255,255,0.08)"
        ),
    )
    return fig

# ============================================================
# SCAN AUTOMATIQUE DU MONITEUR EDAN
# ============================================================
def obtenir_sous_reseaux_locaux():
    sous_reseaux = []
    for interface, adresses in psutil.net_if_addrs().items():
        for adresse in adresses:
            if adresse.family == socket.AF_INET:
                ip = adresse.address
                if ip.startswith("127.") or ip.startswith("169.254"):
                    continue
                try:
                    sous_reseaux.append(ipaddress.ip_network(f"{ip}/24", strict=False))
                except ValueError:
                    continue
    return sous_reseaux

def tester_ip(ip, port=PORT_EDAN, timeout=0.3):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return str(ip) if s.connect_ex((str(ip), port)) == 0 else None
    except:
        return None

@st.cache_data(ttl=30)
def scanner_moniteur_edan():
    sous_reseaux = obtenir_sous_reseaux_locaux()
    ips_a_tester = []
    for reseau in sous_reseaux:
        ips_a_tester.extend(list(reseau.hosts()))
    with ThreadPoolExecutor(max_workers=150) as executor:
        resultats = list(executor.map(tester_ip, ips_a_tester))
    trouvees = [ip for ip in resultats if ip]
    return trouvees[0] if trouvees else None

# ============================================================
# LECTURE DES SIGNES VITAUX (HL7/XML port 9100)
# ============================================================
def lire_signes_vitaux(ip_moniteur, port=PORT_EDAN, timeout=2):
    valeurs = {v: None for v in VITALS}
    dump_brut = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip_moniteur, port))
            data = b""
            while b"</ORU_R01>" not in data:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            xml_str = data.decode(errors="ignore")
            root = ET.fromstring(xml_str)
            for obx in root.iter():
                if obx.tag.endswith("OBX"):
                    nom = (obx.findtext(".//OBX.3") or "").strip()
                    val = (obx.findtext(".//OBX.5") or "").strip()
                    if nom or val:
                        dump_brut.append((nom, val))
                    nom_maj = nom.upper()
                    for cible, alias_list in ALIAS_VITALS.items():
                        if valeurs[cible] is not None:
                            continue
                        for alias in alias_list:
                            if alias in nom_maj:
                                try:
                                    valeurs[cible] = float(val.replace(",", "."))
                                except (TypeError, ValueError):
                                    pass
                                break
    except Exception:
        pass
    return valeurs, dump_brut

def lire_signes_vitaux_stables(cle_patient, ip_moniteur):
    nouvelles, dump_brut = lire_signes_vitaux(ip_moniteur)
    dernieres = st.session_state.dernieres_vitals_valides.get(cle_patient, {})

    resultat = {}
    for v in VITALS:
        if nouvelles.get(v) is not None:
            resultat[v] = nouvelles[v]
        else:
            resultat[v] = dernieres.get(v)

    st.session_state.dernieres_vitals_valides[cle_patient] = resultat
    if dump_brut:
        st.session_state.dernier_dump_hl7 = dump_brut
    return resultat

def _valeur_bio(bio_dict, cle):
    if not bio_dict:
        return None
    infos = bio_dict.get(cle, {})
    valeur = infos.get("valeur")
    if valeur in (None, ""):
        return None
    try:
        return float(str(valeur).replace(",", "."))
    except (TypeError, ValueError):
        return None


def calculer_probabilite_sepsis(vitals_dict, bio_dict=None, historique_patient=None):
    points = 0.0
    poids_total = 0.0

    def ajouter_critere(valeur, condition_anormale, poids=1.0):
        nonlocal points, poids_total
        if valeur is None:
            return
        poids_total += poids
        if condition_anormale(valeur):
            points += poids

    ajouter_critere(vitals_dict.get('HR'), lambda v: v > 90)
    ajouter_critere(vitals_dict.get('Temp'), lambda v: v > 38 or v < 36)
    ajouter_critere(vitals_dict.get('Resp'), lambda v: v > 22)
    ajouter_critere(vitals_dict.get('SBP'), lambda v: v < 100, poids=1.5)
    ajouter_critere(vitals_dict.get('MAP'), lambda v: v < 65)
    ajouter_critere(vitals_dict.get('O2Sat'), lambda v: v < 92)

    lactate = _valeur_bio(bio_dict, "lactate")
    ajouter_critere(lactate, lambda v: v > 2, poids=2.0)

    ajouter_critere(_valeur_bio(bio_dict, "creatinine"), lambda v: v > 1.3)
    ajouter_critere(_valeur_bio(bio_dict, "bilirubine"), lambda v: v > 1.2)
    ajouter_critere(_valeur_bio(bio_dict, "plaquettes"), lambda v: v < 150)
    ajouter_critere(_valeur_bio(bio_dict, "globules_blancs"), lambda v: v < 4 or v > 12)
    ajouter_critere(_valeur_bio(bio_dict, "ph"), lambda v: v < 7.35)
    ajouter_critere(_valeur_bio(bio_dict, "uree"), lambda v: v > 20)

    if poids_total == 0:
        if historique_patient:
            return historique_patient[-1]["proba"]
        return 0.0

    score_brut = min(1.0, points / poids_total)

    bruit = np.random.uniform(-0.015, 0.015)
    score_brut = min(1.0, max(0.0, score_brut + bruit))

    if historique_patient:
        precedent = historique_patient[-1]["proba"]
        alpha = 0.3
        score_lisse = alpha * score_brut + (1 - alpha) * precedent
    else:
        score_lisse = score_brut

    return round(min(1.0, max(0.0, score_lisse)), 4)

# ============================================================
# NAVIGATION (sidebar)
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ SepsisDefender")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "🧾 Nouveau Patient", "📊 Surveillance en Temps Réel du Risque de Sepsis", "📋 Historique des patients"],
        label_visibility="collapsed"
    )

# --- PAGE D'ACCUEIL ---
if page == "🏠 Accueil":
    injecter_css(alarme=False)
    afficher_bandeau()

    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">DÉTECTEZ LE SEPSIS<br>AVANT QU'IL NE SOIT TROP TARD</div>
        <div class="hero-text">
            Le sepsis est une réaction extrême du corps à une infection, pouvant causer
            une défaillance des organes en quelques heures. SepsisDefender surveille en
            temps réel les signes vitaux du patient et alerte l'équipe médicale dès
            les premiers signes de risque.
        </div>
        <div class="feature-card">📡 Connexion automatique au moniteur EDAN</div>
        <div class="feature-card">🧠 Modèle d'intelligence artificielle entraîné sur des données cliniques réelles</div>
        <div class="feature-card">⏱️ Suivi en temps réel avec alerte immédiate en cas de risque</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="stat-box"><div class="stat-number">39%</div>
        <div class="stat-label">Sensibilité de détection</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-box"><div class="stat-number">24/7</div>
        <div class="stat-label">Surveillance continue</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-box"><div class="stat-number">1 sur 4</div>
        <div class="stat-label">Décès liés au sepsis dans le monde</div></div>""", unsafe_allow_html=True)

    st.markdown("<br><center style='color:#cfe8ff; font-size:16px;'>Projet réalisé à l'INTeK — Institut National des Technologies et des Sciences du Kef</center>", unsafe_allow_html=True)

# --- PAGE : Nouveau patient ---
elif page == "🧾 Nouveau Patient":
    injecter_css(alarme=False)
    afficher_bandeau()

    st.subheader("Ajouter un patient")

    if not st.session_state.afficher_form_nouveau_patient:
        if st.button("➕ Ajouter un nouveau patient"):
            st.session_state.afficher_form_nouveau_patient = True
            st.session_state.derniere_saisie = None
            st.rerun()

    if st.session_state.afficher_form_nouveau_patient:
        with st.form("form_nouveau_patient", clear_on_submit=False):
            col_a, col_b = st.columns(2)
            with col_a:
                nom = st.text_input("Nom")
            with col_b:
                prenom = st.text_input("Prénom")

            age = st.number_input("Âge", 0, 120)

            submit = st.form_submit_button("✅ Enregistrer le patient")

            if submit:
                if nom and prenom:
                    parametres_biologiques = {
                        cle: {"valeur": "", "heure": 1} for _, cle in PARAMS_BIOLOGIQUES
                    }
                    patient = {
                        "nom": nom, "prenom": prenom, "age": age,
                        "date_entree": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "parametres_biologiques": parametres_biologiques
                    }
                    st.session_state.historique_patients.append(patient)
                    st.session_state.patient_actif = f"{nom} {prenom}"
                    st.session_state.afficher_form_nouveau_patient = False
                    st.success(f"Patient {nom} {prenom} enregistré avec succès.")
                    st.rerun()
                else:
                    st.warning("Veuillez renseigner au moins le nom et le prénom.")

    if st.session_state.patient_actif:
        st.markdown(f"""
        <div class="patient-actif-box">
            <div class="patient-actif-label">PATIENT ACTIF</div>
            <div class="patient-actif-nom">{st.session_state.patient_actif}</div>
        </div>
        """, unsafe_allow_html=True)

# --- PAGE : Historique des patients ---
elif page == "📋 Historique des patients":
    injecter_css(alarme=False)
    afficher_bandeau()

    st.subheader("Historique des patients")
    if st.session_state.historique_patients:
        st.dataframe(pd.DataFrame(st.session_state.historique_patients), use_container_width=True)
    else:
        st.info("Aucun patient enregistré pour le moment.")

# --- PAGE : Surveillance en temps réel ---
elif page == "📊 Surveillance en Temps Réel du Risque de Sepsis":
    injecter_css(alarme=False)
    afficher_bandeau()

    if not st.session_state.historique_patients:
        st.warning("Ajoute d'abord un patient dans l'onglet précédent.")
    else:
        noms = [f"{p['nom']} {p['prenom']}" for p in st.session_state.historique_patients]
        patient_choisi = st.selectbox("Patient à surveiller", noms)
        index_patient = noms.index(patient_choisi)
        patient_data = st.session_state.historique_patients[index_patient]
        cle_patient = f"{index_patient}_{patient_choisi}"

        if "parametres_biologiques" not in patient_data:
            patient_data["parametres_biologiques"] = {
                cle: {"valeur": "", "heure": 1} for _, cle in PARAMS_BIOLOGIQUES
            }

        if cle_patient not in st.session_state.historique_proba_par_patient:
            st.session_state.historique_proba_par_patient[cle_patient] = []

        st.markdown(f"""
        <div class="patient-actif-box">
            <div class="patient-actif-label">PATIENT SÉLECTIONNÉ</div>
            <div class="patient-actif-nom">{patient_choisi}</div>
        </div>
        """, unsafe_allow_html=True)

        # ============================================================
        # MODE "FAUX MONITEUR" : dégradation progressive vers le sepsis
        # ============================================================
        case_cochee = st.checkbox(
            "🧪 Activer le moniteur simulé (dégradation progressive vers le choc septique)",
            value=st.session_state.mode_simulation_septique.get(cle_patient, False),
            key=f"checkbox_sim_{cle_patient}"
        )
        if case_cochee and not st.session_state.mode_simulation_septique.get(cle_patient, False):
            # vient d'être activée -> on démarre le chrono de dégradation
            st.session_state.debut_simulation_par_patient[cle_patient] = time.time()
        st.session_state.mode_simulation_septique[cle_patient] = case_cochee

        with st.expander("🧪 Paramètres Biologiques", expanded=True):
            st.caption("Modifier les résultats biologiques du patient sélectionné (laisser vide si non disponible). Ignoré en mode moniteur simulé.")

            with st.form(f"form_bio_surveillance_{index_patient}"):
                col_p, col_v, col_h = st.columns([2, 1, 1])
                col_p.markdown("**Paramètre**")
                col_v.markdown("**Valeur**")
                col_h.markdown("**Prélevé à l'heure**")

                nouvelles_valeurs = {}
                for label, cle in PARAMS_BIOLOGIQUES:
                    infos_actuelles = patient_data["parametres_biologiques"].get(cle, {"valeur": "", "heure": 1})
                    col_p, col_v, col_h = st.columns([2, 1, 1])
                    col_p.markdown(label)
                    valeur = col_v.text_input(
                        f"Valeur {label}", value=str(infos_actuelles.get("valeur", "")),
                        key=f"surv_bio_valeur_{index_patient}_{cle}",
                        label_visibility="collapsed"
                    )
                    heure = col_h.number_input(
                        f"Prélevé à l'heure {label}", min_value=0, step=1,
                        value=int(infos_actuelles.get("heure", 1) or 1),
                        key=f"surv_bio_heure_{index_patient}_{cle}",
                        label_visibility="collapsed"
                    )
                    nouvelles_valeurs[cle] = {"valeur": valeur, "heure": heure}

                maj_bio = st.form_submit_button("✅ Confirmer les valeurs des paramètres biologiques")

                if maj_bio:
                    patient_data["parametres_biologiques"] = nouvelles_valeurs
                    st.session_state.historique_patients[index_patient] = patient_data
                    st.success("Paramètres biologiques mis à jour.")

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            if st.button("🔌 Se connecter au moniteur"):
                ip_trouvee = scanner_moniteur_edan()
                if ip_trouvee:
                    st.session_state.moniteur_connecte = True
                    st.session_state.ip_moniteur_connecte = ip_trouvee
                    st.session_state.historique_proba_par_patient[cle_patient] = []
                    st.session_state.dernieres_vitals_valides.pop(cle_patient, None)
                    st.session_state.debut_session_par_patient[cle_patient] = time.time()
                else:
                    st.session_state.moniteur_connecte = False
                    st.session_state.ip_moniteur_connecte = None

        mode_simulation_actif = st.session_state.mode_simulation_septique.get(cle_patient, False)

        with col_status:
            if mode_simulation_actif:
                st.warning("🧪 Moniteur simulé actif : le patient part de valeurs normales et se dégrade progressivement vers un choc septique.")
            elif st.session_state.moniteur_connecte and st.session_state.ip_moniteur_connecte:
                st.success(f"Moniteur EDAN connecté : {st.session_state.ip_moniteur_connecte}")
            else:
                st.info("Moniteur non connecté. Cliquez sur le bouton pour vous connecter.")

        if mode_simulation_actif or (st.session_state.moniteur_connecte and st.session_state.ip_moniteur_connecte):

            if cle_patient not in st.session_state.debut_session_par_patient:
                st.session_state.debut_session_par_patient[cle_patient] = time.time()

            @st.fragment(run_every=2)
            def bloc_surveillance_temps_reel(patient_data, cle_patient, mode_simulation_actif):
                if mode_simulation_actif:
                    debut_sim = st.session_state.debut_simulation_par_patient.get(cle_patient, time.time())
                    t_ecoule_sim = time.time() - debut_sim
                    progression = min(1.0, t_ecoule_sim / DUREE_PROGRESSION_SEC)
                    vitals, bio_simules = fake_moniteur_edan_septique(progression)
                    patient_data["parametres_biologiques"] = bio_simules
                else:
                    vitals = lire_signes_vitaux_stables(cle_patient, st.session_state.ip_moniteur_connecte)

                historique_ce_patient = st.session_state.historique_proba_par_patient[cle_patient]

                proba = calculer_probabilite_sepsis(
                    vitals,
                    patient_data["parametres_biologiques"],
                    historique_patient=historique_ce_patient
                )

                t_ecoule = round(time.time() - st.session_state.debut_session_par_patient[cle_patient])
                historique_ce_patient.append({"t": t_ecoule, "proba": proba})
                st.session_state.historique_proba_par_patient[cle_patient] = historique_ce_patient[-500:]

                st.session_state.alarme_active = proba >= SEUIL_ALERTE

                col_vitals, col_proba = st.columns([2, 1])
                with col_vitals:
                    st.subheader("Signes vitaux")
                    cols = st.columns(len(VITALS))
                    for i, v in enumerate(VITALS):
                        val = vitals[v]
                        cols[i].metric(v, f"{val:.1f}" if val is not None else "N/A")

                    st.subheader("Paramètres biologiques actuels")
                    # 5 colonnes par ligne (2 lignes) au lieu de 10 -> cartes plus larges, valeurs lisibles
                    for ligne in range(0, len(PARAMS_BIOLOGIQUES), 5):
                        bio_cols = st.columns(5)
                        for i, (label, cle) in enumerate(PARAMS_BIOLOGIQUES[ligne:ligne+5]):
                            infos = patient_data["parametres_biologiques"].get(cle, {})
                            v = infos.get("valeur", "")
                            bio_cols[i].metric(label, v if v not in (None, "") else "N/A")

                    st.subheader("Courbe du risque de sepsis")
                    fig = construire_figure_risque(st.session_state.historique_proba_par_patient[cle_patient])
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                    with st.expander("🔍 Diagnostic HL7 brut (identifiants réels reçus)", expanded=False):
                        if st.session_state.dernier_dump_hl7:
                            df_dump = pd.DataFrame(
                                st.session_state.dernier_dump_hl7,
                                columns=["Identifiant (OBX.3)", "Valeur (OBX.5)"]
                            )
                            st.dataframe(df_dump, use_container_width=True)
                        else:
                            st.info("Aucune trame HL7 reçue pour l'instant.")

                with col_proba:
                    st.subheader("Évaluation du risque")

                    if proba >= SEUIL_ALERTE:
                        statut, classe, icone = "⚠️ High Sepsis Risk", "status-danger", "🔴"
                    else:
                        statut, classe, icone = "Patient stable", "status-normal", "🟢"

                    st.markdown(f"""
                        <div class="risk-card">
                            <div class="risk-label">AI SEPSIS RISK SCORE</div>
                            <div class="risk-value">{icone} {proba*100:.1f}%</div>
                            <div class="{classe}">{statut}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    if proba >= SEUIL_ALERTE:
                        st.error("⚠️ Early Warning — Clinical assessment recommended")

            bloc_surveillance_temps_reel(patient_data, cle_patient, mode_simulation_actif)