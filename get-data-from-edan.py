import socket
import ipaddress
import psutil
import time
import datetime
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

PORT = 9100


def horodatage():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ==========================================
# DÉCOUVERTE AUTOMATIQUE DU MONITEUR
# ==========================================
def obtenir_sous_reseaux_locaux():
    """Détecte tous les sous-réseaux /24 accessibles depuis ce PC
    (Wi-Fi, Ethernet, etc.), en ignorant loopback et link-local."""
    sous_reseaux = []
    for interface, adresses in psutil.net_if_addrs().items():
        for adresse in adresses:
            if adresse.family == socket.AF_INET:
                ip = adresse.address
                if ip.startswith("127.") or ip.startswith("169.254."):
                    continue
                try:
                    reseau = ipaddress.ip_network(f"{ip}/24", strict=False)
                    if reseau not in sous_reseaux:
                        sous_reseaux.append(reseau)
                except Exception:
                    continue
    return sous_reseaux


def tester_ip(ip, port=PORT, timeout=0.3):
    """Teste si le port est ouvert sur cette IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            resultat = s.connect_ex((str(ip), port))
            return str(ip) if resultat == 0 else None
    except Exception:
        return None


def decouvrir_moniteur_edan(port=PORT, max_workers=50):
    """Scanne tous les sous-réseaux locaux, retourne la première IP
    qui répond vraiment en HL7/XML (contient OBX ou ORU_R01)."""
    sous_reseaux = obtenir_sous_reseaux_locaux()

    ips_a_tester = []
    for reseau in sous_reseaux:
        ips_a_tester.extend(list(reseau.hosts()))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        resultats = executor.map(lambda ip: tester_ip(ip, port), ips_a_tester)
        candidats = [ip for ip in resultats if ip is not None]

    for ip in candidats:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((ip, port))
                data = recevoir_message_complet(s)
                if b"OBX" in data or b"ORU_R01" in data:
                    return ip
        except Exception:
            continue

    return None


# ==========================================
# LECTURE ET DÉCODAGE DU MONITEUR (HL7/XML)
# ==========================================
def recevoir_message_complet(sock):
    buffer = b""
    sock.settimeout(3)
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
    except socket.timeout:
        pass
    return buffer


def decoder_xml_hl7(data):
    try:
        texte = data.decode("utf-8", errors="ignore")
        root = ET.fromstring(texte)
        resultats = {}
        for obx in root.iter("OBX"):
            nom_elem = obx.find("OBX.3")
            val_elem = obx.find("OBX.5")
            unite_elem = obx.find("OBX.6")
            nom = nom_elem.text if nom_elem is not None else "?"
            val = val_elem.text if val_elem is not None else "?"
            unite = unite_elem.text if unite_elem is not None else ""
            resultats[nom] = (val, unite)
        return resultats
    except ET.ParseError:
        return None


def lire_moniteur(ip, port=PORT):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((ip, port))
            data = recevoir_message_complet(s)
            if data:
                return decoder_xml_hl7(data)
    except Exception:
        return None
    return None


# ==========================================
# PROGRAMME PRINCIPAL
# ==========================================
print("=" * 60)
print("DÉTECTION AUTOMATIQUE DU MONITEUR EDAN ELITE V6")
print("=" * 60)

print(f"\n[{horodatage()}] 🔍 Scan du réseau en cours...")
ip_moniteur = decouvrir_moniteur_edan()

if ip_moniteur is None:
    print(f"[{horodatage()}] ❌ Aucun moniteur EDAN trouvé sur le réseau.")
    exit()

print(f"[{horodatage()}] ✅ Moniteur trouvé : {ip_moniteur}:{PORT}\n")
print("Ctrl+C pour arrêter.\n")

while True:
    try:
        parametres = lire_moniteur(ip_moniteur)

        if parametres:
            print(f"[{horodatage()}]", end="  ")
            for nom, (val, unite) in parametres.items():
                print(f"{nom}={val}{unite}", end="  ")
            print()
        else:
            print(f"[{horodatage()}] ⏳ Aucune donnée reçue")

    except Exception as e:
        print(f"[{horodatage()}] Erreur : {e}")

    time.sleep(2)