import random

def fake_moniteur_edan_septique():
    """
    Simule un moniteur EDAN connecté à un patient en choc septique.
    À chaque appel, renvoie de nouvelles valeurs légèrement différentes
    (comme un vrai moniteur qui fluctue en continu), toujours dans la
    marge clinique du choc septique.
    """
    vitals = {
        "HR": round(random.uniform(115, 145), 1),
        "O2Sat": round(random.uniform(85, 93), 1),
        "Temp": round(random.uniform(38.5, 40.2), 1),
        "SBP": round(random.uniform(75, 95), 1),
        "MAP": round(random.uniform(50, 62), 1),
        "DBP": round(random.uniform(38, 52), 1),
        "Resp": round(random.uniform(22, 32), 1),
    }

    biologie = {
        "lactate": {"valeur": str(round(random.uniform(3.0, 6.5), 1)), "heure": 1},
        "creatinine": {"valeur": str(round(random.uniform(1.8, 3.0), 2)), "heure": 1},
        "bilirubine": {"valeur": str(round(random.uniform(1.5, 3.5), 2)), "heure": 1},
        "plaquettes": {"valeur": str(round(random.uniform(60, 120), 0)), "heure": 1},
        "globules_blancs": {"valeur": str(round(random.uniform(14, 22), 1)), "heure": 1},
        "uree": {"valeur": str(round(random.uniform(35, 60), 1)), "heure": 1},
        "ph": {"valeur": str(round(random.uniform(7.20, 7.32), 2)), "heure": 1},
        "paco2": {"valeur": str(round(random.uniform(28, 36), 1)), "heure": 1},
        "fio2": {"valeur": str(round(random.uniform(0.4, 0.6), 2)), "heure": 1},
        "glucose": {"valeur": str(round(random.uniform(150, 220), 0)), "heure": 1},
    }

    return vitals, biologie