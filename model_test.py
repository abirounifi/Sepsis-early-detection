import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================
# CHARGEMENT DU MODÈLE (XGBoost)
# ==========================================
model = joblib.load("./processed_data/xgboost_sepsis_model.pkl")
feature_cols = list(model.feature_names_in_)  # liste des colonnes utilisées à l'entraînement (XGBoost)

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
VITALS_LABELS = {
    'HR': 'Fréq. cardiaque (bpm)',
    'O2Sat': 'SpO2 (%)',
    'Temp': 'Température (°C)',
    'SBP': 'PA systolique (mmHg)',
    'MAP': 'PA moyenne (mmHg)',
    'DBP': 'PA diastolique (mmHg)',
    'Resp': 'Fréq. respiratoire (/min)',
}

LABS = ['Lactate', 'Creatinine', 'Bilirubin_total', 'Platelets', 'WBC', 'BUN', 'pH', 'PaCO2', 'FiO2', 'Glucose']
LABS_LABELS = {
    'Lactate': 'Lactate (mmol/L)',
    'Creatinine': 'Créatinine (mg/dL)',
    'Bilirubin_total': 'Bilirubine tot. (mg/dL)',
    'Platelets': 'Plaquettes (x10³/µL)',
    'WBC': 'Globules blancs (x10³/µL)',
    'BUN': 'Urée (BUN, mg/dL)',
    'pH': 'pH sanguin',
    'PaCO2': 'PaCO2 (mmHg)',
    'FiO2': 'FiO2 (fraction, ex: 0.21)',
    'Glucose': 'Glucose (mg/dL)',
}

ALL_LABS_FULL = ['EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

THRESHOLD = 0.3854   # seuil (recall ~80%)

COL_BG = "#f4f6f8"
COL_HEADER = "#1e3a5f"
COL_HEADER_TEXT = "#ffffff"
COL_CARD = "#ffffff"
COL_BORDER = "#d7dde3"
COL_ACCENT = "#2c7a4d"
COL_TEXT = "#1f2933"
COL_SUBTEXT = "#5f6b76"
COL_DANGER = "#c0392b"
COL_WARN = "#d68910"
COL_OK = "#1e8449"

# ==========================================
# FENÊTRE PRINCIPALE
# ==========================================
root = tk.Tk()
root.withdraw()

N_HEURES = simpledialog.askinteger(
    "Durée de surveillance",
    "Combien d'heures veux-tu surveiller ce patient ?",
    minvalue=1, maxvalue=48, initialvalue=6
)
if not N_HEURES:
    root.destroy()
    raise SystemExit

root.deiconify()
root.title("Moniteur d'alerte précoce — Sepsis (XGBoost)")
root.configure(bg=COL_BG)
try:
    root.state('zoomed')
except tk.TclError:
    root.attributes('-zoomed', True)

# ---------- En-tête ----------
header = tk.Frame(root, bg=COL_HEADER, height=64)
header.pack(fill='x', side='top')
header.pack_propagate(False)

tk.Label(header, text="⚕", font=('Segoe UI', 20), bg=COL_HEADER, fg=COL_HEADER_TEXT).pack(side='left', padx=(20, 8))
title_box = tk.Frame(header, bg=COL_HEADER)
title_box.pack(side='left', pady=8)
tk.Label(title_box, text="Prédiction précoce du sepsis", font=('Segoe UI', 14, 'bold'),
         bg=COL_HEADER, fg=COL_HEADER_TEXT, anchor='w').pack(anchor='w')
tk.Label(title_box, text="Test manuel — modèle XGBoost (label prospectif 6h, scale_pos_weight)", font=('Segoe UI', 9),
         bg=COL_HEADER, fg="#c8d6e5", anchor='w').pack(anchor='w')

# ---------- Corps scrollable ----------
body_canvas = tk.Canvas(root, bg=COL_BG, highlightthickness=0)
body_scroll = ttk.Scrollbar(root, orient='vertical', command=body_canvas.yview)
body_frame = tk.Frame(body_canvas, bg=COL_BG)

body_frame.bind("<Configure>", lambda e: body_canvas.configure(scrollregion=body_canvas.bbox("all")))
body_canvas.create_window((0, 0), window=body_frame, anchor='nw')
body_canvas.configure(yscrollcommand=body_scroll.set)
body_canvas.pack(side='left', fill='both', expand=True)
body_scroll.pack(side='right', fill='y')

def _on_mousewheel(event):
    body_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
body_canvas.bind_all("<MouseWheel>", _on_mousewheel)

content = tk.Frame(body_frame, bg=COL_BG)
content.pack(fill='both', expand=True, padx=30, pady=24)

def make_card(parent, title):
    card = tk.Frame(parent, bg=COL_CARD, highlightbackground=COL_BORDER, highlightthickness=1)
    card.pack(fill='x', pady=(0, 16))
    inner = tk.Frame(card, bg=COL_CARD)
    inner.pack(fill='both', expand=True, padx=22, pady=16)
    tk.Label(inner, text=title, font=('Segoe UI', 12, 'bold'), bg=COL_CARD, fg=COL_TEXT, anchor='w').pack(anchor='w', pady=(0, 12))
    return inner

def build_table(parent, param_list, labels_dict):
    table = tk.Frame(parent, bg=COL_CARD)
    table.pack(fill='x')
    tk.Label(table, text="Paramètre", width=22, font=('Segoe UI', 9, 'bold'),
             bg=COL_CARD, fg=COL_SUBTEXT, anchor='w').grid(row=0, column=0, sticky='w', pady=(0, 8))
    for h in range(N_HEURES):
        tk.Label(table, text=f"H{h+1}", width=9, font=('Segoe UI', 9, 'bold'),
                 bg=COL_CARD, fg=COL_SUBTEXT).grid(row=0, column=h+1, pady=(0, 8))

    entry_dict = {}
    for i, p in enumerate(param_list):
        tk.Label(table, text=labels_dict[p], width=22, font=('Segoe UI', 10),
                 bg=COL_CARD, fg=COL_TEXT, anchor='w').grid(row=i+1, column=0, sticky='w', pady=4)
        for h in range(N_HEURES):
            e = tk.Entry(table, width=8, justify='center', font=('Segoe UI', 10),
                         relief='solid', bd=1, highlightthickness=1,
                         highlightbackground=COL_BORDER, highlightcolor=COL_ACCENT)
            e.grid(row=i+1, column=h+1, padx=4, pady=4, ipady=4)
            entry_dict[(h, p)] = e
    return entry_dict

# ---------- Carte : Informations patient ----------
card_infos = make_card(content, "Informations patient")
infos_row = tk.Frame(card_infos, bg=COL_CARD)
infos_row.pack(fill='x')

def labeled_entry(parent, label, default, col):
    box = tk.Frame(parent, bg=COL_CARD)
    box.grid(row=0, column=col, padx=(0, 30), sticky='w')
    tk.Label(box, text=label, font=('Segoe UI', 9), bg=COL_CARD, fg=COL_SUBTEXT, anchor='w').pack(anchor='w')
    e = tk.Entry(box, width=14, font=('Segoe UI', 11), relief='solid', bd=1,
                 highlightthickness=1, highlightbackground=COL_BORDER, highlightcolor=COL_ACCENT)
    e.insert(0, default)
    e.pack(ipady=4, pady=(3, 0))
    return e

entry_age = labeled_entry(infos_row, "Âge", "60", 0)
entry_gender = labeled_entry(infos_row, "Sexe (0=F, 1=H)", "1", 1)
entry_hosp = labeled_entry(infos_row, "HospAdmTime (h)", "-10", 2)

# ---------- Carte : Signes vitaux (une valeur par heure) ----------
card_vitals = make_card(content, f"Paramètres physiologiques — {N_HEURES} heures")
vital_entries = build_table(card_vitals, VITALS, VITALS_LABELS)

# ==========================================
# CARTE : Résultats biologiques (une seule valeur par labo + heure de prélèvement)
# ==========================================
card_labs = make_card(content, "Résultats biologiques (dernière valeur connue, laisser vide si non disponible)")

labs_table = tk.Frame(card_labs, bg=COL_CARD)
labs_table.pack(fill='x')

tk.Label(labs_table, text="Paramètre", width=22, font=('Segoe UI', 9, 'bold'),
         bg=COL_CARD, fg=COL_SUBTEXT, anchor='w').grid(row=0, column=0, sticky='w', pady=(0, 8))
tk.Label(labs_table, text="Valeur", width=10, font=('Segoe UI', 9, 'bold'),
         bg=COL_CARD, fg=COL_SUBTEXT).grid(row=0, column=1, pady=(0, 8))
tk.Label(labs_table, text="Prélevé à l'heure", width=16, font=('Segoe UI', 9, 'bold'),
         bg=COL_CARD, fg=COL_SUBTEXT).grid(row=0, column=2, pady=(0, 8))

lab_value_entries = {}
lab_hour_entries = {}
for i, lab in enumerate(LABS):
    tk.Label(labs_table, text=LABS_LABELS[lab], width=22, font=('Segoe UI', 10),
             bg=COL_CARD, fg=COL_TEXT, anchor='w').grid(row=i+1, column=0, sticky='w', pady=4)

    e_val = tk.Entry(labs_table, width=10, justify='center', font=('Segoe UI', 10),
                     relief='solid', bd=1, highlightthickness=1,
                     highlightbackground=COL_BORDER, highlightcolor=COL_ACCENT)
    e_val.grid(row=i+1, column=1, padx=4, pady=4, ipady=4)
    lab_value_entries[lab] = e_val

    e_hour = tk.Spinbox(labs_table, from_=1, to=N_HEURES, width=6, justify='center',
                        font=('Segoe UI', 10), relief='solid', bd=1)
    e_hour.delete(0, 'end')
    e_hour.insert(0, "1")
    e_hour.grid(row=i+1, column=2, padx=4, pady=4, ipady=2)
    lab_hour_entries[lab] = e_hour

# ---------- Bouton d'action ----------
btn_frame = tk.Frame(content, bg=COL_BG)
btn_frame.pack(fill='x', pady=(4, 16))

btn = tk.Button(btn_frame, text="Calculer la probabilité de sepsis", command=lambda: calculer(),
                bg=COL_ACCENT, fg="white", font=('Segoe UI', 11, 'bold'),
                relief='flat', padx=18, pady=10, cursor='hand2',
                activebackground="#256b43", activeforeground="white")
btn.pack(anchor='w')

# ---------- Carte : Résultats ----------
card_result_holder = tk.Frame(content, bg=COL_BG)

# ==========================================
# ESTIMATION DU DÉLAI DE SURVENUE (tendance)
# ==========================================
def estimer_delai_survenue(proba, seuil_alerte=THRESHOLD, seuil_critique=0.6):
    heures = np.arange(1, len(proba) + 1)
    proba = np.array(proba)

    heures_alerte = heures[proba >= seuil_alerte]
    premiere_alerte = int(heures_alerte[0]) if len(heures_alerte) > 0 else None

    if len(proba) >= 2:
        pente, ordonnee = np.polyfit(heures, proba, 1)
    else:
        pente, ordonnee = 0.0, proba[-1] if len(proba) else 0.0

    derniere_proba = proba[-1]

    resultat = {
        'premiere_alerte_heure': premiere_alerte,
        'pente': pente,
        'derniere_proba': derniere_proba,
        'heures_jusqua_critique': None,
        'tendance': None
    }

    if pente > 0.001:
        resultat['tendance'] = 'aggravation'
        delta = (seuil_critique - derniere_proba) / pente
        if delta > 0:
            resultat['heures_jusqua_critique'] = round(delta, 1)
    elif pente < -0.001:
        resultat['tendance'] = 'amelioration'
    else:
        resultat['tendance'] = 'stable'

    return resultat

def afficher_resultats(proba, sofa_scores):
    for w in card_result_holder.winfo_children():
        w.destroy()
    inner = make_card(card_result_holder, "Résultat de la prédiction")

    for h in range(N_HEURES):
        p = proba[h]
        row = tk.Frame(inner, bg=COL_CARD)
        row.pack(fill='x', pady=3)

        tk.Label(row, text=f"Heure {h+1}", width=10, font=('Segoe UI', 10),
                 bg=COL_CARD, fg=COL_TEXT, anchor='w').pack(side='left')

        bar_bg = tk.Frame(row, bg="#e7ebee", width=380, height=16)
        bar_bg.pack(side='left', padx=(8, 12))
        bar_bg.pack_propagate(False)
        fill_w = max(2, int(380 * min(p, 1.0)))
        color = COL_DANGER if p >= THRESHOLD else (COL_WARN if p >= THRESHOLD * 0.6 else COL_OK)
        tk.Frame(bar_bg, bg=color, width=fill_w, height=16).place(x=0, y=0)

        tk.Label(row, text=f"{p*100:5.1f}%", width=8, font=('Segoe UI', 10, 'bold'),
                 bg=COL_CARD, fg=color).pack(side='left')

        statut = "ALERTE" if p >= THRESHOLD else ("Surveillance" if p >= THRESHOLD * 0.6 else "Normal")
        tk.Label(row, text=statut, font=('Segoe UI', 9, 'bold'),
                 bg=COL_CARD, fg=color).pack(side='left', padx=(6, 0))

        tk.Label(row, text=f"SOFA: {sofa_scores[h]}", font=('Segoe UI', 9),
                 bg=COL_CARD, fg=COL_SUBTEXT).pack(side='left', padx=(14, 0))

    alerte_heures = [h + 1 for h in range(N_HEURES) if proba[h] >= THRESHOLD]
    summary = tk.Label(inner, font=('Segoe UI', 11, 'bold'), bg=COL_CARD, anchor='w')
    if alerte_heures:
        summary.config(text=f"⚠  Alerte sepsis déclenchée dès l'heure {alerte_heures[0]} (seuil {THRESHOLD})", fg=COL_DANGER)
    else:
        summary.config(text=f"✓  Aucune alerte déclenchée (seuil {THRESHOLD})", fg=COL_OK)
    summary.pack(anchor='w', pady=(12, 0))

    # ---------- Estimation textuelle du délai de survenue ----------
    estimation = estimer_delai_survenue(proba)
    est_label = tk.Label(inner, font=('Segoe UI', 10), bg=COL_CARD, anchor='w', justify='left')

    if estimation['premiere_alerte_heure']:
        texte = f"📍 Première alerte détectée à l'heure {estimation['premiere_alerte_heure']}.\n"
    else:
        texte = "📍 Aucune alerte sur la période saisie.\n"

    if estimation['tendance'] == 'aggravation':
        texte += f"📈 Tendance : aggravation (pente +{estimation['pente']*100:.1f} pts/h)."
        if estimation['heures_jusqua_critique']:
            texte += f"\n⏱  Estimation : risque critique (≥60%) atteint dans environ {estimation['heures_jusqua_critique']} heure(s) si la tendance se poursuit."
    elif estimation['tendance'] == 'amelioration':
        texte += f"📉 Tendance : amélioration (pente {estimation['pente']*100:.1f} pts/h)."
    else:
        texte += "➡  Tendance : stable."

    texte += "\n\n⚠ Estimation indicative basée sur l'extrapolation de la tendance — ne remplace pas une évaluation clinique."

    est_label.config(text=texte, fg=COL_SUBTEXT)
    est_label.pack(anchor='w', pady=(14, 0))

    # ==========================================
    # GRAPHIQUE : allure du sepsis (probabilité + SOFA dans le temps)
    # ==========================================
    graph_card = make_card(card_result_holder, "Allure du sepsis dans le temps")

    fig = Figure(figsize=(7.5, 3.2), dpi=100)
    fig.patch.set_facecolor(COL_CARD)

    ax1 = fig.add_subplot(111)
    ax1.set_facecolor(COL_CARD)

    heures = list(range(1, N_HEURES + 1))
    proba_pct = [p * 100 for p in proba]

    ax1.plot(heures, proba_pct, color=COL_DANGER, marker='o', linewidth=2, label='Probabilité de sepsis (%)')
    ax1.axhline(y=THRESHOLD * 100, color=COL_WARN, linestyle='--', linewidth=1, label=f"Seuil d'alerte ({THRESHOLD*100:.1f}%)")
    ax1.set_xlabel('Heure')
    ax1.set_ylabel('Probabilité (%)', color=COL_DANGER)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='y', labelcolor=COL_DANGER)
    ax1.set_xticks(heures)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(heures, sofa_scores, color=COL_HEADER, marker='s', linewidth=1.5, linestyle=':', label='SOFA_score')
    ax2.set_ylabel('SOFA_score', color=COL_HEADER)
    ax2.tick_params(axis='y', labelcolor=COL_HEADER)
    ax2.set_ylim(0, max(max(sofa_scores) + 2, 10))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, framealpha=0.9)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=graph_card)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    card_result_holder.pack(fill='x')

# ==========================================
# CALCUL DU SOFA_SCORE (simplifié, sans vasopresseurs ni Glasgow)
# ==========================================
def sofa_respiratoire(sao2, fio2):
    if pd.isna(sao2) or pd.isna(fio2) or fio2 == 0:
        return 0
    ratio = (sao2 / fio2)
    if ratio >= 400: return 0
    if ratio >= 300: return 1
    if ratio >= 200: return 2
    if ratio >= 100: return 3
    return 4

def sofa_coagulation(platelets):
    if pd.isna(platelets): return 0
    if platelets >= 150: return 0
    if platelets >= 100: return 1
    if platelets >= 50: return 2
    if platelets >= 20: return 3
    return 4

def sofa_hepatique(bilirubin):
    if pd.isna(bilirubin): return 0
    if bilirubin < 1.2: return 0
    if bilirubin < 2.0: return 1
    if bilirubin < 6.0: return 2
    if bilirubin < 12.0: return 3
    return 4

def sofa_cardiovasculaire(map_val):
    if pd.isna(map_val): return 0
    return 0 if map_val >= 70 else 1

def sofa_renal(creatinine):
    if pd.isna(creatinine): return 0
    if creatinine < 1.2: return 0
    if creatinine < 2.0: return 1
    if creatinine < 3.5: return 2
    if creatinine < 5.0: return 3
    return 4

def calculer_sofa_row(row):
    return (
        sofa_respiratoire(row.get('O2Sat'), row.get('FiO2')) +
        sofa_coagulation(row.get('Platelets')) +
        sofa_hepatique(row.get('Bilirubin_total')) +
        sofa_cardiovasculaire(row.get('MAP')) +
        sofa_renal(row.get('Creatinine'))
    )

def calculer():
    try:
        rows = []
        for h in range(N_HEURES):
            row = {}
            for v in VITALS:
                val = vital_entries[(h, v)].get().strip()
                row[v] = float(val) if val != "" else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)

        for lab in LABS:
            df[lab] = np.nan
        for lab in LABS:
            val_str = lab_value_entries[lab].get().strip()
            if val_str != "":
                h_saisie = int(lab_hour_entries[lab].get()) - 1
                h_saisie = min(max(h_saisie, 0), N_HEURES - 1)
                df.loc[h_saisie, lab] = float(val_str)
        df[LABS] = df[LABS].ffill()

        df['ICULOS'] = range(1, N_HEURES + 1)
        df['Age'] = float(entry_age.get())
        df['Gender'] = int(entry_gender.get())
        df['HospAdmTime'] = float(entry_hosp.get())
        df['Unit1'] = 0
        df['Unit2'] = 1

        for v in VITALS:
            df[f'{v}_delta'] = df[v].diff()
            df[f'{v}_rollmean3h'] = df[v].rolling(3, min_periods=1).mean()
            df[f'{v}_rollmean6h'] = df[v].rolling(6, min_periods=1).mean()
            df[f'{v}_rollstd6h'] = df[v].rolling(6, min_periods=1).std()
            df[f'{v}_pente3h'] = df[v].diff(3) / 3
            df[f'{v}_ecart_baseline'] = df[v] - df[v].iloc[0]

        df['choc_index'] = df['HR'] / df['SBP']
        df['HR_MAP_ratio'] = df['HR'] / df['MAP']

        for lab in ALL_LABS_FULL:
            if lab not in df.columns:
                df[lab] = np.nan
            df[f'{lab}_mesure'] = df[lab].notna().astype(int)

        df['SIRS_score'] = (
            ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(int) +
            (df['HR'] > 90).astype(int) +
            (df['Resp'] > 20).astype(int)
        )
        df['SIRS_positif'] = (df['SIRS_score'] >= 2).astype(int)
        df['qSOFA_score'] = (
            (df['Resp'] >= 22).astype(int) +
            (df['SBP'] <= 100).astype(int)
        )
        df['qSOFA_positif'] = (df['qSOFA_score'] >= 2).astype(int)

        sofa_scores = df.apply(calculer_sofa_row, axis=1)
        df['SOFA_score'] = sofa_scores
        df['SOFA_delta'] = df['SOFA_score'].diff()
        df['SOFA_delta6h'] = df['SOFA_score'].diff(6)
        df['NEWS_score'] = 0
        df['PatientID'] = 'test_manuel'

        sig_cols = [c for c in feature_cols if c.startswith('sig_')]
        sig_data = {c: 0.0 for c in sig_cols}
        df = pd.concat([df, pd.DataFrame([sig_data] * len(df))], axis=1)

        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        X = df[feature_cols].copy()

        proba = model.predict_proba(X)[:, 1]
        afficher_resultats(proba, sofa_scores.tolist())

    except Exception as e:
        messagebox.showerror("Erreur", str(e))

root.mainloop()