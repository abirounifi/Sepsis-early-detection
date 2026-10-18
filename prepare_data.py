import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = r"C:\Users\abir2\OneDrive\Desktop\sepsis early detection 1\sepsis_dataset"
OUTPUT_DIR = "./processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VITALS_CONTINUS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']

LABO_COLS = [
    'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT',
    'WBC', 'Fibrinogen', 'Platelets'
]

# ==========================================
# 2. RECHERCHE DES FICHIERS
# ==========================================
print("🔍 Recherche des fichiers .psv dans :", DATA_DIR)

psv_files = glob.glob(os.path.join(DATA_DIR, "*.psv"))
if not psv_files:
    psv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.psv"), recursive=True)

print(f"📊 {len(psv_files)} fichiers patients trouvés.")

if len(psv_files) == 0:
    print("❌ Aucun fichier trouvé ! Vérifie le chemin du dossier.")
    exit()

# ==========================================
# 3. FONCTION : calcul des scores cliniques NaN-aware pour UN patient
# ==========================================
def calculer_scores_cliniques(df):
    # --- SF ratio : FiO2=0/manquant -> air ambiant (0.21) UNIQUEMENT pour ce calcul ---
    fio2_pour_sf = df['FiO2'].replace(0, np.nan).fillna(0.21)
    sf_ratio = df['SaO2'] / fio2_pour_sf

    # --- SIRS (NaN-aware : un composant manquant reste NaN, pas 0) ---
    sirs_temp = pd.Series(np.where(df['Temp'].isna(), np.nan,
                 ((df['Temp'] > 38) | (df['Temp'] < 36)).astype(float)), index=df.index)
    sirs_hr = pd.Series(np.where(df['HR'].isna(), np.nan,
                 (df['HR'] > 90).astype(float)), index=df.index)
    sirs_resp = pd.Series(np.where(df['Resp'].isna() & df['PaCO2'].isna(), np.nan,
                 ((df['Resp'] > 20) | (df['PaCO2'] < 32)).astype(float)), index=df.index)
    sirs_wbc = pd.Series(np.where(df['WBC'].isna(), np.nan,
                 ((df['WBC'] > 12) | (df['WBC'] < 4)).astype(float)), index=df.index)

    comp_sirs = pd.DataFrame({'temp': sirs_temp, 'hr': sirs_hr, 'resp': sirs_resp, 'wbc': sirs_wbc})
    df['SIRS_score'] = comp_sirs.sum(axis=1, min_count=1)
    df['SIRS_complet'] = comp_sirs.notna().sum(axis=1)
    df['SIRS_positif'] = (df['SIRS_score'] >= 2).astype(int)

    # --- qSOFA ---
    qsofa_resp = pd.Series(np.where(df['Resp'].isna(), np.nan,
                  (df['Resp'] >= 22).astype(float)), index=df.index)
    qsofa_sbp = pd.Series(np.where(df['SBP'].isna(), np.nan,
                 (df['SBP'] <= 100).astype(float)), index=df.index)
    comp_qsofa = pd.DataFrame({'resp': qsofa_resp, 'sbp': qsofa_sbp})
    df['qSOFA_score'] = comp_qsofa.sum(axis=1, min_count=1)
    df['qSOFA_positif'] = (df['qSOFA_score'] >= 2).astype(int)

    # --- SOFA (chaque composant : NaN si donnée manquante, sinon score par seuils) ---
    def seuils_sofa(serie, bornes, scores):
        # bornes croissantes, scores correspondants ; NaN si serie manquante
        out = pd.Series(0.0, index=serie.index)
        for b, s in zip(bornes, scores):
            out = np.where(serie >= b, s, out)
        out = pd.Series(out, index=serie.index)
        out[serie.isna()] = np.nan
        return out

    sofa_resp = seuils_sofa(-sf_ratio, [-400, -300, -200, -100], [1, 2, 3, 4]) if False else None
    # (sf_ratio décroît avec la gravité -> on seuille directement)
    sofa_resp = pd.Series(np.select(
        [sf_ratio < 100, sf_ratio < 200, sf_ratio < 300, sf_ratio < 400],
        [4, 3, 2, 1], default=0), index=df.index).astype(float)
    sofa_resp[sf_ratio.isna()] = np.nan

    sofa_coag = pd.Series(np.select(
        [df['Platelets'] < 20, df['Platelets'] < 50, df['Platelets'] < 100, df['Platelets'] < 150],
        [4, 3, 2, 1], default=0), index=df.index).astype(float)
    sofa_coag[df['Platelets'].isna()] = np.nan

    sofa_liver = pd.Series(np.select(
        [df['Bilirubin_total'] >= 12, df['Bilirubin_total'] >= 6,
         df['Bilirubin_total'] >= 2, df['Bilirubin_total'] >= 1.2],
        [4, 3, 2, 1], default=0), index=df.index).astype(float)
    sofa_liver[df['Bilirubin_total'].isna()] = np.nan

    sofa_cardio = pd.Series(np.where(df['MAP'] < 70, 1.0, 0.0), index=df.index)
    sofa_cardio[df['MAP'].isna()] = np.nan

    sofa_renal = pd.Series(np.select(
        [df['Creatinine'] >= 5, df['Creatinine'] >= 3.5,
         df['Creatinine'] >= 2, df['Creatinine'] >= 1.2],
        [4, 3, 2, 1], default=0), index=df.index).astype(float)
    sofa_renal[df['Creatinine'].isna()] = np.nan

    comp_sofa = pd.DataFrame({
        'resp': sofa_resp, 'coag': sofa_coag, 'liver': sofa_liver,
        'cardio': sofa_cardio, 'renal': sofa_renal
    })
    df['SOFA_score'] = comp_sofa.sum(axis=1, min_count=1)
    df['SOFA_complet'] = comp_sofa.notna().sum(axis=1)

    # --- ΔSOFA (critère Sepsis-3 : ΔSOFA >= 2, pas juste SOFA élevé) ---
    df['SOFA_delta'] = df['SOFA_score'].diff()
    df['SOFA_delta6h'] = df['SOFA_score'].diff(6)

    # --- NEWS (inchangé, moins central à votre problème) ---
    news_resp = np.select(
        [df['Resp'] <= 8, df['Resp'] <= 11, df['Resp'] <= 20, df['Resp'] <= 24],
        [3, 1, 0, 2], default=3)
    news_o2sat = np.select(
        [df['O2Sat'] <= 91, df['O2Sat'] <= 93, df['O2Sat'] <= 95], [3, 2, 1], default=0)
    news_temp = np.select(
        [df['Temp'] <= 35.0, df['Temp'] <= 36.0, df['Temp'] <= 38.0, df['Temp'] <= 39.0],
        [3, 1, 0, 1], default=2)
    news_sbp = np.select(
        [df['SBP'] <= 90, df['SBP'] <= 100, df['SBP'] <= 110, df['SBP'] <= 219],
        [3, 2, 1, 0], default=3)
    news_hr = np.select(
        [df['HR'] <= 40, df['HR'] <= 50, df['HR'] <= 90, df['HR'] <= 110, df['HR'] <= 130],
        [3, 1, 0, 1, 2], default=3)
    df['NEWS_score'] = news_resp + news_o2sat + news_temp + news_sbp + news_hr

    return df

# ==========================================
# 4. ASSEMBLAGE, SANS IMPUTATION DES NaN
# ==========================================
toutes_les_lignes = []
fichiers_ignores = 0

print("\n⚙️ Assemblage et enrichissement, sans imputation des valeurs manquantes...")

for file_path in tqdm(psv_files):
    patient_id = os.path.basename(file_path).replace(".psv", "")

    try:
        df = pd.read_csv(file_path, sep='|')
    except Exception:
        fichiers_ignores += 1
        continue

    if df.empty:
        fichiers_ignores += 1
        continue

    df.columns = df.columns.str.strip()

    if df.shape[1] < 35 or 'HR' not in df.columns:
        fichiers_ignores += 1
        continue

    # --- Deltas pour les vitaux continus ---
    for col in VITALS_CONTINUS:
        if col in df.columns:
            df[f'{col}_delta'] = df[col].diff()

    # --- Moyennes glissantes 3h et 6h ---
    for col in VITALS_CONTINUS:
        if col in df.columns:
            df[f'{col}_rollmean3h'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_rollmean6h'] = df[col].rolling(window=6, min_periods=1).mean()

    # --- Pentes, écarts baseline, variabilité ---
    for col in VITALS_CONTINUS:
        if col in df.columns:
            df[f'{col}_pente3h'] = (df[col] - df[col].shift(3)) / 3
            baseline = df[col].iloc[:3].mean()
            df[f'{col}_ecart_baseline'] = df[col] - baseline
            df[f'{col}_rollstd6h'] = df[col].rolling(window=6, min_periods=1).std()

    # --- Interactions cliniques ---
    if 'HR' in df.columns and 'SBP' in df.columns:
        df['choc_index'] = df['HR'] / df['SBP'].replace(0, np.nan)
    if 'HR' in df.columns and 'MAP' in df.columns:
        df['HR_MAP_ratio'] = df['HR'] / df['MAP'].replace(0, np.nan)

    # --- Indicateurs mesuré / non mesuré pour les labos ---
    for col in LABO_COLS:
        if col in df.columns:
            df[f'{col}_mesure'] = df[col].notna().astype(int)

    # --- Scores cliniques NaN-aware + delta SOFA, calculés PAR PATIENT ici (pas de groupby global) ---
    df = calculer_scores_cliniques(df)

    df['PatientID'] = patient_id

    if 'SepsisLabel' not in df.columns:
        df['SepsisLabel'] = 0

    toutes_les_lignes.append(df)

# ==========================================
# 5. ASSEMBLAGE FINAL
# ==========================================
df_final = pd.concat(toutes_les_lignes, ignore_index=True)

print(f"\n✅ Terminé.")
if fichiers_ignores > 0:
    print(f"⚠️  {fichiers_ignores} fichiers ignorés (vides ou illisibles)")
print(f"📐 {len(df_final)} lignes, {df_final['PatientID'].nunique()} patients uniques")
print("✅ Scores ajoutés : SIRS_score, SIRS_complet, qSOFA_score, SOFA_score, SOFA_complet, SOFA_delta, SOFA_delta6h, NEWS_score")
print(f"\n📊 Répartition des classes : \n{df_final['SepsisLabel'].value_counts()}")

# ==========================================
# 6. SAUVEGARDE
# ==========================================
df_final.to_csv(os.path.join(OUTPUT_DIR, "dataset_raw.csv"), index=False)
print(f"\n💾 Sauvegardé dans {OUTPUT_DIR}/dataset_raw.csv")