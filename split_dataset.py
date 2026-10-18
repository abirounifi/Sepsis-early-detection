import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_FILE = "./processed_data/dataset_raw.csv"
OUTPUT_DIR = "./processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# 2. CHARGEMENT
# ==========================================

print("📂 Chargement du dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"📐 {len(df):,} lignes")
print(f"👥 {df['PatientID'].nunique():,} patients uniques")

# ==========================================
# 3. LABEL AU NIVEAU PATIENT
# ==========================================
# Un patient est considéré comme positif s'il
# possède au moins une ligne SepsisLabel = 1.

patient_labels = df.groupby("PatientID")["SepsisLabel"].max()

print("\n📊 Patients selon leur statut :")
print(patient_labels.value_counts())
print("\n📊 Pourcentage de patients :")
print(patient_labels.value_counts(normalize=True) * 100)

# ==========================================
# 4. PREMIER SPLIT : 80% / 20%
# ==========================================

patients_train_val, patients_test = train_test_split(
    patient_labels.index.to_numpy(),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=patient_labels.values
)

labels_train_val = patient_labels.loc[patients_train_val]

# ==========================================
# 5. DEUXIÈME SPLIT : TRAIN / VALIDATION
# ==========================================

patients_train, patients_val = train_test_split(
    np.array(patients_train_val),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels_train_val.values
)

# ==========================================
# 6. CONSTRUCTION DES DATAFRAMES
# ==========================================

df_train = df[df["PatientID"].isin(patients_train)].reset_index(drop=True)
df_val = df[df["PatientID"].isin(patients_val)].reset_index(drop=True)
df_test = df[df["PatientID"].isin(patients_test)].reset_index(drop=True)

# ==========================================
# 7. AFFICHAGE DES RÉSULTATS
# ==========================================

print("\n" + "=" * 60)
print("✅ SPLIT PAR PATIENT TERMINÉ")
print("=" * 60)

print(f"\nTRAIN : {len(df_train):,} lignes | {df_train['PatientID'].nunique():,} patients")
print(f"VAL   : {len(df_val):,} lignes | {df_val['PatientID'].nunique():,} patients")
print(f"TEST  : {len(df_test):,} lignes | {df_test['PatientID'].nunique():,} patients")

# ==========================================
# 8. RÉPARTITION DES LABELS
# ==========================================

print("\n📊 LIGNES - TRAIN")
print(df_train["SepsisLabel"].value_counts())
print(df_train["SepsisLabel"].value_counts(normalize=True).mul(100).round(3))

print("\n📊 LIGNES - VALIDATION")
print(df_val["SepsisLabel"].value_counts())
print(df_val["SepsisLabel"].value_counts(normalize=True).mul(100).round(3))

print("\n📊 LIGNES - TEST")
print(df_test["SepsisLabel"].value_counts())
print(df_test["SepsisLabel"].value_counts(normalize=True).mul(100).round(3))

# ==========================================
# 9. VÉRIFICATION : AUCUN PATIENT EN COMMUN
# ==========================================

train_patients = set(df_train["PatientID"])
val_patients = set(df_val["PatientID"])
test_patients = set(df_test["PatientID"])

print("\n🔍 Vérification du leakage patient...")
print("TRAIN ∩ VAL  :", len(train_patients & val_patients))
print("TRAIN ∩ TEST :", len(train_patients & test_patients))
print("VAL ∩ TEST   :", len(val_patients & test_patients))

assert len(train_patients & val_patients) == 0
assert len(train_patients & test_patients) == 0
assert len(val_patients & test_patients) == 0

print("\n✅ Aucun patient commun entre TRAIN / VAL / TEST.")

# ==========================================
# 10. SAUVEGARDE
# ==========================================

train_file = os.path.join(OUTPUT_DIR, "train_set.csv")
val_file = os.path.join(OUTPUT_DIR, "validation_set.csv")
test_file = os.path.join(OUTPUT_DIR, "test_set.csv")

df_train.to_csv(train_file, index=False)
df_val.to_csv(val_file, index=False)
df_test.to_csv(test_file, index=False)

print("\n💾 Fichiers sauvegardés :")
print(f"   → {train_file}")
print(f"   → {val_file}")
print(f"   → {test_file}")