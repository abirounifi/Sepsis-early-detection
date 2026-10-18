import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report
)
import joblib

# ==========================================
# 1. CHARGEMENT DES DONNÉES DÉJÀ SPLITTÉES
# ==========================================
print("📂 Chargement des sets...")

train = pd.read_csv("./processed_data/train_set.csv")
val = pd.read_csv("./processed_data/validation_set.csv")
test = pd.read_csv("./processed_data/test_set.csv")

print(f"TRAIN: {len(train):,} lignes | VAL: {len(val):,} lignes | TEST: {len(test):,} lignes")

# ==========================================
# 2. SÉPARATION FEATURES / LABEL
# ==========================================
COLS_A_EXCLURE = ['SepsisLabel', 'PatientID']

X_train = train.drop(columns=COLS_A_EXCLURE)
y_train = train['SepsisLabel']

X_val = val.drop(columns=COLS_A_EXCLURE)
y_val = val['SepsisLabel']

X_test = test.drop(columns=COLS_A_EXCLURE)
y_test = test['SepsisLabel']

print(f"\n📊 {X_train.shape[1]} features utilisées")

# ==========================================
# 3. CALCUL DU scale_pos_weight
# ==========================================
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print(f"\n⚖️  scale_pos_weight = {n_neg:,} / {n_pos:,} = {scale_pos_weight:.2f}")

# ==========================================
# 4. ENTRAÎNEMENT XGBOOST
# ==========================================
print("\n🚀 Entraînement du modèle XGBoost...")

model = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',          # optimise directement l'AUPRC
    early_stopping_rounds=50,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# ==========================================
# 5. ÉVALUATION SUR VALIDATION (AUROC / AUPRC)
# ==========================================
proba_val = model.predict_proba(X_val)[:, 1]

auroc_val = roc_auc_score(y_val, proba_val)
auprc_val = average_precision_score(y_val, proba_val)

print(f"\n📈 VALIDATION — AUROC: {auroc_val:.4f} | AUPRC: {auprc_val:.4f}")

# ==========================================
# 6. CHOIX DU SEUIL (cible recall ~80%)
# ==========================================
prec, rec, seuils = precision_recall_curve(y_val, proba_val)

RECALL_CIBLE = 0.80
idx = np.argmin(np.abs(rec - RECALL_CIBLE))
seuil_choisi = seuils[max(idx - 1, 0)] if idx < len(seuils) else 0.5

print(f"\n🎯 Seuil choisi pour recall ≈ {RECALL_CIBLE}: {seuil_choisi:.4f}")
print(f"   → Precision correspondante: {prec[idx]:.4f}")
print(f"   → Recall correspondant: {rec[idx]:.4f}")

# ==========================================
# 7. ÉVALUATION FINALE SUR TEST (jamais touché avant)
# ==========================================
proba_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (proba_test >= seuil_choisi).astype(int)

auroc_test = roc_auc_score(y_test, proba_test)
auprc_test = average_precision_score(y_test, proba_test)

print(f"\n" + "=" * 60)
print("✅ RÉSULTATS FINAUX SUR TEST")
print("=" * 60)
print(f"AUROC: {auroc_test:.4f} | AUPRC: {auprc_test:.4f}")

print("\n📊 Matrice de confusion (lignes) :")
print(confusion_matrix(y_test, y_pred_test))

print("\n📊 Rapport de classification :")
print(classification_report(y_test, y_pred_test, digits=4))

# ==========================================
# 8. ÉVALUATION PAR PATIENT (métrique clinique réelle)
# ==========================================
test_eval = test.copy()
test_eval['proba'] = proba_test
test_eval['alerte'] = y_pred_test

patient_resultats = test_eval.groupby('PatientID').agg(
    vrai_label=('SepsisLabel', 'max'),
    a_ete_alerte=('alerte', 'max')
)

patients_septiques = patient_resultats[patient_resultats['vrai_label'] == 1]
patients_detectes = patients_septiques[patients_septiques['a_ete_alerte'] == 1]

print(f"\n🏥 ÉVALUATION PAR PATIENT")
print(f"Patients septiques dans le test : {len(patients_septiques)}")
print(f"Patients septiques détectés (≥1 alerte) : {len(patients_detectes)}")
print(f"Taux de détection par patient : {len(patients_detectes)/len(patients_septiques)*100:.1f}%")

patients_non_septiques = patient_resultats[patient_resultats['vrai_label'] == 0]
faux_positifs_patients = patients_non_septiques[patients_non_septiques['a_ete_alerte'] == 1]
print(f"Faux positifs par patient : {len(faux_positifs_patients)}/{len(patients_non_septiques)} ({len(faux_positifs_patients)/len(patients_non_septiques)*100:.1f}%)")

# ==========================================
# 9. FEATURE IMPORTANCE
# ==========================================
importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"\n🔝 Top 15 features les plus importantes :")
print(importances.head(15))

# ==========================================
# 10. SAUVEGARDE DU MODÈLE
# ==========================================
joblib.dump(model, "./processed_data/xgboost_sepsis_model.pkl")
print(f"\n💾 Modèle sauvegardé : ./processed_data/xgboost_sepsis_model.pkl")
print(f"💾 Seuil retenu à réutiliser : {seuil_choisi:.4f}")