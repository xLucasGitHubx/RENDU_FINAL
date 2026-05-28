"""Entraîne le meilleur modèle et sauvegarde tout ce qu'il faut pour l'app Streamlit.

Lancer une fois avant la première utilisation de l'app :
    python train_model.py
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "data" / "Cleaned-Data.csv"
ARTIFACTS_PATH = Path(__file__).parent / "artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)


def main():
    print(f"Chargement {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape : {df.shape}")

    # Multi-hot encoding des symptômes
    mlb_sym = MultiLabelBinarizer()
    mlb_exp = MultiLabelBinarizer()

    sym_lists = df["Symptoms"].apply(lambda s: [t.strip() for t in s.split(",")])
    exp_lists = df["Experiencing_Symptoms"].apply(lambda s: [t.strip() for t in s.split(",")])

    sym_encoded = pd.DataFrame(
        mlb_sym.fit_transform(sym_lists),
        columns=[f"sym_{c}" for c in mlb_sym.classes_],
        index=df.index,
    )
    exp_encoded = pd.DataFrame(
        mlb_exp.fit_transform(exp_lists),
        columns=[f"exp_{c}" for c in mlb_exp.classes_],
        index=df.index,
    )

    # One-hot des variables simples
    cat_cols = ["Country", "Age", "Gender", "Severity", "Contact"]
    df_ohe = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False)

    X = pd.concat([df_ohe, sym_encoded, exp_encoded], axis=1).astype(np.float32)
    y = df["COVID_19"].values
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Entraînement Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy test : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 test       : {f1_score(y_test, y_pred):.4f}")

    # Vocabulaires pour le frontend
    vocab = {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols}

    artifacts = {
        "model": model,
        "mlb_sym": mlb_sym,
        "mlb_exp": mlb_exp,
        "feature_names": feature_names,
        "vocab": vocab,
        "symptom_tokens": list(mlb_sym.classes_),
        "experiencing_tokens": list(mlb_exp.classes_),
    }

    out_file = ARTIFACTS_PATH / "chatbot_artifacts.joblib"
    joblib.dump(artifacts, out_file, compress=3)
    print(f"Artefacts sauvegardés : {out_file}")


if __name__ == "__main__":
    main()
