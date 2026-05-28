"""Chatbot COVID-19 - Front-end Streamlit.

Lancer :
    streamlit run app.py

Pré-requis :
    python train_model.py  (génère artifacts/chatbot_artifacts.joblib)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_FILE = Path(__file__).parent / "artifacts" / "chatbot_artifacts.joblib"

st.set_page_config(
    page_title="Chatbot COVID-19",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    if not ARTIFACTS_FILE.exists():
        st.error(
            f"Fichier introuvable : {ARTIFACTS_FILE}\n\n"
            "Lance d'abord : `python train_model.py`"
        )
        st.stop()
    return joblib.load(ARTIFACTS_FILE)


def encode_answers(answers, mlb_sym, mlb_exp, feature_names):
    temp = pd.DataFrame([{
        "Country": answers["Country"],
        "Age": answers["Age"],
        "Gender": answers["Gender"],
        "Severity": answers["Severity"],
        "Contact": answers["Contact"],
    }])
    ohe = pd.get_dummies(temp, prefix=temp.columns.tolist())

    sym_vec = mlb_sym.transform([answers["Symptoms"]])
    exp_vec = mlb_exp.transform([answers["Experiencing_Symptoms"]])
    sym_df = pd.DataFrame(sym_vec, columns=[f"sym_{c}" for c in mlb_sym.classes_])
    exp_df = pd.DataFrame(exp_vec, columns=[f"exp_{c}" for c in mlb_exp.classes_])

    row = pd.concat(
        [ohe.reset_index(drop=True), sym_df.reset_index(drop=True), exp_df.reset_index(drop=True)],
        axis=1,
    )
    row = row.reindex(columns=feature_names, fill_value=0).astype(np.float32)
    return row


def build_questions(art):
    vocab = art["vocab"]
    return [
        {"key": "Country", "label": "Pays visité récemment ?",
         "options": vocab["Country"], "type": "single"},
        {"key": "Age", "label": "Groupe d'âge ?",
         "options": vocab["Age"], "type": "single"},
        {"key": "Gender", "label": "Genre ?",
         "options": vocab["Gender"], "type": "single"},
        {"key": "Symptoms", "label": "Symptômes principaux (sélection multiple) :",
         "options": art["symptom_tokens"], "type": "multi"},
        {"key": "Experiencing_Symptoms", "label": "Autres symptômes (sélection multiple) :",
         "options": art["experiencing_tokens"], "type": "multi"},
        {"key": "Severity", "label": "Gravité ressentie ?",
         "options": vocab["Severity"], "type": "single"},
        {"key": "Contact", "label": "Contact avec un patient COVID ?",
         "options": vocab["Contact"], "type": "single"},
    ]


def reset_session(questions):
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour. Je suis votre assistant pré-diagnostic COVID-19. "
                                          "Je vais vous poser 7 questions pour évaluer votre situation."}
    ]
    st.session_state.questions = questions
    st.session_state.finished = False


def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🩺" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"], unsafe_allow_html=True)


def render_result(answers, art):
    X_in = encode_answers(answers, art["mlb_sym"], art["mlb_exp"], art["feature_names"])
    label = int(art["model"].predict(X_in)[0])
    proba = art["model"].predict_proba(X_in)[0]

    tag = "POSITIF" if label == 1 else "NÉGATIF"
    color = "#3c6dd7" if label == 1 else "#388e3c"
    conf = proba[label] * 100

    summary_lines = []
    for k, v in answers.items():
        v_str = ", ".join(v) if isinstance(v, list) else v
        if not v_str:
            v_str = "_(aucun)_"
        summary_lines.append(f"- **{k}** : {v_str}")
    summary = "\n".join(summary_lines)

    return f"""
<div style="padding:20px; border-radius:12px;
            background:{color}1A; border:2px solid {color};">
    <h2 style="color:{color}; margin:0;">Diagnostic : COVID-19 {tag}</h2>
    <p style="font-size:18px; margin:12px 0;">
        Confiance du modèle : <strong>{conf:.1f}%</strong>
    </p>
    <p>Probabilité positive : <strong>{proba[1]*100:.1f}%</strong> &nbsp;|&nbsp;
       Probabilité négative : <strong>{proba[0]*100:.1f}%</strong></p>
</div>

**Réponses collectées :**

{summary}

<small style="color:#666;">Ce résultat ne constitue pas un avis médical.</small>
"""


def main():
    art = load_artifacts()
    questions = build_questions(art)

    st.title("🩺 Chatbot COVID-19")
    st.caption("Pré-diagnostic à partir d'un interrogatoire médical guidé.")

    with st.sidebar:
        st.header("À propos")
        st.markdown(
            "- **Modèle** : Random Forest\n"
            "- **Dataset** : 316 800 cas synthétiques\n"
            "- **Features** : 38 variables encodées\n"
        )
        if st.button("🔄 Réinitialiser la conversation", use_container_width=True):
            reset_session(questions)
            st.rerun()
        st.divider()
        st.caption("⚠️ Ce diagnostic est à but pédagogique uniquement.")

    if "step" not in st.session_state:
        reset_session(questions)

    display_chat()

    if st.session_state.finished:
        return

    step = st.session_state.step
    if step >= len(questions):
        result_md = render_result(st.session_state.answers, art)
        st.session_state.messages.append({"role": "assistant", "content": result_md})
        st.session_state.finished = True
        st.rerun()
        return

    q = questions[step]

    # On affiche la question dans le chat si pas encore fait
    last_assistant = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        None,
    )
    question_text = f"**Question {step + 1}/{len(questions)}** — {q['label']}"
    if not last_assistant or q["label"] not in last_assistant["content"]:
        st.session_state.messages.append({"role": "assistant", "content": question_text})
        st.rerun()

    # Formulaire de réponse
    with st.form(key=f"form_step_{step}", clear_on_submit=False):
        if q["type"] == "single":
            value = st.radio(q["label"], q["options"], key=f"input_{step}")
        else:
            value = st.multiselect(q["label"], q["options"], key=f"input_{step}")

        submitted = st.form_submit_button(
            "Suivant" if step < len(questions) - 1 else "🔬 Prédire",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if q["type"] == "multi":
            answer_value = list(value) if value else []
            display_value = ", ".join(answer_value) if answer_value else "_(aucun)_"
        else:
            answer_value = value
            display_value = value

        st.session_state.answers[q["key"]] = answer_value
        st.session_state.messages.append({"role": "user", "content": display_value})
        st.session_state.step += 1
        st.rerun()


if __name__ == "__main__":
    main()
