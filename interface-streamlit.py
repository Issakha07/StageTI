import streamlit as st
import requests

# URL de ton API Flask (en local ou déployée)
API_URL = "http://localhost:8000/chat"  # Remplace par l'URL de ton Web App après déploiement

st.title("Chatbot RAG avec Azure Cognitive Search + OpenAI")

# --- ✅ Formulaire pour permettre la validation avec Entrée ---
with st.form("chat_form", clear_on_submit=False):
    question = st.text_input("Posez votre question :", "")
    submitted = st.form_submit_button("Envoyer")

# --- Si l’utilisateur appuie sur Entrée ou clique sur Envoyer ---
if submitted:
    if question.strip() == "":
        st.warning("Veuillez entrer une question.")
    else:
        try:
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                answer = response.json().get("answer", "Pas de réponse disponible.")
                st.success(answer)
            else:
                st.error(f"Erreur API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")
