import streamlit as st
import requests

# URL de ton API Flask (en local ou déployée)
API_URL = "http://localhost:8000/chat"  # Remplace par l'URL de ton Web App après déploiement
#http://localhost:8501

st.title("Chatbot RAG avec Azure Cognitive Search + OpenAI")

# Champ de saisie pour la question
question = st.text_input("Posez votre question :", "")

if st.button("Envoyer"):
    if question.strip() == "":
        st.warning("Veuillez entrer une question.")
    else:
        # Appel à l'API Flask
        try:
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                answer = response.json().get("answer", "Pas de réponse disponible.")
                st.success(answer)
            else:
                st.error(f"Erreur API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")