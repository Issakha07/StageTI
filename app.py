import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Variables d'environnement
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

app = Flask(__name__)

# -------------------------------------------------------
# 🌐 Fonction de traduction si question FR -> EN
# -------------------------------------------------------
def translate_to_english(question):
    """Traduit la question française en anglais via Azure OpenAI."""
    if not any(c in "éèàùçâêîôûëïü" for c in question.lower()):
        return question  # déjà en anglais

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    payload = {
        "messages": [
            {"role": "system", "content": "Translate the following French question into English, without explanation."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    translated = response.json()["choices"][0]["message"]["content"].strip()
    return translated

# -------------------------------------------------------
# 🔍 Recherche dans Azure Cognitive Search
# -------------------------------------------------------
def search_documents(query):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_KEY}
    payload = {"search": query, "top": 3}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# -------------------------------------------------------
# 🧠 Génération de réponse avec OpenAI
# -------------------------------------------------------
def generate_answer(context, question):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}

    # Détecte la langue de la question
    lang = "fr" if any(c in "éèàùçâêîôûëïü" for c in question.lower()) else "en"

    # Prompt bilingue amélioré
    system_message = (
        "You are a bilingual hospital IT support chatbot. "
        "The knowledge base is in English, so always search information in English. "
        "If the user asks in French, translate their question internally into English for searching, "
        "but reply in French. "
        "If the user asks in English, reply in English. "
        "If no relevant information is found, politely say that the topic is outside the hospital IT support scope. "
        "Do not invent facts. Be clear, concise, and professional."
    )
    

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion ({lang}): {question}"}
    ]

    payload = {"messages": messages, "temperature": 0.2}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# -------------------------------------------------------
# 🧩 Route principale Flask
# -------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Veuillez poser une question."}), 200

        # Étape 1️⃣ Traduction si question FR
        translated_query = translate_to_english(question)

        # Étape 2️⃣ Recherche dans Cognitive Search
        search_results = search_documents(translated_query)
        context = " ".join([doc.get("content", "") for doc in search_results.get("value", [])])

        # Étape 3️⃣ Si aucun document trouvé → message clair et poli
        if not context.strip():
            # Si la question est en français
            if any(c in "éèàùçâêîôûëïü" for c in question.lower()):
                msg = (
                    "Je suis désolé, mais votre question manque de clarté ou ne relève pas du "
                    "service d'assistance informatique hospitalière. "
                    "Pourriez-vous la préciser ou fournir davantage de contexte ?"
                )
            # Si la question est en anglais
            else:
                msg = (
                    "I'm sorry, but your question is unclear or falls outside the hospital IT support scope. "
                    "Could you please clarify or provide more context?"
                )
            return jsonify({"answer": msg}), 200


        # Étape 4️⃣ Génération de la réponse
        answer = generate_answer(context, question)
        final_answer = answer["choices"][0]["message"]["content"]
        return jsonify({"answer": final_answer})

    except requests.exceptions.HTTPError as e:
        print("Erreur HTTP Azure :", e.response.text)
        return jsonify({"error": f"Erreur Azure API : {e.response.text}"}), 500
    except Exception as e:
        print("Erreur interne :", str(e))
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500


# -------------------------------------------------------
# 🚀 Lancement du serveur
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
