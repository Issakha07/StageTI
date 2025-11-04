import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer les variables
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

# Initialiser Flask
app = Flask(__name__)

# Fonction pour interroger Azure Cognitive Search
def search_documents(query):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    payload = {
        "search": query,
        "top": 3  # Nombre de passages à récupérer
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Fonction pour générer une réponse avec Azure OpenAI
def generate_answer(context, question):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/chat/completions?api-version=2023-07-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    messages = [
        {"role": "system", "content": "Tu es un assistant qui répond uniquement à partir du contexte fourni."},
        {"role": "user", "content": f"Contexte : {context}\nQuestion : {question}"}
    ]
    payload = {
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Endpoint API pour le chatbot
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("question", "")

        # Étape 1 : Recherche dans Cognitive Search
        search_results = search_documents(question)
        context = " ".join([doc.get('content', '') for doc in search_results.get('value', [])])

        # Vérifie si la recherche renvoie du contenu
        if not context.strip():
            return jsonify({"answer": "Aucun document trouvé dans l'index Azure Cognitive Search."}), 200

        # Étape 2 : Génération de la réponse
        answer = generate_answer(context, question)
        return jsonify({"answer": answer['choices'][0]['message']['content']})

    except requests.exceptions.HTTPError as e:
        print("Erreur HTTP Azure :", e.response.text)
        return jsonify({"error": f"Erreur Azure API : {e.response.text}"}), 500
    except Exception as e:
        print("Erreur interne :", str(e))
        return jsonify({"error": f"Erreur interne : {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)