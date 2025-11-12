# -*- coding: utf-8 -*-
"""
IT Support Chatbot - Backend FastAPI Amélioré
Version simplifiée pour tests locaux avec Azure Search + OpenAI
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# Configuration langdetect
DetectorFactory.seed = 0

# Charger variables d'environnement
load_dotenv()

# ==========================================
# 🔧 CONFIGURATION
# ==========================================
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "it-support-docs")

# Limites de sécurité
MAX_QUESTION_LENGTH = 500
MAX_HISTORY_SIZE = 10
RATE_LIMIT_SECONDS = 3

# ==========================================
# 📊 LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 🏗️ INITIALISATION FASTAPI
# ==========================================
app = FastAPI(
    title="IT Support Chatbot API",
    description="Healthcare IT Support avec Azure OpenAI",
    version="2.0.0"
)

# CORS pour Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 📦 MODÈLES PYDANTIC
# ==========================================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    session_id: Optional[str] = None
    
    @validator('question')
    def validate_question(cls, v):
        # Nettoyer la question
        v = v.strip()
        
        # Détection injections basiques
        dangerous = ['<script>', 'javascript:', 'DROP TABLE', 'DELETE FROM']
        if any(pattern.lower() in v.lower() for pattern in dangerous):
            raise ValueError("Contenu suspect détecté dans la question")
        
        return v

class ChatResponse(BaseModel):
    answer: str
    language: str
    sources: List[str] = []
    session_id: str

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ==========================================
# 💾 GESTION SESSIONS (EN MÉMOIRE)
# ==========================================
# Pour production multi-instance, utiliser Redis
sessions_store: Dict[str, Dict] = {}

def get_or_create_session(session_id: Optional[str]) -> tuple[str, Dict]:
    """Récupère ou crée une session utilisateur"""
    if not session_id or session_id not in sessions_store:
        # Créer nouvelle session
        session_id = os.urandom(16).hex()
        sessions_store[session_id] = {
            'chat_history': [],
            'last_question': None,
            'last_time': None,
            'created_at': datetime.now()
        }
        logger.info(f"Nouvelle session créée: {session_id}")
    
    return session_id, sessions_store[session_id]

def clean_old_sessions():
    """Nettoie sessions > 2h (appelé périodiquement)"""
    now = datetime.now()
    to_delete = [
        sid for sid, data in sessions_store.items()
        if (now - data['created_at']).total_seconds() > 7200
    ]
    for sid in to_delete:
        del sessions_store[sid]
    
    if to_delete:
        logger.info(f"Nettoyé {len(to_delete)} sessions expirées")

# ==========================================
# 🌐 DÉTECTION DE LANGUE
# ==========================================
def detect_language(text: str) -> str:
    """
    Détecte la langue de la question (fr/en)
    Returns 'fr' for French, 'en' for English
    """
    try:
        detected = detect(text.strip())
        return 'fr' if detected == 'fr' else 'en'
    except Exception as e:
        logger.error(f"Erreur détection langue: {e}")
        return 'en'  # Default to English on error

# ==========================================
# 🔄 TRADUCTION AZURE OPENAI
# ==========================================
def translate_to_english(question: str) -> str:
    """Traduit FR->EN si nécessaire"""
    if detect_language(question) == 'en':
        return question
    
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "Translate this French IT question to English. Return ONLY the translation."
            },
            {"role": "user", "content": question}
        ],
        "temperature": 0.0,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        translated = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Traduction: {question[:30]}... -> {translated[:30]}...")
        return translated
    except Exception as e:
        logger.error(f"Erreur traduction: {e}")
        return question  # Fallback

# ==========================================
# 🔍 RECHERCHE AZURE COGNITIVE SEARCH
# ==========================================
def search_documents(query: str, top_k: int = 3) -> Dict:
    """Recherche dans Azure Search avec retry"""
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2023-07-01-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }
    payload = {
        "search": query,
        "top": top_k,
        "select": "content"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limit Azure Search, attente {retry_after}s")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            results = response.json()
            logger.info(f"Recherche OK: {len(results.get('value', []))} docs trouvés")
            return results
        
        except requests.exceptions.Timeout:
            logger.error(f"Timeout recherche (tentative {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                raise
        
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            raise
    
    return {"value": []}

# ==========================================
# 🧠 GÉNÉRATION RÉPONSE OPENAI
# ==========================================
def generate_answer(context: str, question: str, chat_history: List[Dict]) -> str:
    """Génère réponse avec GPT-4o"""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    
    user_lang = detect_language(question)
    
    current_language = "French" if user_lang == "fr" else "English"
    system_prompt = f"""You are a bilingual hospital IT Support assistant.

STRICT LANGUAGE RULE:
The user's question is in {current_language}. You MUST respond in {current_language}.
- If question is in English → respond in English
- If question is in French → respond in French
- NEVER mix languages in your response
- Translate context/knowledge if needed but keep response in user's language

RESPONSE STYLE:
- Clear and concise
- Use bullet points for steps
- Professional and courteous
- Use ONLY information from provided context
- If no relevant info → politely say it's outside IT scope


CONTEXTE:
{context if context else "Aucune information pertinente trouvée"}
"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Ajouter historique (5 derniers messages max)
    for msg in chat_history[-5:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    messages.append({
        "role": "user",
        "content": question
    })
    
    payload = {
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        logger.info(f"Réponse générée: {len(answer)} caractères")
        return answer
    
    except requests.exceptions.Timeout:
        logger.error("Timeout génération réponse")
        if user_lang == 'fr':
            return "Désolé, le service met trop de temps. Réessayez."
        return "Sorry, service timeout. Please retry."
    
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        raise

# ==========================================
# 🌐 ROUTES API
# ==========================================
@app.get("/")
async def root():
    """Page d'accueil API"""
    return {
        "service": "IT Support Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "reset": "/api/reset/{session_id}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions_store)
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Endpoint principal du chatbot"""
    try:
        # Nettoyer vieilles sessions
        clean_old_sessions()
        
        # Récupérer/créer session
        session_id, session_data = get_or_create_session(request.session_id)
        
        question = request.question
        user_lang = detect_language(question)
        
        logger.info(f"[Session {session_id[:8]}] Question ({user_lang}): {question[:80]}")
        
        # === DÉTECTION DOUBLONS ===
        if session_data['last_question'] == question and session_data['last_time']:
            time_diff = (datetime.now() - session_data['last_time']).total_seconds()
            if time_diff < RATE_LIMIT_SECONDS:
                logger.warning(f"Question dupliquée détectée: {question[:50]}")
                msg = ("Veuillez attendre quelques secondes avant de soumettre la même question."
                       if user_lang == 'fr' else
                       "Please wait a few seconds before submitting the same question.")
                raise HTTPException(status_code=429, detail=msg)
        
        # === PIPELINE RAG ===
        # 1. Traduction
        translated_query = translate_to_english(question)
        
        # 2. Recherche documents
        search_results = search_documents(translated_query)
        documents = search_results.get("value", [])
        
        # 3. Construire contexte
        if documents:
            context = "\n\n".join([
                f"[Document: {doc.get('title', 'N/A')}]\n{doc.get('content', '')}"
                for doc in documents[:3]
            ])
            sources = [doc.get('title', 'Document sans titre') for doc in documents[:3]]
        else:
            context = ""
            sources = []
        
        # 4. Générer réponse
        if not context.strip():
            if user_lang == 'fr':
                answer = ("Je n'ai pas trouvé d'information pertinente dans ma base de connaissances IT. "
                         "Pouvez-vous reformuler ou contacter le support au poste 5555?")
            else:
                answer = ("I couldn't find relevant information in my IT knowledge base. "
                         "Could you rephrase or contact support at extension 5555?")
        else:
            chat_history = session_data['chat_history']
            answer = generate_answer(context, question, chat_history)
        
        # === MISE À JOUR SESSION ===
        session_data['chat_history'].append({"role": "user", "content": question})
        session_data['chat_history'].append({"role": "assistant", "content": answer})
        
        # Limiter taille historique
        if len(session_data['chat_history']) > MAX_HISTORY_SIZE * 2:
            session_data['chat_history'] = session_data['chat_history'][-MAX_HISTORY_SIZE*2:]
        
        session_data['last_question'] = question
        session_data['last_time'] = datetime.now()
        
        return ChatResponse(
            answer=answer,
            language=user_lang,
            sources=sources,
            session_id=session_id
        )
    
    # === GESTION ERREURS ===
    except HTTPException:
        raise
    
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        logger.error(f"Erreur Azure ({status_code}): {e.response.text}")
        
        if status_code == 429:
            msg = ("Services Azure surchargés. Réessayez dans 30s."
                   if user_lang == 'fr' else
                   "Azure services busy. Retry in 30s.")
            raise HTTPException(status_code=503, detail=msg)
        
        elif status_code >= 500:
            msg = ("Erreur serveur Azure."
                   if user_lang == 'fr' else
                   "Azure server error.")
            raise HTTPException(status_code=503, detail=msg)
        
        else:
            raise HTTPException(status_code=500, detail="Erreur communication Azure")
    
    except Exception as e:
        logger.exception("Erreur interne inattendue")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/api/reset/{session_id}")
async def reset_session(session_id: str):
    """Réinitialise une session"""
    if session_id in sessions_store:
        del sessions_store[session_id]
        logger.info(f"Session réinitialisée: {session_id}")
        return {"message": "Session réinitialisée", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session introuvable")

# ==========================================
# 🚀 DÉMARRAGE
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    # Vérifier variables critiques
    required_vars = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"❌ Variables manquantes: {missing}")
        exit(1)
    
    logger.info("🚀 Démarrage IT Support Chatbot API...")
    logger.info(f"📚 Index Azure: {INDEX_NAME}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en dev
        log_level="info"
    )