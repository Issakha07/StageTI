# -*- coding: utf-8 -*-
"""
IT Support Chatbot - Frontend Streamlit Amélioré
Interface moderne avec historique conversationnel
"""

import streamlit as st
import requests
from datetime import datetime
import time

# ==========================================
# 🎨 CONFIGURATION PAGE
# ==========================================
st.set_page_config(
    page_title="IT Support Chatbot 🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🔧 CONFIGURATION API
# ==========================================
API_URL = "http://localhost:8000/api/chat"
API_RESET_URL = "http://localhost:8000/api/reset"
API_TIMEOUT = 30

# ==========================================
# 🎨 STYLES CSS
# ==========================================
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Messages utilisateur */
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Messages assistant */
    .assistant-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1rem 1.5rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #9c27b0;
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .message-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: #666;
    }
    
    .message-content {
        line-height: 1.7;
        color: #333;
    }
    
    /* Sources */
    .sources-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        animation: fadeIn 0.4s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Statistiques */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.95rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Boutons */
    .stButton>button {
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Input */
    .stTextInput>div>div>input {
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🔧 INITIALISATION SESSION
# ==========================================
def init_session():
    """Initialise l'état de session"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour! 👋 Je suis votre assistant IT Support. Comment puis-je vous aider aujourd'hui?",
                "timestamp": datetime.now(),
                "sources": []
            }
        ]
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "request_count" not in st.session_state:
        st.session_state.request_count = 0
    
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()

init_session()

# ==========================================
# 🛠️ FONCTIONS UTILITAIRES
# ==========================================
def format_time(dt: datetime) -> str:
    """Formate timestamp"""
    return dt.strftime("%H:%M")

def calculate_duration() -> str:
    """Calcule durée session"""
    delta = datetime.now() - st.session_state.session_start
    minutes = int(delta.total_seconds() / 60)
    seconds = int(delta.total_seconds() % 60)
    return f"{minutes}m {seconds}s"

def send_message(question: str) -> dict:
    """Envoie message à l'API"""
    payload = {
        "question": question,
        "session_id": st.session_state.session_id
    }
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                # Sauvegarder session_id pour continuité
                st.session_state.session_id = data.get("session_id")
                return data
            
            elif response.status_code == 429:
                error = response.json().get("detail", "Trop de requêtes")
                st.error(f"⚠️ {error}")
                return None
            
            elif response.status_code == 503:
                if attempt < max_retries - 1:
                    st.warning(f"⏳ Service occupé, nouvelle tentative ({attempt+2}/{max_retries})...")
                    time.sleep(2)
                    continue
                else:
                    st.error("❌ Service Azure temporairement indisponible. Réessayez dans 30s.")
                    return None
            
            else:
                error = response.json().get("detail", "Erreur inconnue")
                st.error(f"❌ Erreur API ({response.status_code}): {error}")
                return None
        
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                st.warning(f"⏱️ Timeout, nouvelle tentative ({attempt+2}/{max_retries})...")
                time.sleep(1)
                continue
            else:
                st.error("⏱️ La requête a expiré. Le service met trop de temps à répondre.")
                return None
        
        except requests.exceptions.ConnectionError:
            st.error("🔌 Impossible de se connecter à l'API. Vérifiez que le backend est lancé sur le port 8000.")
            return None
        
        except Exception as e:
            st.error(f"❌ Erreur inattendue: {str(e)}")
            return None
    
    return None

def reset_conversation():
    """Réinitialise la conversation"""
    try:
        if st.session_state.session_id:
            requests.post(
                f"{API_RESET_URL}/{st.session_state.session_id}",
                timeout=5
            )
    except:
        pass
    
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Conversation réinitialisée. Posez-moi vos questions IT! 🚀",
            "timestamp": datetime.now(),
            "sources": []
        }
    ]
    st.session_state.session_id = None
    st.session_state.request_count = 0
    st.session_state.session_start = datetime.now()
    st.rerun()

# ==========================================
# 🎨 INTERFACE
# ==========================================

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🏥 IT Support Chatbot</h1>
    <p>Assistant intelligent pour le support informatique hospitalier</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📊 Statistiques de Session")
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{st.session_state.request_count}</div>
        <div class="stat-label">Questions</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{calculate_duration()}</div>
        <div class="stat-label">Durée</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Actions
    st.markdown("### ⚙️ Actions")
    
    if st.button("🔄 Nouvelle conversation", use_container_width=True, type="primary"):
        reset_conversation()
    
    if st.button("📥 Exporter l'historique", use_container_width=True):
        export = "=== HISTORIQUE CONVERSATION ===\n\n"
        for msg in st.session_state.messages:
            role = "VOUS" if msg["role"] == "user" else "ASSISTANT"
            export += f"[{format_time(msg['timestamp'])}] {role}:\n{msg['content']}\n"
            if msg.get("sources"):
                export += f"Sources: {', '.join(msg['sources'])}\n"
            export += "\n" + "="*50 + "\n\n"
        
        st.download_button(
            label="💾 Télécharger TXT",
            data=export,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.divider()
    
    # Informations
    st.markdown("### ℹ️ Informations")
    st.markdown("""
    <div class="info-box">
    <strong>✨ Fonctionnalités:</strong><br>
    • 🌐 Bilingue FR/EN<br>
    • 📚 Base Azure Search<br>
    • 💬 Historique contextuel<br>
    • 🔒 Sessions sécurisées
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Limites:</strong><br>
    • Max 500 caractères/question<br>
    • Délai 3s entre doublons<br>
    • Timeout 30s par requête
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📞 Support Direct")
    st.info("""
    **Téléphone:** 📞 Poste 5555  
    **Email:** ✉️ it-support@hopital.qc.ca  
    **Urgences:** 🚨 Poste 9999
    """)
    
    # Version
    st.markdown("---")
    st.caption("Version 2.0.0 | Propulsé par Azure OpenAI")

# --- ZONE MESSAGES ---
st.markdown("### 💬 Conversation")

# Container scrollable
chat_container = st.container()
with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <div class="message-header">
                    👤 <strong>Vous</strong> • {format_time(msg['timestamp'])}
                </div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <div class="message-header">
                    🤖 <strong>Assistant IT</strong> • {format_time(msg['timestamp'])}
                </div>
                <div class="message-content">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher sources si présentes
            if msg.get("sources") and len(msg["sources"]) > 0:
                sources_text = "<br>• ".join(msg["sources"])
                st.markdown(f"""
                <div class="sources-box">
                    📚 <strong>Sources consultées:</strong><br>
                    • {sources_text}
                </div>
                """, unsafe_allow_html=True)

# --- ZONE INPUT ---
st.divider()

# Formulaire de saisie
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Votre question:",
            placeholder="Ex: Comment réinitialiser mon mot de passe Windows?",
            label_visibility="collapsed",
            max_chars=500
        )
    
    with col2:
        submit = st.form_submit_button("Envoyer 📤", use_container_width=True, type="primary")

# Traitement
if submit and user_input:
    if len(user_input.strip()) == 0:
        st.warning("⚠️ Veuillez entrer une question.")
    
    else:
        # Ajouter message utilisateur
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "sources": []
        })
        
        # Appeler API
        with st.spinner("🤔 Recherche en cours..."):
            response = send_message(user_input)
        
        if response:
            # Ajouter réponse
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("answer", "Erreur de réponse"),
                "timestamp": datetime.now(),
                "sources": response.get("sources", [])
            })
            
            st.session_state.request_count += 1
        
        # Rafraîchir
        st.rerun()

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem; padding: 1rem 0;">
    <p>🔒 <strong>Confidentialité:</strong> Vos conversations sont temporaires et sécurisées</p>
    <p>Développé avec ❤️ pour l'hôpital • Propulsé par Azure OpenAI GPT-4o</p>
</div>
""", unsafe_allow_html=True)