# 📚 Chat avec plusieurs PDFs (RAG + Streamlit)

Ce projet est une application Streamlit qui vous permet de **poser des questions en langage naturel** sur plusieurs fichiers PDF, grâce à une approche **RAG (Retrieval-Augmented Generation)**.

L'application extrait le texte de vos PDF, le segmente, le vectorise avec des embeddings de Hugging Face, et vous permet d'interagir avec vos documents via un modèle LLM hébergé (LLaMA-2 via HuggingFace Hub ou OpenRouter).

---

## ✨ Fonctionnalités

- 📥 Chargement de plusieurs fichiers PDF
- 📄 Extraction et découpage intelligent du texte
- 🔎 Recherche vectorielle avec FAISS
- 🤖 Chat contextuel avec mémoire de conversation
- 🧠 Modèles LLM gratuits via HuggingFace Hub ou OpenRouter
- 🌐 Interface simple et interactive via Streamlit

---

## 🧰 Technologies utilisées

| Composant                    | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `Streamlit`                 | Interface web interactive                            |
| `LangChain`                 | Orchestration RAG et chaîne conversationnelle        |
| `HuggingFaceEmbeddings`     | Encodage des textes pour la recherche sémantique     |
| `FAISS`                     | Base vectorielle pour la similarité de texte         |
| `LLaMA 2` ou `Mistral`      | Modèles LLM (gratuits via HuggingFace Hub ou OpenRouter) |
| `PyPDF2`                    | Extraction de texte depuis des fichiers PDF          |
| `.env`                      | Gestion sécurisée des clés API                       |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/votre-utilisateur/multi-pdf-chat-rag.git
cd multi-pdf-chat-rag
```
### 2. Créer et activer un environnement virtuel
```
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```
### 3. Installer les dépendances
```
pip install -r requirements.txt
```
### 4. Créer un fichier .env
```
HUGGINGFACEHUB_API_KEY=your_huggingface_token
# ou si vous utilisez OpenRouter à la place :
OPENROUTER_API_KEY=your_openrouter_key
```
### Exécution de l'application
```
streamlit run app.py
```
Puis ouvrez l’application dans votre navigateur à l’adresse [http://localhost:8501]

## Personnalisation
### Changer de modèle LLM
**Dans get_conversation_chain() :**

**- Pour Hugging Face :**
```
from langchain_community.llms import huggingface_hub
llm = huggingface_hub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    token=os.getenv("HUGGINGFACEHUB_API_KEY"),
    model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
)
```
**- Pour OpenRouter :**
```
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)
```

