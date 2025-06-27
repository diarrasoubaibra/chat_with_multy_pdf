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
'''
