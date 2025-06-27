import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_hub

from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

from htlmTemplates import css, bot_template, user_template


load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() 
    return text

def get_text_chunks(texts):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(texts)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    # llm = ChatOpenAI()
    # llm = ChatOpenAI(
    #     model="microsoft/mai-ds-r1:free",
    #     openai_api_base="https://openrouter.ai/api/v1",
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY")
    # )

    llm = huggingface_hub(repo_id="meta-llama/Llama-2-7b-chat-hf",
                          token=os.getenv("HUGGINGFACEHUB_API_KEY"),
                          model_kwargs={"temperature": 0.1, "max_new_tokens": 512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,)
    return conversation_chain
def handle_userinput(user_question):
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("La chaîne de conversation n'a pas été initialisée.")
        return

    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":book:",
    )
    st.write(css, unsafe_allow_html=True)
     
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input(
        "Entrez votre question",
        placeholder="Posez votre question sur vos documments...",
    )
    if user_question:
        handle_userinput(user_question)
    

    st.write(user_template.replace("{{MSG}}", st.session_state.get("user_input", "Salut Ndev")), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Bonjour, comment puis-je vous aider ?"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Vos documents PDF")
        pfd_docs = st.file_uploader(
            "Importez vos fichiers PDF et cliquez sur 'Analyser'",
            type="pdf",
            accept_multiple_files=True,
        )
        if st.button("Analyser", key="analyze_button"):
            with st.spinner("Analyse en cours..."):
            # get pdf text
                raw_texts = get_pdf_text(pfd_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_texts)
            # st.write(text_chunks)

            # get vector store
            vector_store = get_vector_store(text_chunks)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vector_store)



if __name__ == "__main__":
    main()