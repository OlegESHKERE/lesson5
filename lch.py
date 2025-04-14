import os
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class WebQnA:
    def __init__(self):
        load_dotenv()
        self.ogpt_key = os.getenv("kry")

        # Налаштування LLM
        self.llm = ChatOpenAI(
            openai_api_key=self.ogpt_key,
            base_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-4o-mini"
        )

    def fetch_page_text(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logging.error(f"Помилка при завантаженні сторінки: {e}")
            return ""

    def build_qa_chain(self, text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([text])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return qa_chain

    def ask_question_about_url(self, url, question):
        page_text = self.fetch_page_text(url)
        if not page_text:
            print("Не вдалося завантажити текст сторінки.")
            return

        qa_chain = self.build_qa_chain(page_text)
        response = qa_chain.invoke({"query": question})
        answer = response["result"]
        sources = response["source_documents"]

        print(f"Питання: {question}")
        print(f"Відповідь: {answer}")
        

# === Використання ===
if __name__ == "__main__":
    qna = WebQnA()
    url = "https://uk.wikipedia.org/wiki/JavaScript"
    qna.ask_question_about_url(url, "що таке JavaScript?")
