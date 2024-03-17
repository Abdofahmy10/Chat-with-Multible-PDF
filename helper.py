import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Parameters:
        pdf_docs (List[str]): List of PDF document file paths.

    Returns:
        str: Concatenated text extracted from all the PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            with open(pdf, 'rb') as file:
                pdf_reader = PdfReader(file)
                text += ''.join(page.extract_text() for page in pdf_reader.pages)
        except FileNotFoundError:
            print(f"File '{pdf}' not found.")
        except Exception as e:
            print(f"An error occurred while processing '{pdf}': {e}")
    
    return text

def get_text_chunks(text):
    """
    Splits the given text into smaller chunks.

    Parameters:
        text (str): The input text to be split.

    Returns:
        List[str]: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=800,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """
    Generates a vector store from a list of text chunks.

    Parameters:
        text_chunks (List[str]): List of text chunks.

    Returns:
        VectorStore: The generated vector store.
    """
    embeddings =  GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")
    pkl = vectorstore.serialize_to_bytes()
    return vectorstore ,pkl

def get_conversational_chain():
    """
    The chain will take a list of documents 
    inserts them all into a prompt, and passes that prompt to an LLM

    """

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.5)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) 

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings , allow_dangerous_deserialization = True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Your answer is : \n " , response["output_text"])