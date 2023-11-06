from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfFileReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

def main():
    st.set_page_config(page_title="Chat with your PDF file")
    st.header("Chat with your PDF file 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    # Extract the text from the PDF file
    if pdf is not None:
        pdf_reader = PdfFileReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = char_text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()  # Check the OpenAIEmbeddings constructor arguments
        docsearch = FAISS.from_texts(text_chunks, embeddings)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        # Show user input
        query = st.text_input("Type your question:")
        if query:
            docs = docsearch.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)

            st.write(response)

if __name__ == '__main__':
    main()
