from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import shutil
import os
import openai
from pathlib import Path
load_dotenv()

DATA_PATH = "data/books"
CHROMA_PATH  = "chroma"

openai.api_key = os.environ['OPENAI_API_KEY']


# Load all the .txt files that we currently have in our books folder

def load_documents():
    loader = DirectoryLoader(DATA_PATH , glob="*.txt")
    documents = loader.load()
    return documents

# Method for adding information at the start and end of the chunks
def add_info_to_chunks(chunks, prefix , suffix):
    if prefix is None or prefix == "":
        prefix = "No hay titulo del articulo"
    if suffix is None or suffix == "":
        suffix = "No se tiene una firma"

    for chunk in chunks:
        chunk.page_content = f"{prefix}\n{chunk.page_content}\n{suffix}"

# Set how we want our chunking to be done

def split_text(documents: list[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Set the chunk size in characters
        chunk_overlap=500, # Each chunk is going to have an overlap of 500 characters
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    source_of_documents = chunks[10].metadata.get("source" , "")
    title = Path(source_of_documents).name
    add_info_to_chunks(chunks, title, "Firmado por Alejito")
    
    # Print how many documents we got and how many chunks they generated
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # See the content of chunk number 10
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Delete the data base if there is one already created
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Save the chunks into a chroma database using OpeinAIEmbeddings 
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()





