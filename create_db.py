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
import logging
import json
load_dotenv()

DATA_PATH = "data/books"
CHROMA_PATH  = "chroma"

openai.api_key = os.environ['OPENAI_API_KEY']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='create_db.log',
    filemode='w'
)

# Load all the .txt files that we currently have in our books folder

def load_documents():
    logging.info("Loading documents...")
    loader = DirectoryLoader(DATA_PATH , glob="*.txt")
    documents = loader.load()
    return documents

# Method for adding information at the start and end of the chunks
def add_info_to_chunks(chunk, prefix , suffix):
    if prefix is None or prefix == "":
        prefix = "No hay titulo del articulo"
    if suffix is None or suffix == "":
        suffix = "No se tiene una firma"

    chunk.page_content = f"{prefix}\n{chunk.page_content}\n{suffix}"

# Set how we want our chunking to be done

def split_text(documents: list[Document]):
    logging.info("Creating chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Set the chunk size in characters
        chunk_overlap=500, # Each chunk is going to have an overlap of 500 characters
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    
    # Print how many documents we got and how many chunks they generated
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # See the content of chunk number 10 on the logs
    logging.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    document = chunks[10]
    logging.info(
        "Example of chunk number 10:\n%s\n\n--- METADATA ---\n%s",
        document.page_content,
        document.metadata
    )

    return chunks

# Get metadata for chunks
def get_metadata_arguments(chunks : list[Document]):
    logging.info("Getting metadata for each chunk...")
    metas = []

    for i, doc in enumerate(chunks):
        source = doc.metadata.get("source", "")
        source_file = Path(source).name if source else None

        # Here goes logic for getting the meta data
        title_i = None
        signature_i = None
        date_i = None

        add_info_to_chunks(doc, title_i, signature_i) # Change when we got the original information

        metas.append({
            "articulo": title_i,
            "firma": signature_i,
            "fecha": date_i,
            "fuente": source_file,
            "chunk_index": i,
        })

    return metas

# Add metadata to chunks
def attach_metadata(chunks : list[Document] , metas : list[dict]):
    logging.info("Adding metadata...")

    if len(chunks) != len(metas):
        logging.error(f"Chunks and meta array have different sizes Chunk size: {len(chunks)} Meta size: {len(metas)} ")
        raise ValueError(f"Chunks and meta array have different sizes \nChunk size: {len(chunks)} \nMeta size: {len(metas)}")

    for doc , meta in zip(chunks , metas):
        doc.metadata.update(meta)

    logging.info(f"This is how chunks metadata look: {chunks[10].metadata}")
    return chunks

def save_to_chroma(chunks: list[Document]):
    logging.info("Saving information to db...")
    # Delete the data base if there is one already created
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Save the chunks into a chroma database using OpeinAIEmbeddings 
    Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    documents = load_documents()
    raw_chunks = split_text(documents)
    metas = get_metadata_arguments(raw_chunks)
    chunks = attach_metadata(raw_chunks , metas)
    save_to_chroma(chunks)

    # Test for looking at the db and how this thingy is saved
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    one = db._collection.peek(limit=1)  # trae 1 registro cualquiera

    print("ID:", one["ids"][0])
    print("METADATA:", one["metadatas"][0])
    print("TEXTO (primeros 500 chars):\n", one["documents"][0][:200])

if __name__ == "__main__":
    try:
        main()
        logging.info("Finished :)")
    except:
        logging.error("Something went bad :(")
    




