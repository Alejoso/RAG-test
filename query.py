import openai
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import argparse
from langchain_core.prompts import ChatPromptTemplate


CHROMA_PATH  = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def main():

    # Define a parser for inputing the information on the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text" , type=str , help="The query text")
    args = parser.parse_args()
    query_text = args.query_text


    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH , embedding_function= embedding_function)

    # Do the similatiry and extract 3 results
    results = db.similarity_search_with_relevance_scores(query_text , k = 3)

    if len(results) == 0 or results[0][1] < 0.7: # 0.7 is the threshold we want to decide for our results
        print("Unable to find mathching results")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text , question = query_text)
    print(prompt)

    model = ChatOpenAI(
        model="gpt-5-nano"
    )
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source" , None) for doc, _score in results]
    formatted_response = f"Response: {response_text.text}\nSources: {sources}"
    print(formatted_response)



if __name__ == "__main__":
    main()
