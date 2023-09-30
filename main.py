import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import OpenAI
import pinecone
from langchain.chains import RetrievalQA

pinecone.init(environment='gcp-starter',api_key='')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Hello VectorStore')
    loader = TextLoader("mediumblogs/mediumblog1.txt")
    document = loader.load()
    print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('openai_api_key'))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name='langchain-tutorial')

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type='stuff', retriever= docsearch.as_retriever()
    )

    query= "What is a vector database. Give me a 15 word answer for a beginner"
    result= qa({"query": query})


    print(result)


