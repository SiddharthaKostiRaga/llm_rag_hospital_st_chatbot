import os
from langchain.vectorstores.neo4j_vector import Neo4jVector

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq 

from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")

hf_embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=hf_embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="reviews",
    node_label="Review",
    text_node_properties=[
        "physician_name",
        "patient_name",
        "text",
        "hospital_name",
    ],
    embedding_node_property="embedding",
)