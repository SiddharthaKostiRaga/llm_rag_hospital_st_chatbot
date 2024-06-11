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

review_template = """Your job is to use patient
reviews to answer questions about their experience at a hospital. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

#system prompt
review_system_prompt = SystemMessagePromptTemplate(prompt=PromptTemplate(template=review_template, input_variables=["context"]))

#human prompt
review_human_prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(template='{question}', input_variables = ["question"]))

messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(messages=messages, input_variables=["context", "question"])