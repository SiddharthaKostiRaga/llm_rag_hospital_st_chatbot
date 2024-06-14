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

"""create reviews vector chain using a Neo4j vector index retriever that returns 12 reviews embeddings from a similarity search. 
By setting chain_type to "stuff" in .from_chain_type(), ensure chain to pass all 12 reviews to the prompt."""

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(temperature=0, model_name="mixtral-8x7b-32768"), 
    chain_type = "stuff",
    retriever = neo4j_vector_index.as_retriever(k=12),
    )

reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt

"""To understand this line, we need to dissect the components and their relationships:

reviews_vector_chain:
This is an instance of RetrievalQA, created earlier using the from_chain_type method.
The RetrievalQA class is part of the langchain.chains module, which facilitates retrieval-based question answering.

combine_documents_chain:
Within the RetrievalQA instance, there's a chain responsible for combining documents retrieved from the vector store.
This is typically a chain that processes the retrieved documents (in this case, the reviews) and prepares them to be used by the language model (LLM).
llm_chain:

Within the combine_documents_chain, there's a chain that interacts with the language model.
The llm_chain handles the input to the language model, including formatting the prompt and processing the response.

prompt:
The prompt attribute of llm_chain specifies the template or structure used to formulate the input that the language model will process.
This prompt defines how the context (retrieved reviews) and the user's question are presented to the language model.

review_prompt:
review_prompt is an instance of ChatPromptTemplate created earlier in the code.
It combines system and human prompts and expects two input variables: context (the retrieved reviews) and question (the user's query).


Putting It All Together
The line of code reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt is configuring the QA system to use 
the review_prompt as the template for interacting with the language model. Here's the detailed flow:

Retrieval:
When a question is asked, the RetrievalQA system first retrieves the top 12 most similar reviews from the Neo4j vector store based on the query.

Combining Documents:
These retrieved reviews are then combined in a meaningful way by the combine_documents_chain. This chain ensures that the context provided to the language model is coherent and relevant.

Language Model Interaction:
The llm_chain within the combine_documents_chain takes this combined context and uses the review_prompt to format the input for the language model.
The review_prompt specifies that the context (retrieved reviews) and the user's question should be included in the prompt in a structured manner.

Prompt Structure:
review_prompt includes both a system message (providing instructions and context) and a human message (the user's question).
This structured prompt ensures the language model has clear guidelines on how to use the retrieved reviews to answer the question."""