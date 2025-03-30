from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI                                             
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv
load_dotenv()

def process_chat(query):
    query= str(query)
    
    
    embeddings = VertexAIEmbeddings(model="text-embedding-004")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3},
    )
    
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    
    standaloneQuestionTempaltePromt = PromptTemplate.from_template(
        "Given an  query, you will make an standalone question ,userinput:{query}, stand-alone question:"
    )
    

    setupTempaltePromt = PromptTemplate.from_template(
        "You are an AI that replies in friendly and helpful way you are a chatbot make sure that is is breif note that context may or may not related yu have to decide. The userinput is:{query}, context:{context}"
    )
    
    chain_retrival_chain = standaloneQuestionTempaltePromt | llm | StrOutputParser() 
    chain_relevant_doc = chain_retrival_chain.invoke({'query': query})
    context_and_query_chain = setupTempaltePromt | llm | StrOutputParser()
    final_chain = context_and_query_chain.invoke({'query': query, 'context':  chain_relevant_doc })
    return {'answer': final_chain}
    print('final--->',final_chain)
    
# process_chat('What is the best way to learn Python?')
