from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI                                             
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os
from dotenv import load_dotenv
load_dotenv()
def process_query(query):
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
        "Given an email you have to make a stand-alone question. Don't miss the specific detail, email:{email}, stand-alone question:"
    )
    
    subjectTempaltePromt = PromptTemplate.from_template(
        "You are an AI that writes only one subject which is concise and appropriate for the given email. Note that you are the writer of this mail and should respond accordingly, email:{email}"
    )
    
    setupTempaltePromt = PromptTemplate.from_template(
        "You are an AI that replies professionally. The email belongs to 'parthakadam2007@gmail' and the name is Partha Kadam. Reply professionally and provide all requested details, email:{email}, context:{context}"
    )
    
    chain_retrival = standaloneQuestionTempaltePromt | llm | StrOutputParser()
    chain_generate_subject = subjectTempaltePromt | llm | StrOutputParser()
    
    chain_retrival_and_generate_output = RunnableParallel(
        standaloneQuestion=chain_retrival, subject=chain_generate_subject
    )
    
    standaloneQuestion_and_subject = chain_retrival_and_generate_output.invoke({'email': query})
    standaloneQuestion, subject = standaloneQuestion_and_subject['standaloneQuestion'], standaloneQuestion_and_subject['subject']
    
    chain_relevant_doc = retriever.invoke(standaloneQuestion)
    chain_final_answer = setupTempaltePromt | llm | StrOutputParser()
    body = chain_final_answer.invoke({'email': query, 'context': chain_relevant_doc})
    
    return {'subject': subject, 'body':body}
