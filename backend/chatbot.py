# Ensure your VertexAI credentials are configured
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI                                             
from langchain_google_vertexai import VertexAIEmbeddings
import os
embeddings = VertexAIEmbeddings(model="text-embedding-004")
from langchain_chroma import Chroma



from langchain_chroma import Chroma
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

embeddings = VertexAIEmbeddings(model="text-embedding-004")
current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db')
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=persistent_directory,  # Where to save data locally, remove if not necessary
)



import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.checkpoint.memory import MemorySaver
# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# all_splits = text_splitter.split_documents(docs)

# # Index chunks
# _ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
# print(response["context"])
