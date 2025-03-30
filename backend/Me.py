import pickle
import os
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

print('Imported')

def upload_Embeddings():
    """Function to process files from the 'uploads' directory and store embeddings."""
    
    # Define paths
    current_dir = Path(__file__).parent  # ✅ Ensures correct current directory
    uploads_path = current_dir / 'uploads'
    persist_directory = current_dir / 'db' / 'chroma_db'
    path_in_database = current_dir / "in_database.pkl"  # ✅ Use Path object

    print("Current Directory:", current_dir)
    print("Path to database:", path_in_database)

    # Check if the pickle file exists and is not empty
    if not path_in_database.exists() or path_in_database.stat().st_size == 0:
        print("File is empty or does not exist, initializing in_database as empty list.")
        in_database = []
    else:
        try:
            with open(path_in_database, 'rb') as file:
                in_database = pickle.load(file)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error reading pickle file: {e}, initializing as empty list.")
            in_database = []

    # Get all files in the uploads directory
    files = list(uploads_path.glob("*"))  # ✅ More robust than os.listdir()

    for f in files:
        if f.name not in in_database:
            ext = f.suffix.lower().lstrip('.')  # ✅ Get file extension safely

            # Load document based on file type
            if ext == 'txt':
                print(f'{f.name} ---> is a txt')
                loader = TextLoader(str(f))
            elif ext == 'pdf':
                print(f'{f.name} ---> is a pdf')
                loader = PyPDFLoader(str(f))
            elif ext == 'docx':
                print(f'{f.name} ---> is a docx')
                loader = Docx2txtLoader(str(f))
            elif ext == 'xlsx':
                print(f'{f.name} ---> is an xlsx')
                loader = UnstructuredExcelLoader(str(f))
            else:
                print(f'{f.name} ---> skipped')
                continue  

            document = loader.load()
            in_database.append(f.name)

            # Split the document into chunks
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            docs = text_splitter.split_documents(document)

            if not docs:
                print(f"Split into {len(docs)} chunks, so skipped")
                continue

            print(f"Split into {len(docs)} chunks")

            # Generate embeddings and store in Chroma
            embeddings = VertexAIEmbeddings(model="text-embedding-004")
            db = Chroma.from_documents(docs, embeddings, persist_directory=str(persist_directory))
            
            # Store data using pickle
            with open(path_in_database, 'wb') as file:
                pickle.dump(in_database, file)
                
            print(f'Done with embeddings for {f}')
        else:
            print(f'{f.name} file already in the database')

    print(in_database)
    return in_database

upload_Embeddings()
