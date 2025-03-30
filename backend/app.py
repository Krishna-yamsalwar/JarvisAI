import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from flask import Flask, jsonify, request, render_template,redirect,session,flash
from flask_cors import CORS
from flask_mail import Mail, Message
import imaplib
import email
from email.utils import parsedate_to_datetime
import re
from Me import upload_Embeddings
from mailReAndPromt import process_query
from dotenv import load_dotenv
import logging
from werkzeug.utils import secure_filename  # ✅ Import this!
from langchain_google_genai import ChatGoogleGenerativeAI                     
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredExcelLoader,Docx2txtLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import json
# from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from chatbot2 import process_chat
persist_directory = os.getenv("CHROMA_DB_PATH", "db/chroma_db")


# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
CORS(app)


# Email credentials (Replace with your credentials)
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "parthakadam2007@gmail.com"
APP_PASSWORD = "cusy oiub nwvd yxsy"

# Flask-Mail Configuration (Replace with your credentials)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'parthakadam2007@gmail.com'
app.config['MAIL_PASSWORD'] = 'cusy oiub nwvd yxsy'
app.config['MAIL_DEFAULT_SENDER'] = 'parthakadam2007@gmail.com'


mail = Mail(app)



# Function to fetch emails
def fetch_emails():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
        mail.select("inbox")

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()
        email_list = []
        count = 1

        for email_id in email_ids[-10:]:  # Fetch last 10 emails
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            parsed_date = parsedate_to_datetime(msg["Date"])

            sender_raw = msg["From"]
            sender_email = re.findall(r'<([^<>]+)>', sender_raw)
            sender_email = sender_email[0] if sender_email else sender_raw

            email_data = {
                'id': count,
                'email_id': f'{email_id}',
                "from": msg["From"],
                "subject": msg["Subject"],
                "body": extract_body(msg),
                "date": parsed_date.strftime("%Y-%m-%d"),
                "time": parsed_date.strftime("%H:%M:%S"),
                "sender_email": sender_email,
            }
            count += 1
            email_list.append(email_data)

        mail.close()
        mail.logout()
        return email_list

    except Exception as e:
        return [{"error": str(e)}]

# Function to extract email body
def extract_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()
    return "No message body"


@app.route("/")
def default():
    return render_template('startpage.html')

@app.route("/chatbot", methods=["POST"])
def chatbot():
    if request.method == "POST":
        data = request.get_json()
        print("data")
        print( data)
        query = data.get("message")
        
        print("query:",query)
        response = process_chat(query)
        return jsonify({"response": response['answer']})
    else:
        return jsonify({"error": "Invalid request method"}), 405


@app.route("/mainpage")
def mainpage():
    return render_template('mainpage.html')

@app.route('/loginPage')
def get_login():
    return render_template('loginPage.html')

@app.route("/interface")
def get_interface():
    return render_template('index.html')


# Route to fetch emails
@app.route("/emails", methods=["GET"])
def get_emails():
    return jsonify(fetch_emails())


@app.route('/generate_reply',methods=['POST',"GET"])
def generate_reply():
    
    data =request.get_json()
    response = process_query(data)
    body = response['body']
    subject = response['subject']
    return jsonify( {"body":body,"subject":subject})                                          
    
  
# Route to send an email
@app.route('/send_mail', methods=['POST'])
def send_mail():
    data = request.get_json()
    recipient = data['email']
    body = data['body']
    sub = data['sub']
    msg = Message(subject=sub, recipients=[recipient], body=body)
    try:
        mail.send(msg)
        return jsonify({"success": True, "message": "Email sent successfully!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# UPLOAD_FOLDER = r"C://Users//DELL//Desktop//easyMail//backend//uploads"  
UPLOAD_FOLDER =os.path.join(os.getcwd(),'uploads')
print('upl----->',UPLOAD_FOLDER )

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','xlsx','csv','docx'}

# 🔹 Create the folder if it doesn’t exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Required for flash messages

# 🔹 Set up logging
logging.basicConfig(level=logging.DEBUG)


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/uploadFiles', methods=['GET', 'POST'])
def upload_file():
    print((request))
    """Handle multiple file uploads"""
    if request.method == 'POST':
        print('hit post request')
        if 'files' not in request.files:
            flash('Error: No files found in request.')
            logging.error("No files found in request.")
            return redirect(request.url)

        files = request.files.getlist('files')  # ✅ Get multiple files
        uploaded_files = []

        for file in files:
            if file.filename == '':
                flash('Error: One or more files have no filename.')
                continue

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)  # ✅ Use secure_filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)  # ✅ Save file
                uploaded_files.append(filename)
                logging.info(f"File '{filename}' uploaded successfully.\n")

        if uploaded_files:
            flash(f'Success! Uploaded files: {", ".join(uploaded_files)}')
            upload_Embeddings()
            flash('Embeddings generated successfully.') 
            
        else:
            flash('Error: No valid files uploaded.')

        return redirect(request.url)

    return render_template("uploadfilepage.html")


# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
