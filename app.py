import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()  # Load GOOGLE_API_KEY from .env file

from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os

# Initialize Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure uploads folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Validate API key on startup
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to a .env file or set it as an environment variable.")

# Initialize Gemini Chat model and Embeddings
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Handle PDF upload and question
@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return "No file part"

    file = request.files["resume"]
    question = request.form.get("question", "")

    if file.filename == "":
        return "No selected file"

    # Save PDF
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load PDF
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(docs)

    # Create FAISS vectorstore with Gemini embeddings
    vectorstore = FAISS.from_documents(docs_split, embedding=embeddings)

    # Query the document
    if question.strip() == "":
        result = "Please enter a question!"
    else:
        docs_with_answer = vectorstore.similarity_search(question, k=3)
        answer_text = "\n\n".join([d.page_content for d in docs_with_answer])
        prompt = f"Answer the question based on the document:\n\n{answer_text}\n\nQuestion: {question}"
        response = chat_model.invoke(prompt)
        result = response.content

    return render_template("index.html", result=result, question=question)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)