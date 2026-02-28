from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load main medical book DB
def load_main_db():
    vectordb = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )
    return vectordb

# Process uploaded patient report
def process_uploaded_pdf(file):
    with open("temp_report.pdf", "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader("temp_report.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)

    report_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return report_db