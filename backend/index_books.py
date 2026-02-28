from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Loading book...")

loader = PyPDFLoader("data/medical_book.pdf")
docs = loader.load()

print("Splitting text...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
splits = splitter.split_documents(docs)

for i, doc in enumerate(splits):
    doc.metadata["page"] = doc.metadata.get("page", i)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="db"
)



print("✅ BOOK INDEXED SUCCESSFULLY")