# 🩺 AI Medical Assistant with RAG & Report Analysis

An AI-powered medical assistant that analyzes medical reports and answers health-related questions using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**.

This project enables users to upload medical reports and receive intelligent insights by combining document retrieval with generative AI.

---

## 🚀 Features

* 📄 **Medical Report Analysis** – Upload and analyze medical reports.
* 🤖 **AI Question Answering** – Ask health-related questions about the reports.
* 🔎 **RAG Pipeline** – Retrieves relevant medical knowledge before generating answers.
* 🧠 **LLM Integration** – Uses a large language model for intelligent responses.
* 📚 **Medical Knowledge Base** – Uses a medical book dataset for retrieval.

---

## 🏗️ Project Structure

```
medical-ai-assistant
│
├── backend/
│   ├── index_books.py       # Index medical books into vector database
│   ├── rag.py               # Retrieval-Augmented Generation pipeline
│   └── llm.py               # LLM integration
│
├── models/                  # ML / embedding models
├── app.py                   # Main application
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **LLMs**
* **Retrieval-Augmented Generation (RAG)**
* **Vector Database (ChromaDB)**
* **Embeddings**
* **Document Processing**

---

## 📦 Installation

Clone the repository:

```
git clone https://github.com/Dharmadhaashan/AI-Medical-Assistant-with-RAG-Report-Analysis.git
cd AI-Medical-Assistant-with-RAG-Report-Analysis
```

Create a virtual environment:

```
python -m venv venv
```

Activate it:

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your API key:

```
API_KEY=your_api_key_here
```

---

## 📚 Adding Medical Data

Place your medical documents or PDFs inside the `data/` directory before indexing.

Then run:

```
python backend/index_books.py
```

This will create the vector database for retrieval.

---

## ▶️ Running the Application

Start the application:

```
python app.py
```

You can then upload reports and interact with the AI assistant.

---

## 🧠 How It Works

1. Medical documents are converted into **embeddings**.
2. The embeddings are stored in a **vector database (ChromaDB)**.
3. When a user asks a question:

   * Relevant documents are retrieved.
   * The LLM generates an answer using the retrieved context.

This approach improves accuracy compared to normal LLM responses.

---

## 📌 Future Improvements

* Medical entity extraction
* Multi-document report analysis
* Improved medical knowledge base
* Web interface
* Deployment with Docker / Cloud

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only** and should **not be used as a substitute for professional medical advice**.

---

## 👨‍💻 Author

**Dharmadhaashan**

GitHub:
https://github.com/Dharmadhaashan

---

⭐ If you find this project useful, consider giving it a star!
