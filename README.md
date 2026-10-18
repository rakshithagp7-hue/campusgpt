# CampusGPT

A RAG-based chatbot that answers student questions from uploaded PDFs, with source and page citations.

## Flow
PDF -> extract text -> chunks -> Gemini embeddings -> FAISS index -> top-k search -> Gemini answer

## Run
1. `python -m venv venv` then `venv\Scripts\activate`
2. `pip install -r requirements.txt`
3. Create a `.env` file with `GOOGLE_API_KEY=your_key`
4. Put your PDF in `data/` and run `python main.py`
5. Run `python app.py` and open http://127.0.0.1:7860
6. Upload a PDF, click "Process PDFs", then ask questions
