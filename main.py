# CampusGPT - RAG chatbot that answers questions from your PDFs
# Flow: PDF -> text -> chunks -> embeddings -> FAISS -> top-k search -> Gemini answer

import os
import numpy as np
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# ---------- Setup ----------
load_dotenv()  # reads GOOGLE_API_KEY from the .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

GEN_MODEL_NAME = "gemini-3.8-flash"
EMBED_MODEL_NAME = "models/gemini-embedding-001"
gen_model = genai.GenerativeModel(GEN_MODEL_NAME)

chat_history = []  # conversation memory


# ---------- Module 1: PDF loader ----------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    source_name = os.path.basename(file_path)
    pages_data = []

    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages_data.append({"source": source_name, "page": i, "text": text})

    print(f"Loaded '{source_name}' -> {len(pages_data)} pages with text.")
    return pages_data


def load_multiple_pdfs(file_paths):
    all_pages = []
    for path in file_paths:
        all_pages.extend(load_pdf(path))
    return all_pages


# ---------- Module 2: Chunking ----------
def chunk_pages(pages_data, chunk_size=150, overlap=30):
    chunks = []
    chunk_id = 0

    for page in pages_data:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
            })
            chunk_id += 1

            if end >= len(words):
                break
            start = end - overlap

    print(f"Created {len(chunks)} chunks.")
    return chunks


# ---------- Module 3: Embeddings ----------
def embed_texts(texts, task_type="retrieval_document"):
    all_embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBED_MODEL_NAME,
            content=text,
            task_type=task_type,
        )
        all_embeddings.append(result["embedding"])
    return np.array(all_embeddings, dtype="float32")


# ---------- Module 4: FAISS vector index ----------
def build_vector_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, task_type="retrieval_document")

    faiss.normalize_L2(embeddings)  # so inner product = cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Vector index built: {index.ntotal} chunks, dimension {dim}.")
    return index


# ---------- Module 5: Retriever ----------
def retrieve_relevant_chunks(question, index, chunks, top_k=4):
    query_embedding = embed_texts([question], task_type="retrieval_query")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


# ---------- Module 6: Prompt builder ----------
def build_prompt(question, retrieved_chunks, chat_history=None):
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(
            f"[Source: {c['source']}, Page: {c['page']}]\n{c['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    history_text = ""
    if chat_history:
        for turn in chat_history[-6:]:  # last 3 exchanges
            history_text += f"{turn['role'].capitalize()}: {turn['content']}\n"

    prompt = f"""You are CampusGPT, an AI learning assistant for students.
Answer ONLY using the context below, which comes from the student's own
uploaded documents.

If the answer is not present in the context, reply exactly with:
"I couldn't find that information in the uploaded PDF(s)."

Cite the source and page for any fact you use, like this: (Source: file.pdf, Page: 3)

Previous conversation (if relevant):
{history_text}

Context:
{context}

Question:
{question}
"""
    return prompt


# ---------- Module 7: Answer generation ----------
def generate_answer(prompt):
    response = gen_model.generate_content(prompt)
    return response.text


# ---------- Module 9: Full pipeline ----------
def ask(question, index, chunks, top_k=4, verbose=True):
    retrieved = retrieve_relevant_chunks(question, index, chunks, top_k=top_k)

    if not retrieved:
        answer = "I couldn't find that information in the uploaded PDF(s)."
    else:
        prompt = build_prompt(question, retrieved, chat_history)
        answer = generate_answer(prompt)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    if verbose:
        print("\nA:", answer)
        if retrieved:
            print("\nSources used:")
            for c in retrieved:
                print(f"  - {c['source']} · page {c['page']} · similarity {c['score']:.3f}")

    return answer, retrieved


# ---------- Module 10: Run the chatbot ----------
if __name__ == "__main__":
    pdf_files = ["data/AI_data.pdf"]

    pages_data = load_multiple_pdfs(pdf_files)
    chunks = chunk_pages(pages_data, chunk_size=150, overlap=30)
    vector_index = build_vector_index(chunks)

    print("\nCampusGPT is ready! Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        ask(question, vector_index, chunks)
        print()