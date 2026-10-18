import gradio as gr
from main import load_multiple_pdfs, chunk_pages, build_vector_index, ask

# Build the knowledge base once, when the app starts
pdf_files = ["data/AI_data.pdf"]
pages_data = load_multiple_pdfs(pdf_files)
chunks = chunk_pages(pages_data, chunk_size=150, overlap=30)
vector_index = build_vector_index(chunks)


def respond(message, history):
    answer, retrieved = ask(message, vector_index, chunks, verbose=False)

    # list each source page once
    pages = sorted({(c["source"], c["page"]) for c in retrieved})
    sources = "\n".join(f"- {src}, page {pg}" for src, pg in pages)

    return f"{answer}\n\n**Sources:**\n{sources}" if sources else answer


demo = gr.ChatInterface(
    fn=respond,
    title="CampusGPT",
    description="Ask questions from your uploaded PDFs. Answers include source and page.",
    examples=["What is ML?", "What are the types of machine learning?", "What is RAG?"],
)

if __name__ == "__main__":
    demo.launch()