import gradio as gr
from main import load_multiple_pdfs, chunk_pages, build_vector_index, ask, chat_history

# Holds the currently loaded PDF knowledge base
kb = {"index": None, "chunks": None}


def process_pdfs(files):
    if not files:
        return "Please upload at least one PDF first."

    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        pages = load_multiple_pdfs(paths)
        if not pages:
            return "No readable text found. Scanned PDFs (photos of pages) won't work."

        chunks = chunk_pages(pages, chunk_size=150, overlap=30)
        kb["index"] = build_vector_index(chunks)
        kb["chunks"] = chunks
        chat_history.clear()  # start a fresh conversation for the new PDFs

        return f"Ready! Loaded {len(paths)} PDF(s) with {len(chunks)} chunks. Ask your questions below."
    except Exception as e:
        return f"Error: {e}"


def respond(message, history):
    if kb["index"] is None:
        return "Please upload a PDF and click 'Process PDFs' first."

    answer, retrieved = ask(message, kb["index"], kb["chunks"], verbose=False)

    pages = sorted({(c["source"], c["page"]) for c in retrieved})
    sources = "\n".join(f"- {src}, page {pg}" for src, pg in pages)
    return f"{answer}\n\n**Sources:**\n{sources}" if sources else answer


with gr.Blocks(title="CampusGPT") as demo:
    gr.Markdown("# CampusGPT\nUpload your PDF(s), click **Process PDFs**, then ask questions.")

    files = gr.File(label="Upload PDF(s)", file_count="multiple", file_types=[".pdf"])
    process_btn = gr.Button("Process PDFs", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    process_btn.click(process_pdfs, inputs=files, outputs=status)

    gr.ChatInterface(fn=respond)

if __name__ == "__main__":
    demo.launch()