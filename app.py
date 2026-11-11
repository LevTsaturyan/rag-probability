import gradio as gr

from rag.chunking import make_chunks
from rag.generator import load_generator
from rag.loaders import load_document
from rag.pipeline import answer, format_sources
from rag.retriever import Retriever, load_embedder
from rag import config

embed = load_embedder("cpu" if config.LLM_4BIT else None)  # видеопамять целиком отдаём LLM
generate = load_generator(config.LLM_MODEL, config.LLM_4BIT)


def index(files):
    if not files:
        raise gr.Error("Загрузите хотя бы один файл PDF или TXT")
    pages = [p for f in files for p in load_document(f)]
    chunks = make_chunks(pages)
    if not chunks:
        raise gr.Error("Не удалось извлечь текст. Возможно, это скан без текстового слоя")
    return Retriever(chunks, embed), f"Готово: файлов {len(files)}, фрагментов {len(chunks)}"


def ask(question, retriever):
    if retriever is None:
        raise gr.Error("Сначала загрузите документы")
    result = answer(question, retriever, generate)
    return result.text, format_sources(result.hits)


with gr.Blocks(title="Вопросы по конспектам") as demo:
    gr.Markdown("# Вопросы по конспектам\nЗагрузите PDF или TXT и задайте вопрос. Ответ строится только по этим файлам.")
    state = gr.State()
    files = gr.File(label="Документы", file_count="multiple", file_types=[".pdf", ".txt"], type="filepath")
    status = gr.Textbox(label="Статус", interactive=False)
    files.change(index, files, [state, status])
    question = gr.Textbox(label="Вопрос")
    button = gr.Button("Спросить", variant="primary")
    result = gr.Textbox(label="Ответ", lines=6)
    sources = gr.Textbox(label="Источники", lines=12)
    button.click(ask, [question, state], [result, sources])

if __name__ == "__main__":
    demo.launch()