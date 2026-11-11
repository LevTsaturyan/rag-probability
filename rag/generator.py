from rag import config

SYSTEM = ("Ты отвечаешь на вопросы только по фрагментам из контекста.\n"
          "Правила:\n"
          "1. Используй только факты из фрагментов. Ничего не добавляй от себя.\n"
          "2. После каждого предложения ставь номер фрагмента, из которого взят факт, в квадратных скобках.\n"
          "3. Если ответа во фрагментах нет, напиши: «В загруженных документах этого нет.»\n"
          "Пример ответа:\n"
          "Бустинг строит модели последовательно [2]. Каждая новая модель исправляет ошибки предыдущих [2][4].")


def build_messages(question, hits):
    context = "\n\n".join(f"[{n}] {h.chunk.text}" for n, h in enumerate(hits, 1))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}]


def load_generator(model_name=config.LLM_MODEL, load_4bit=False):
    """Возвращает функцию: (вопрос, фрагменты) -> текст ответа."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16) if load_4bit else None
    tok = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto", quantization_config=quant)

    def generate(question, hits):
        prompt = tok.apply_chat_template(build_messages(question, hits), tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(llm.device)
        with torch.inference_mode():
            out = llm.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate