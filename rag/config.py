EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CHARS = 600        # максимальная длина фрагмента
OVERLAP_SENTS = 1      # сколько предложений повторять между фрагментами
TOP_K = 5              # сколько фрагментов отдавать модели
THRESHOLD = 0.46       # ниже этого сходства отвечаем "нет в тексте"
MAX_NEW_TOKENS = 700
NO_ANSWER = "В загруженных документах этого нет."