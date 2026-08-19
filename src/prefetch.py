import os, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
MODELS = ["MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
          "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
          "joeddav/xlm-roberta-large-xnli",
          "Davlan/afro-xlmr-base"]
for m in MODELS:
    for attempt in range(4):
        try:
            p = snapshot_download(m, allow_patterns=["*.json","*.txt","*.model","*.safetensors"],
                                  ignore_patterns=["onnx/*"], max_workers=4)
            print(f"OK {m}", flush=True); break
        except Exception as e:
            print(f"retry {attempt+1} {m}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        print(f"FAILED {m}", flush=True)
print("PREFETCH_DONE", flush=True)
