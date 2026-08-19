import os, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
for m in ["joeddav/xlm-roberta-large-xnli", "Davlan/afro-xlmr-base"]:
    for attempt in range(6):
        try:
            snapshot_download(m, allow_patterns=["*.json","*.txt","*.model","*.safetensors"],
                              ignore_patterns=["onnx/*"], max_workers=2,
                              etag_timeout=60, resume_download=True)
            print(f"OK {m}", flush=True); break
        except Exception as e:
            print(f"retry {attempt+1} {m}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            time.sleep(10)
    else:
        print(f"FAILED {m}", flush=True)
print("PREFETCH2_DONE", flush=True)
