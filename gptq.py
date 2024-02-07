import modal
from modal import Image, Volume, Stub, gpu
from pydantic import BaseModel

stub = Stub('quantize')
IMAGE_MODEL_DIR = "/model"
MODEL_ID = 'deepseek-ai/deepseek-math-7b-instruct'

def download_model_and_dataset():
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/MathInstruct")
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=IMAGE_MODEL_DIR)


gpu_image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    )
    .env({"BUILD_CUDA_EXT" : "0"})
    .pip_install(
        "torch",
        "hf_transfer",
        "numpy", 
        "transformers",
        "huggingface-hub",
        "datasets",
        "auto-gptq",
        "optimum",
    )
    #.run_commands("pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model_and_dataset)
)

with gpu_image.imports():
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
    import torch
    import os
    from datasets import load_dataset
    import os
    from huggingface_hub import login

@stub.function(gpu=gpu.A100(), image=gpu_image, secrets=[modal.Secret.from_name("HUGGINGFACE_API_KEY")], timeout=60*50)
def quantize_and_upload_model():
    login(token = os.environ["HF_TOKEN"])
    dataset = load_dataset("TIGER-Lab/MathInstruct")

    print("CHECKING\n\nis cuda available:", torch.cuda.is_available())
    train_data = dataset['train']
    df = train_data.to_pandas()
    df_10 = df[:10].copy()  

    df_10['merged'] = df_10['instruction'] + df_10['output'] # merge the instruction and output

    quantization_config = GPTQConfig(
        bits=4,
        group_size=128,
        dataset=df_10['merged'].to_list(),
        desc_act=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        IMAGE_MODEL_DIR, 
        quantization_config=quantization_config,  
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    repo_id = MODEL_ID.split("/")[-1] + "-GPTQ"
    model.push_to_hub(repo_id = repo_id)


@stub.local_entrypoint()
def main():
    quantize_and_upload_model.remote()
