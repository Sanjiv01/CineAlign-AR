from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # 也可用 float16
    device_map="auto"
)

def query_movie(movie_title):
    prompt = f"""<s>[INST] Answer the following:
1. Is the movie "{movie_title}" an animated film or a live-action film?
2. List the top 10 main actors or voice actors of the movie. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=300)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

# Example
print(query_movie("Toy Story (1995)"))
print(query_movie("The Dark Knight (2008)"))

