from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

model_name = "DISCOStudy/PhenoGemini"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).eval()

WRAP_BREAK_1 = "\nWe can find some patients with similar phenotypes, such as:"
WRAP_BREAK_2 = "\nBased on all the information above, and my own general knowledge, this patient is likely to have a mutation in"
GENE_PREFIX = "<|PhenoGemini-Special-Token-Entrez-ID-"

def wrap_prompt(prompt: str, tokenizer) -> str:
    try:
        part1, remainder = prompt.split(WRAP_BREAK_1, 1)
        between, part3 = remainder.split(WRAP_BREAK_2, 1)
        part2 = WRAP_BREAK_1 + between
        part3 = WRAP_BREAK_2 + part3
        messages = [{"role": "user", "content": part1 + " We want to find the pathogenic gene mutation."}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + part2[1:] + "\n</think>" + part3 + " "
        return text
    except ValueError:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def detect_gene_token_space(tokenizer):
    vocab = tokenizer.get_vocab()
    gene_ids = sorted(idx for tok, idx in vocab.items() if tok.startswith(GENE_PREFIX))
    if not gene_ids:
        raise RuntimeError("No gene tokens found in tokenizer")
    base_cutoff = gene_ids[0]
    tensor_ids = torch.tensor(gene_ids, dtype=torch.long)
    return base_cutoff, len(gene_ids), tensor_ids


def infer(prompt, gene_tok):
    with torch.no_grad():
        text=wrap_prompt(prompt, tokenizer)
        inputs = tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        outputs = model(**inputs)
        # Take logits for the last token only
        logits = outputs.logits[0, -1, :].to("cpu", dtype=torch.float32)
        _,  _, gene_token_tensor = detect_gene_token_space(tokenizer)
        gene_token_tensor = gene_token_tensor.to(dtype=torch.long, device=logits.device)
        gene_logits = logits[gene_token_tensor]
        order = torch.argsort(gene_logits, descending=True)
        sorted_gene_ids = gene_token_tensor[order]
        sorted_gene_tok = tokenizer.convert_ids_to_tokens(sorted_gene_ids.tolist())
        
        return sorted_gene_tok.index(gene_tok) + 1, sorted_gene_tok


df = pd.read_excel("example_data.xlsx")
for idx, row in df.iterrows():
    prompt = row['Prompt']
    gene_tok = row['Gene Token']
    rank, sorted_gene_tok = infer(prompt, gene_tok)
    df.at[idx, 'Rank_new'] = rank if isinstance(rank, int) else -1
    df.at[idx, 'Sorted Gene Tokens'] = ", ".join(sorted_gene_tok)
    df.to_excel("example_output.xlsx", index=False)