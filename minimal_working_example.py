from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "DISCOStudy/PhenoGemini"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
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

_,  _, gene_token_tensor = detect_gene_token_space(tokenizer)

with torch.no_grad():
    text=wrap_prompt("""For a patient who presents with Recurrent epistaxis, Hepatomegaly, Splenomegaly, Hepatosplenomegaly, hepatomegaly, splenomegaly, prolonged bleeding.
We can find some patients with similar phenotypes, such as:
A patient presents with Recurrent sinusitis, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-959|>.
A patient presents with Chronic hemolysis, Increased indirect bilirubin, Splenomegaly, Hepatomegaly, Hepatosplenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-54658|>.
A patient presents with Splenomegaly, Hepatomegaly, Ascites, Placentomegaly, Hepatosplenomegaly, Cholestatic jaundice, Liver failure. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Splenomegaly, Hepatomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-4893|>.
A patient presents with Splenomegaly, Hepatomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-4893|>.
A patient presents with Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-10648|>.
A patient presents with Splenomegaly, Hepatomegaly, Cerebral hemorrhage. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-25|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Recurrent hemophagocytic lymphohistiocytosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Recurrent hemophagocytic lymphohistiocytosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Chronic hemolysis, Increased indirect bilirubin concentration, Reticulocytosis, Spherocytosis, Splenomegaly, Hepatomegaly, Hepatosplenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-6708|>.
A patient presents with Dyspnea, Hepatomegaly, Splenomegaly, Hepatosplenomegaly, Splenic lesion, Short stature, Epistaxis, Abnormality of thyroid-stimulating hormone concentration, Hypothyroidism. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-6609|>.
A patient presents with Pruritus, Hepatomegaly, Jaundice, Hepatic fibrosis, Abnormality of the liver, Gastrointestinal hemorrhage, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-5244|>.
A patient presents with Splenomegaly, Hepatomegaly, Thrombosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3717|>.
A patient presents with Splenomegaly, Hepatomegaly, Thrombosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3717|>.
A patient presents with Hepatomegaly, Splenomegaly, Recurrent infections, Positive direct Coombs test. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-356|>.
A patient presents with Hepatosplenomegaly, Cytopenia, Abdominal distension, Cough, Hepatomegaly, Splenomegaly, Elevated hepatic transaminase, Bone marrow hypercellularity, Erythroid hyperplasia, Pruritus, Epistaxis, Hypersplenism. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-5244|>.
A patient presents with Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-8647|>.
A patient presents with Jaundice, Abdominal distension, Developmental delay, Hyperbilirubinemia, Hepatosplenomegaly, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-10577|>.
A patient presents with Hemolytic anemia, Splenomegaly, Hepatomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Acute hepatic failure, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Recurrent hemophagocytic lymphohistiocytosis, Colitis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Hepatomegaly, Splenomegaly, Thrombocytopenia. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-7036|>.
A patient presents with Pallor, Recurrent vomiting, Fever, Hepatomegaly, Splenomegaly, Lipaemia, Anemia, Abdominal pain, Recurrent pancreatitis, Hepatosplenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-4023|>.
A patient presents with Hypertriglyceridemia, Diabetes mellitus, Hypertension, Hepatosplenomegaly, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-4023|>.
A patient presents with Anemia, Hepatomegaly, Splenomegaly, Cholelithiasis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Anemia, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3040|>.
A patient presents with Anemia, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3040|>.
A patient presents with Hepatomegaly, Splenomegaly, Cholelithiasis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-10483|>.
A patient presents with Jaundice, Pale stool, Hepatosplenomegaly, Splenomegaly, Abnormality of the hepatic portal vein. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-5244|>.
A patient presents with Jaundice, Pruritus, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-9414|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Epstein-Barr virus infection. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Hepatomegaly, Splenomegaly, Increased indirect bilirubin concentration. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Anemia, Hepatomegaly, Splenomegaly, Jaundice, Splenectomy. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3040|>.
A patient presents with Hepatomegaly, Cholestasis, Splenomegaly, Elevated hepatic transaminase, Prolonged prothrombin time. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-5265|>.
A patient presents with Splenomegaly, Hepatomegaly, Ascites, Thrombocytopenia, Anemia, Petechiae, Neonatal cholestatic jaundice, Hepatosplenomegaly, Liver failure, Hydropic placenta. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Hepatosplenomegaly, Cardiac hypertrophy, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-178|>.
A patient presents with Splenectomy, Pallor, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Anemia. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Anemia. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Epstein-Barr virus infection, Recurrent hemophagocytic lymphohistiocytosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Epstein-Barr virus infection, Recurrent hemophagocytic lymphohistiocytosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Epstein-Barr virus infection, Recurrent hemophagocytic lymphohistiocytosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Lipemia, Hepatosplenomegaly, Hepatomegaly, Splenomegaly, Hepatic steatosis, Hypertriglyceridemia. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-344|>.
A patient presents with Abdominal distension, Elevated hepatic transaminase, Hepatomegaly, Splenomegaly, Hepatosplenomegaly, Abnormal liver signal intensity on imaging, Hepatocellular enlargement, Abnormality of the liver, Portal inflammation, Bridging hepatic fibrosis, Periodic acid-Schiff stain positive material in hepatocytes, Thrombocytopenia, Elevated circulating bile acid concentration, Hypozincemia. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-2632|>.
A patient presents with Hepatomegaly, Splenomegaly, Hepatic steatosis, Portal fibrosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-540|>.
A patient presents with Hepatosplenomegaly, Splenomegaly, Viral infection. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-331|>.
A patient presents with Cholestasis, Icterus, Hepatomegaly, Splenomegaly, Hepatosplenomegaly, Jaundice, Bridging hepatic fibrosis, Micronodular cirrhosis, Regenerative hepatic nodules, Abnormality of the liver. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Seizure, Splenomegaly, Hepatomegaly, Hepatic steatosis. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-2538|>.
A patient presents with Fever, Lymphadenopathy, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-4068|>.
A patient presents with Severe clinical course, Anemia, Jaundice, Hepatomegaly, Splenomegaly, Frequent transfusions, Splenectomy. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-3039|>.
A patient presents with Splenomegaly, Hepatomegaly, Ascites, Thrombocytopenia, Anemia, Petechiae, Neonatal cholestatic jaundice, Hepatosplenomegaly, Liver failure, Preterm birth. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-257641|>.
A patient presents with Abnormal liver function, Hepatomegaly, Micronodular cirrhosis, Hepatosplenomegaly, Ascites, Hematemeis, Drowsiness, Jaundice, Abdominal distension, Spider angioma, Splenomegaly, Anemia, Esophageal varices. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-2632|>.
A patient presents with Pallor, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Pallor, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Pallor, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
A patient presents with Pallor, Jaundice, Hepatomegaly, Splenomegaly. This patient has a mutation in <|PhenoGemini-Special-Token-Entrez-ID-85344|>.
Based on all the information above, and my own general knowledge, this patient is likely to have a mutation in""", 
        tokenizer
    )  # IMPORTANT: make sure the prompt passed to the LLM does NOT contain any trailing spaces, newline characters, or other extraneous whitespace AT THE END
    
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Take logits for the last token only
    logits = outputs.logits[0, -1, :].to("cpu", dtype=torch.float32)
    # Mask non-gene tokens
    gene_token_tensor = gene_token_tensor.to(dtype=torch.long, device=logits.device)
    gene_logits = logits[gene_token_tensor]
    order = torch.argsort(gene_logits, descending=True)
    sorted_gene_ids = gene_token_tensor[order]
    sorted_gene_tok = tokenizer.convert_ids_to_tokens(sorted_gene_ids.tolist())
