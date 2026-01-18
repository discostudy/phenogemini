# PhenoGemini-LLM Inference Code
## Quick Start
### Step 1: Install dependencies
- transformers==4.56.2
- torch==2.8.0
- pandas==2.3.2

You can install them with:
```bash
pip install transformers==4.56.2 torch==2.8.0 pandas==2.3.2
```

### Step 2: Download model weights
The PhenoGemini LLM weights are hosted on Hugging Face:
https://huggingface.co/DISCOStudy/PhenoGemini

They are usually downloaded automatically when the model is loaded. If you need
to download them manually (the files are large), use the HF CLI:
https://huggingface.co/docs/hub/models-downloading

### Step 3: Run the example
We provide an example dataset `example_data.xlsx` with 20 anonymized samples and pre-built prompts
from the real-world cohort 1 used in the study. The prompts already include "twin
patients". Run `run_on_example_data.py` to generate the results and inspect the ranking 
of the true causal genes.

## Run PhenoGemini on your own data
### Step 1: Build the prompt
Use PhenoGemini Atlas (https://phenogemini.org/) to create a prompt that
includes "twin patients". After you input the patient's phenotypes, the Atlas
retrieves the most similar patients from the literature and appends them to the
final prompt.

### Step 2: Inference
Use `minimal_working_example.py` as a reference to build your own inference
pipeline.

## Inference hardware and latency
Inference is performed on 2x A100 80GB GPUs. The estimated per-sample inference
time is about 40s, including loading sharded weights.

## Contact
For questions about individual modules or accessing additional data, please
email dr.wunan@pumch.cn, jeffchenmed@gmail.com, or caijh09@gmail.com.
