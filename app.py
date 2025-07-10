import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set page config
st.set_page_config(page_title="FaseehGPT Text Generator", layout="centered")

# --- Documentation and About Section ---
st.title("🧠 FaseehGPT - Arabic Text Generation")
st.markdown("""
Welcome to **FaseehGPT**! This app lets you generate high-quality Arabic text using the FaseehGPT model.\

**How to use:**
1. Enter your prompt in Arabic below.
2. Adjust the generation settings in the sidebar for more control.
3. Click **Generate Text** to see the model's output.

---
**About FaseehGPT:**
- FaseehGPT is a state-of-the-art Arabic language model by [Alphatechlogics](https://huggingface.co/alphatechlogics/FaseehGPT).
- It is designed for creative, coherent, and contextually accurate Arabic text generation.
---
""")

# --- Sidebar: Parameter Settings Only ---
st.sidebar.header("⚙️ Generation Settings")
max_tokens = st.sidebar.slider("Max New Tokens", 10, 300, 100, help="Maximum number of tokens to generate in the output.")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 1.0, help="Controls randomness: higher values make output more creative.")
top_k = st.sidebar.slider(
    "Top-K",
    10, 100, 50,
    help="Limits sampling to the top K most likely next words. Higher K = more variety, lower K = more focused output."
)
top_p = st.sidebar.slider(
    "Top-P (Nucleus Sampling)",
    0.1, 1.0, 0.9,
    help="Chooses from the smallest set of words whose probabilities add up to P. Lower P = safer, higher P = more diverse."
)

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("alphatechlogics/FaseehGPT", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("alphatechlogics/FaseehGPT", trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_model()

# Prompt input
st.markdown("#### ✍️ Enter your Arabic prompt:")
prompt = st.text_area("Arabic Prompt", value="السلام عليكم", height=150, help="Type your prompt in Arabic to generate text.")

# Generate button
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
generate_btn = st.button("🚀 Generate Text", help="Click to generate text based on your prompt and settings.")
st.markdown("</div>", unsafe_allow_html=True)

if generate_btn:
    with st.spinner("Generating..."):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.markdown("### 📝 Generated Text:")
        st.success(output_text)

# --- Model Details and Documentation ---
with st.expander("ℹ️ About FaseehGPT (Click to expand)", expanded=False):
    st.markdown("""
**FaseehGPT** is a GPT-style Arabic language model by [Alphatechlogics](https://huggingface.co/alphatechlogics/FaseehGPT), designed for creative, coherent, and contextually accurate Arabic text generation.

**Model Details:**
- **Type:** Decoder-only Transformer (GPT-style)
- **Version:** 1.1 (July 10, 2025)
- **Developers:** [Ahsan Umar](https://huggingface.co/codewithdark)
- **License:** Apache 2.0
- **Framework:** PyTorch, Hugging Face Transformers
- **Language:** Arabic
- **Parameters:** ~70.7M
- **Layers:** 12 | **Attention Heads:** 8 | **Embedding Dim:** 512
- **Tokenizer:** asafaya/bert-base-arabic
- **Max Sequence Length:** 512

**Training:**
- Trained on 50,000 samples from [arbml/Arabic_News](https://huggingface.co/datasets/arbml/Arabic_News) and [arbml/Arabic_Literature](https://huggingface.co/datasets/arbml/Arabic_Literature)
- 20 epochs, AdamW optimizer, P100 GPU (Kaggle)

**Special Features:**
- Supports top-k and top-p sampling
- Weight tying between input/output embeddings
- Optimized for resource-constrained environments (e.g., Colab)
    """)

with st.expander("💡 How it Works & Usage", expanded=False):
    st.markdown("""
**How to Use:**
- Enter an Arabic prompt and adjust generation settings (see sidebar).
- Click **Generate Text** to get a continuation.

**Sample Prompts & Outputs:**
- Prompt: `اللغة العربية`
  > "اللغة العربية اقرب ويح الي كما ذلك هذه البيان شعره قاله الاستاذر من وتج معهم..."
- Prompt: `كان يا مكان في قديم الزمان`
  > "كان يا مكان في قديم الزمان الانسان الانسان بعض لا انر لقد الانسان ذلك انلاركارك..."

**Parameters:**
- `max_new_tokens`: Max tokens to generate (e.g., 100)
- `temperature`: Controls randomness (default: 1.0)
- `top_k`: Limits sampling to top-k tokens (default: 50)
- `top_p`: Nucleus sampling threshold (default: 0.9)

**Example Code:**
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("alphatechlogics/FaseehGPT", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("alphatechlogics/FaseehGPT")
prompt = "السلام عليكم"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```
    """)

with st.expander("⚠️ Limitations & Ethics", expanded=False):
    st.markdown("""
- Generated text may lack full coherence or contain errors due to limited training.
- Only supports Arabic; other languages untested.
- May reflect biases from source data.
- For research/non-commercial use; always validate outputs.
    """)

with st.expander("📖 Citation", expanded=False):
    st.code("""@misc{faseehgpt2025,
  title     = {FaseehGPT: An Arabic Language Model},
  author    = {Rohma, Ahsan Umar},
  year      = {2025},
  url       = {https://huggingface.co/alphatechlogics/FaseehGPT}
}""", language="bibtex")
    st.markdown("For academic or research use, please cite as above.")

st.markdown("---")
st.caption("Powered by FaseehGPT - alphatechlogics • [Hugging Face 🤗](https://huggingface.co/alphatechlogics/FaseehGPT)")
