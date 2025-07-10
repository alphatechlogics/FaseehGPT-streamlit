# FaseehGPT-streamlit

A modern Streamlit web app for generating high-quality Arabic text using the [FaseehGPT](https://huggingface.co/alphatechlogics/FaseehGPT) model by Alphatechlogics.

![](https://raw.github.com/alphatechlogics/FaseehGPT-streamlit/5306327037e921d45ed454c4967217b8dc4a8165/screenshot-07_10%2C%2002_47_29%20PM.jpg)

---

## 🚀 Features
- **Arabic text generation** powered by FaseehGPT (decoder-only GPT-style transformer)
- Adjustable generation parameters: max tokens, temperature, top-k, top-p
- Clean, user-friendly UI with documentation and sample outputs
- Model details, usage, limitations, and citation info included

---

## 🧠 About FaseehGPT
- **Type:** Decoder-only Transformer (GPT-style)
- **Language:** Arabic
- **Parameters:** ~70.7M | **Layers:** 12 | **Embedding Dim:** 512
- **Tokenizer:** asafaya/bert-base-arabic
- **Trained on:** [arbml/Arabic_News](https://huggingface.co/datasets/arbml/Arabic_News), [arbml/Arabic_Literature](https://huggingface.co/datasets/arbml/Arabic_Literature)
- **License:** Apache 2.0

For more, see the [official model card](https://huggingface.co/alphatechlogics/FaseehGPT).

---

## 💡 Usage

1. Clone this repo and install requirements:
   ```bash
   git clone https://github.com/alphatechlogics/FaseehGPT-streamlit.git
   cd FaseehGPT-streamlit
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to the provided local URL.

---

## 📝 Example

- **Prompt:** `اللغة العربية`
- **Output:**
  > "اللغة العربية اقرب ويح الي كما ذلك هذه البيان شعره قاله الاستاذر من وتج معهم..."

---

## ⚙️ Parameters
- `max_new_tokens`: Max tokens to generate (e.g., 100)
- `temperature`: Controls randomness (default: 1.0)
- `top_k`: Limits sampling to top-k tokens (default: 50)
- `top_p`: Nucleus sampling threshold (default: 0.9)

---

## ⚠️ Limitations & Ethics
- Generated text may lack full coherence or contain errors due to limited training.
- Only supports Arabic; other languages untested.
- May reflect biases from source data.
- For research/non-commercial use; always validate outputs.

---

## 📖 Citation
```bibtex
@misc{faseehgpt2025,
  title     = {FaseehGPT: An Arabic Language Model},
  author    = {Rohma, Ahsan Umar},
  year      = {2025},
  url       = {https://huggingface.co/alphatechlogics/FaseehGPT}
}
```

---

## 🔗 Useful Links
- [FaseehGPT on Hugging Face](https://huggingface.co/alphatechlogics/FaseehGPT)
- [GitHub Source Code](https://github.com/alphatechlogics/FaseehGPT)
- [Alphatechlogics LinkedIn](https://www.linkedin.com/company/alphatechlogics/)
