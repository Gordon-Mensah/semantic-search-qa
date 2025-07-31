# 📚 Semantic Search & QA Pipeline with DPR, FAISS, and GPT-2

This project implements a semantic document search and question-answering (QA) pipeline using state-of-the-art NLP models from Hugging Face. It combines **Dense Passage Retrieval (DPR)**, **FAISS**, and **GPT-2** to enable natural language queries over unstructured text documents — with optional visualization using **t-SNE**.

---
 
## 🔍 Use Case

Given a long unstructured policy document, the system allows users to:
- 🔎 Ask natural language questions  
- 🤖 Retrieve the most semantically relevant paragraphs using DPR + FAISS  
- 🧠 Generate a human-readable answer using GPT-2  

---

## 🛠️ Tech Stack

| Component        | Purpose |
|------------------|---------|
| **DPR** | Embedding questions & contexts |
| **FAISS** | Fast similarity search over embeddings |
| **GPT-2** | Generates coherent answers from retrieved context |
| **t-SNE (3D)** | Visualizes embedding clusters |
| **Transformers / Datasets** | NLP tools from Hugging Face |

---

## 📦 Installation

Install all dependencies using:

```bash
pip install --quiet faiss-cpu transformers datasets torch wget
```

To suppress model/tokenizer warnings:

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()
```

---

## 🚀 How It Works

### 1. Load and Preprocess the Document

```python
filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
wget.download(url, out=filename)

def read_and_split_text(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        paragraphs = [p.strip() for p in file.read().split('\n') if p.strip()]
    return paragraphs

paragraphs = read_and_split_text(filename)
```

### 2. Encode Contexts using DPR

```python
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

def encode_contexts(text_list):
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()

context_embeddings = encode_contexts(paragraphs)
```

### 3. Create a FAISS Index

```python
import faiss

embedding_dim = context_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(context_embeddings.astype('float32'))
```

### 4. Encode Questions

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def search_relevant_contexts(question, k=5):
    question_inputs = question_tokenizer(question, return_tensors='pt')
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()
    D, I = index.search(question_embedding, k)
    return D, I
```

### 5. Generate Answers with GPT-2

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.generation_config.pad_token_id = tokenizer.pad_token_id

def generate_answer(question, contexts, max_new_tokens=100):
    input_text = question + ' ' + ' '.join(contexts)
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
    summary_ids = model.generate(inputs['input_ids'], max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```

---

## 🧪 Example

```python
question = "What is the mobile phone policy?"
_, indices = search_relevant_contexts(question)
top_contexts = [paragraphs[i] for i in indices[0]]

answer = generate_answer(question, top_contexts)
print("Answer:", answer)
```

---

## 📊 Visualization (Optional)

Use 3D t-SNE to visualize how different paragraphs cluster in vector space:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def tsne_plot(data):
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, data.shape[0] - 1))
    data_3d = tsne.fit_transform(data)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20(np.linspace(0, 1, len(data_3d)))

    for idx, point in enumerate(data_3d):
        ax.scatter(*point, label=str(idx), color=colors[idx])

    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    plt.title('3D t-SNE Visualization')
    plt.legend(title='Paragraph Index')
    plt.show()
```

---

## 📘 References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [DPR: Dense Passage Retrieval](https://github.com/facebookresearch/DPR)

---

## 📚 Learning Journey

This project is part of my learning journey through **Coursera**, where I’m deepening my skills in machine learning, natural language processing, and AI engineering. I'm applying what I learn by building practical tools and pipelines like this one.

---

## 🤝 Acknowledgments

Thanks to the open-source community and platforms like Hugging Face, Coursera, and GitHub for making high-quality learning and experimentation accessible to all.
