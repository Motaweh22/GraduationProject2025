# 📚 Document Visual Question Answering Using Retrieval-Augmented Generation (DocVQA-RAG)

We’re proud to announce the successful completion of our graduation project as a team of **Computer Science & Artificial Intelligence students**.
Our work tackles one of the most challenging problems in multimodal AI: **Visual Question Answering (VQA)** over real-world documents.

---

## 🧠 Problem Overview

While VQA over natural images is well explored, **Document VQA (DocVQA)** is far more complex. Documents such as research papers, forms, financial reports, and policy documents contain:

* Long multi-page structures
* Dense and domain-specific text
* Tables, figures, and charts
* Complex visual + layout cues

This makes reasoning substantially harder than standard image-based VQA.

---

## 🚀 Our Solution — M3DocRAG Pipeline

To address these challenges, we built a **Retrieval-Augmented Generation pipeline for DocVQA**, inspired by the **M3DocRAG architecture**.

After extensive experimentation, we selected the following components for optimal accuracy and scalability:

### 🔍 **ColPaLi Retriever**

A state-of-the-art **multi-modal retriever** that leverages:

* Visual features
* Textual features
* Layout structure

### 📦 **FAISS Indexing**

Used for:

* Fast vector similarity search
* Efficient multi-document & multi-page retrieval
* Scalable nearest-neighbor lookup

### 🧠 **Qwen2.5 Vision-Language Model**

Responsible for:

* Answer generation
* Cross-page reasoning
* Understanding retrieved evidence

Our pipeline supports **multi-page documents**, **cross-page reasoning**, and generalizes well across multiple domains.

---

## 🧩 Unified Dataset Schema

To support training and evaluation on diverse datasets, we developed a **Unified Dataset Schema** based on a `UnifiedEntry` object.
Each entry captures:

* Question type, domain, and metadata tags
* Document structure & source
* Page-level evidence (text spans, tables, figures, charts)
* Multiple answer variants + rationale
* Flags for low-quality, uncertain, or inferred annotations

This schema enabled:

* Consistent benchmarking
* Error detection
* Structured validation
* Cross-dataset unification

---

## 📚 Datasets Used

We unified and evaluated across **six major DocVQA benchmarks**:

* **MP-DocVQA**
* **DUDE**
* **MMLongBench-Doc**
* **ArxivQA**
* **TAT-DQA**
* **SlideVQA**

These datasets span scientific papers, slides, financial documents, long-form PDFs, and more.

---

## 🔗 Our Datasets & Models on Hugging Face

You can explore all our unified datasets, trained models, schema documentation, and processed files here:

👉 **Hugging Face Organization:**
[https://huggingface.co/AHS-uni](https://huggingface.co/AHS-uni)

---

## 📊 Evaluation Summary

### 🔎 **Retrieval Performance (ColPaLi + FAISS)**

| Metric       | Result                                                             |
| ------------ | ------------------------------------------------------------------ |
| **Recall@5** | Up to **0.88** (MP-DocVQA), consistently **>0.60** across datasets |
| **nDCG@10**  | **0.75 – 0.92**                                                    |
| **Latency**  | ~**0.04s** per retrieval query                                     |

---

### 🧠 **Answer Generation Performance (Qwen2.5)**

| Metric               | Result                                                   |
| -------------------- | -------------------------------------------------------- |
| **ANLS**             | Up to **0.845** (MP-DocVQA) and **0.630** (MM-LongBench) |
| **Generalization**   | Strong performance on unseen domains                     |
| **Generation Speed** | ~**0.15 – 0.51s** per question                           |

---

## ✅ Key Takeaways

Our DocVQA-RAG pipeline demonstrates strong performance across:

* **Accuracy**
* **Speed**
* **Cross-domain robustness**
* **Support for multi-page & multi-modal documents**

This work shows that retrieval-augmented VLMs are a **powerful approach for real-world document reasoning**.

## 🎥 Demo Video

### ▶️ Detailed Walkthrough


https://github.com/user-attachments/assets/728e6723-9ffa-49d6-93d2-fedbdccb14a2
