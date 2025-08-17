# 🦁 Repellica – NLP-based Plagiarism Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Backend-orange?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface\&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Repellica is an AI-powered plagiarism detection tool that uses a fine-tuned SmolLM2-135M model on the MIT Plagiarism Detection Dataset to identify textual similarities between PDF documents.**

</div>  

---

## 📌 Overview

**Repellica** leverages a **fine-tuned Large Language Model (SmolLM2-135M)** for **binary classification** of plagiarism.
The system compares two input PDF files and classifies them as either **Plagiarized (1)** or **Original (0)**.

It provides an **intuitive Streamlit interface** where users can upload documents, view extracted text, and get real-time plagiarism detection results.

---

## ✨ Key Features

| Feature                     | Description                                                                  |
| :-------------------------- | :--------------------------------------------------------------------------- |
| 📂 **PDF Upload**           | Upload two PDF files for comparison.                                         |
| 🔎 **Text Extraction**      | Extracts raw text from PDFs using **PyMuPDF (fitz)**.                        |
| 🧠 **Fine-Tuned Model**     | Custom-trained **SmolLM2-135M** on the **MIT Plagiarism Detection Dataset**. |
| ⚡ **Binary Classification** | Detects plagiarism with outputs: `1 = Plagiarized`, `0 = Non-Plagiarized`.   |
| 📊 **Performance Metrics**  | Evaluated using **Accuracy, F1 Score, Recall**.                              |
| 🖥️ **Streamlit UI**        | Simple, interactive, and user-friendly interface.                            |

---

## 📂 Project Structure

```plaintext
Repellica-Plagiarism-Detector/
├── app.py         # Streamlit application
├── plagarism.ipynb
├── model/                  # Fine-tuned model + tokenizer files
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 📦 Dataset

We used the **MIT Plagiarism Detection Dataset**, which provides **sentence pairs labeled as plagiarized or non-plagiarized**.

* **Train Split**: 70%
* **Validation Split**: 10%
* **Test Split**: 20%

This dataset is well-suited for **sentence-level similarity detection**, making it ideal for plagiarism classification tasks.

---

## 🧠 Technical Details

* **Base Model**: HuggingFaceTB/SmolLM2-135M-Instruct
* **Architecture**: Modified for **sequence classification (2 labels)**
* **Optimizer**: AdamW (lr = 2e-5)
* **Loss Function**: Cross-Entropy Loss
* **Batch Size**: 16
* **Epochs**: 2
* **Padding**: Custom padding token for SmolLM compatibility
* **Metrics**: Accuracy, F1 Score, Recall

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* Torch
* Transformers (Hugging Face)
* Streamlit
* PyMuPDF (fitz)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Repellica-Plagiarism-Detector.git
cd Repellica-Plagiarism-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your fine-tuned model files inside 'model/' directory
```

### Running the App

```bash
streamlit run app.py
```

Open your browser at **[http://localhost:8501](http://localhost:8501)** and start uploading PDFs for plagiarism detection.

---

## 📊 Example Output

* **Upload**: Two PDF documents.
<img width="2137" height="419" alt="image" src="https://github.com/user-attachments/assets/d722dda1-a224-423c-a6a7-ee6a7bffb7df" />

* **Extraction**: Text is extracted and preprocessed.
<img width="2138" height="471" alt="image" src="https://github.com/user-attachments/assets/e1a4e7d3-a5db-463a-87dc-50e1c7157c31" />

* **Prediction**:
<img width="2154" height="519" alt="image" src="https://github.com/user-attachments/assets/f28e77b3-89f1-49a2-aeb9-88039d830029" />

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

<div align="center">

### 🔹 [A. Jayavanth](https://github.com/jayavanth18)

[![GitHub](https://img.shields.io/badge/GitHub-jayavanth18-181717?style=for-the-badge\&logo=github)](https://github.com/jayavanth18)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-jayavanth18-0A66C2?style=for-the-badge\&logo=linkedin)](https://www.linkedin.com/in/jayavanth18/)

---

⭐ If you found this repository useful, don’t forget to **star it**!

</div>  

---
