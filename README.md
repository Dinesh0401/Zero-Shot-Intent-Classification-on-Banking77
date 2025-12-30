This project evaluates the capability of Large Language Models (LLMs) to perform **intent classification** in a **zero-shot setting** using the **Banking77 dataset**.

The work is structured into **two sub-tasks**:

1. Validating zero-shot intent prediction for individual queries.
2. Benchmarking accuracy at scale using a fixed evaluation set.

No model training or fine-tuning is performed — the entire pipeline relies on **prompt-based inference only**.

---

## 🎯 Objectives

* Validate LLMs as intent classifiers using **prompt-only inference**
* Ensure strict prompt separation:

  * Intent labels in **system prompt**
  * User query only in **user prompt**
* Benchmark performance on a **fine-grained intent dataset**
* Compare performance between **GPT-5.2** and **GPT-5-Mini**

---
 
## 📊 Dataset

**Banking77**

* Domain: Banking & customer support
* Total intents: **77**
* Total samples: ~13,000
* Evaluation subset: **500 randomly sampled queries**

Dataset CSV schema:

```text
text, category
```

---

## ✅ Task 1 — Banking77 (COMPLETED)

Task 1 consists of **two sub-tasks**.

---

### 🔹 Task 1A — Zero-shot Intent Classification (Single Prediction)

#### What was done

* Used the **Banking77 dataset** with 77 intent labels
* Sent **one user query at a time** to the LLM
* Provided **all intent labels in the system prompt**
* Enforced the model to return **exactly one label**
* Verified correct intent prediction for individual queries

#### Purpose

* Validate prompt design correctness
* Confirm LLM suitability as an intent classifier

**Status:** ✅ Completed

---

### 🔹 Task 1B — Zero-shot Evaluation at Scale (Accuracy Benchmark)

#### What was done

* Evaluated the same prompt setup on **500 Banking77 samples**
* Compared predictions with **ground-truth labels**
* Tested two LLMs:

  * **GPT-5.2**
  * **GPT-5-Mini**
* Computed **exact-match accuracy**

#### Results

```
GPT-5.2   → 62.00%
GPT-5-Mini → 60.40%
```

#### Purpose

* Establish a quantitative zero-shot baseline
* Compare model performance under identical conditions

**Status:** ✅ Completed

---

## 📐 Evaluation Metric

**Exact-Match Accuracy**

[
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Samples}} \times 100
]

* No training or fine-tuning
* No top-k or partial credit
* Strict label matching

---

## 🔍 Key Observations

* GPT-5.2 marginally outperforms GPT-5-Mini
* Accuracy reflects the difficulty of:

  * Zero-shot classification
  * Fine-grained intent sets (77 labels)
  * Semantic overlap between intents
* Results are within the expected range for prompt-only LLM baselines

---

## 🧠 Key Takeaways

* LLMs can serve as **baseline intent classifiers** without training
* Zero-shot accuracy ~60% is reasonable for complex intent taxonomies
* Performance can be improved with:

  * Label normalization
  * Few-shot prompting
  * Top-k evaluation strategies

---

## 🚀 Next Steps

* Extend the same pipeline to additional intent datasets
* Perform cross-dataset generalization analysis
* Introduce few-shot examples to improve accuracy
* Conduct per-intent error analysis

---

## 🏁 Conclusion

This project demonstrates a structured and reproducible approach to evaluating LLMs for intent classification using zero-shot prompting. It establishes a strong baseline for further experimentation and multi-dataset benchmarking.

---
