# Practical 2 — Bag of Words, TF-IDF & Word2Vec

**Name:** Paras Vishwakarma  
**Class:** C  
**Roll No:** 48  
**Subject:** (410256) Laboratory Practice VI — NLP  
**Academic Year:** 2025-26 / SEM-II  
**College:** Genba Sopanrao Moze College of Engineering  

---

## Problem Statement

Write a Python program using the Car Dataset (Kaggle — CooperUnion) to perform the following text representation techniques:

- Build a **Bag-of-Words** model using raw count occurrence.
- Build a **Bag-of-Words** model using normalized count occurrence (L1 normalization).
- Apply **TF-IDF** (Term Frequency — Inverse Document Frequency) vectorization.
- Create **Word2Vec** embeddings and demonstrate word similarity between car brands/features.

**Dataset:** [CooperUnion Car Dataset — Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)  
**File used:** `data.csv` (1200 rows, 16 columns)

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Load and preprocess the dataset |
| `numpy` | Numerical operations |
| `scikit-learn` | CountVectorizer and TfidfVectorizer |
| `gensim` | Word2Vec model training |

Install command:
```bash
pip install scikit-learn gensim pandas numpy
```

---

## Dataset Description

The Car Dataset contains specifications of various car models. Key columns used:

| Column | Description |
|---|---|
| Make | Car manufacturer (Toyota, Honda, Ford, etc.) |
| Model | Car model name |
| Engine Fuel Type | Type of fuel (gasoline, diesel, electric, etc.) |
| Transmission Type | AUTOMATIC, MANUAL, etc. |
| Vehicle Size | Compact, Midsize, Large |

A text corpus was built by combining these columns into one description per car:
```
"toyota camry regular unleaded automatic midsize"
```

---

## Explanation

### 1. Bag of Words — Count Occurrence
Each document is represented as a vector of raw word counts. The vocabulary is built from all unique words across all documents. Words not present in a document get count = 0.

**Example:**
```
Document: "toyota camry automatic midsize"
Vector:   [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
           (1 for 'automatic', 'camry', 'midsize', 'toyota')
```

### 2. Bag of Words — Normalized Count (L1 Normalization)
Each row is divided by its total word count so values represent **relative frequency**. Sum of each row = 1.0. This removes bias caused by documents of different lengths.

**Formula:**
```
normalized_value = count / sum_of_all_counts_in_document
```

### 3. TF-IDF (Term Frequency — Inverse Document Frequency)
TF-IDF improves on BoW by:
- **TF** — How often a term appears in a document.
- **IDF** — How rare the term is across all documents (log scale).
- **TF-IDF = TF × IDF**

Common words like "automatic", "gasoline" get lower scores.  
Rare but distinctive words like specific model names get higher scores.

### 4. Word2Vec
Word2Vec is a neural network model that learns dense vector representations (embeddings) for words. Words appearing in similar contexts get similar vectors.

**Parameters used:**
- `vector_size = 100` — 100-dimensional embedding
- `window = 5` — Context window of 5 words
- `min_count = 1` — Include all words
- `epochs = 50` — Training iterations

**Result:** Car brands like Toyota, Honda, Nissan appear close together in vector space because they share similar contexts (gasoline, automatic, midsize, etc.)

---

## Conclusion

In this practical, we successfully applied four text representation techniques on the Car Dataset:

- **Count BoW** gives raw word frequency per document but is affected by document length.
- **Normalized BoW** removes the length bias, making documents comparable.
- **TF-IDF** is more informative than BoW — it highlights distinctive terms and reduces the weight of very common words.
- **Word2Vec** produces dense semantic embeddings where similar car brands and features cluster together in vector space, capturing meaning that BoW and TF-IDF cannot.

These techniques form the foundation of modern NLP and information retrieval systems.

---

*Submitted by: Paras Vishwakarma | Roll No: 48 | Class: C | BE SEM-II | 2025-26*
