# Practical 3 — Text Preprocessing & TF-IDF

**Name:** Paras Vishwakarma | **Roll No:** 48 | **Class:** C | **BE SEM-II 2025-26**  
**Subject:** (410256) Laboratory Practice VI — NLP  
**College:** Genba Sopanrao Moze College of Engineering

---

## Problem Statement

Perform the following NLP preprocessing steps on the News Dataset and create TF-IDF representations. Save all outputs.

1. Text Cleaning (lowercase, remove HTML, URLs, numbers, punctuation, extra spaces)
2. Lemmatization using WordNet Lemmatizer
3. Stop Word Removal using NLTK
4. Label Encoding of news categories
5. TF-IDF Vectorization (unigrams + bigrams, top 5000 features)
6. Save all outputs to files

**Dataset:** `News_dataset.pickle` from [PICT-NLP GitHub](https://github.com/PICT-NLP/BE-NLP-Elective/blob/main/3%20Preprocessing/News_dataset.pickle)

---

## Libraries Used

| Library | Purpose |
|---|---|
| `nltk` | Stop words, WordNet lemmatizer, tokenization |
| `scikit-learn` | LabelEncoder, TfidfVectorizer |
| `pandas` | Data loading and manipulation |
| `numpy` | Array operations |
| `pickle` | Loading .pickle dataset |
| `scipy` | Saving sparse TF-IDF matrix |

```bash
pip install nltk scikit-learn pandas numpy scipy
```

---

## Preprocessing Pipeline

```
Raw Text
   │
   ▼
Text Cleaning  ─── lowercase, remove HTML/URLs/numbers/punctuation/extra spaces
   │
   ▼
Stop Word Removal  ─── remove NLTK English stopwords
   │
   ▼
Lemmatization  ─── WordNet lemmatizer (verb POS)
   │
   ▼
TF-IDF Vectorization  ─── max 5000 features, unigrams + bigrams
   │
   ▼
Save Outputs
```

---

## Explanation

### 1. Text Cleaning
Raw news text contains noise: HTML tags, URLs, numbers, punctuation, and inconsistent casing. Cleaning removes these to keep only meaningful words.

| Operation | Example |
|---|---|
| Lowercase | "Breaking NEWS" → "breaking news" |
| Remove HTML | `<b>text</b>` → "text" |
| Remove URLs | `http://cnn.com` → "" |
| Remove numbers | "100 dead" → "dead" |
| Remove punctuation | "hello, world!" → "hello world" |

### 2. Stop Word Removal
Stop words (the, is, at, which, on...) appear very frequently but carry little meaning. NLTK has 179 English stop words. Removing them reduces noise and improves model performance.

### 3. Lemmatization
Reduces words to their dictionary base form using WordNet's morphological rules.
- "running" → "run", "studies" → "study", "better" (verb) → "better"
- More accurate than stemming — always returns a real word.

### 4. Label Encoding
Converts string category labels into integer numbers that machine learning models can process.
- "sports" → 0, "politics" → 1, "technology" → 2, etc.
- Uses `sklearn.preprocessing.LabelEncoder`

### 5. TF-IDF
Represents each document as a weighted vector of words. Rare but meaningful words get higher scores than common ones. Parameters used:
- `max_features=5000` — keep top 5000 most informative terms
- `ngram_range=(1,2)` — include both single words and 2-word phrases
- `min_df=2` — ignore words appearing in only 1 document
- `sublinear_tf=True` — log-scale TF to reduce impact of very frequent words

---

## Saved Output Files

| File | Description |
|---|---|
| `preprocessed_news.csv` | Full DataFrame with all preprocessing stages |
| `tfidf_matrix.npz` | Sparse TF-IDF matrix |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer (for future use) |
| `label_encoder.pkl` | Fitted Label Encoder |
| `encoded_labels.npy` | Numpy array of encoded labels |

---

## Conclusion

In this practical, we successfully preprocessed the News Dataset through a complete NLP pipeline. Text cleaning removed noise and standardized the text. Stop word removal reduced average document length significantly, removing frequent but uninformative words. Lemmatization reduced vocabulary size by mapping word variants to their base forms. Label encoding converted categorical labels into machine-readable numeric values. TF-IDF vectorization created a (documents × 5000) sparse matrix that captures the importance of each term per document. All outputs were saved for use in downstream tasks like classification.

---

*Submitted by: Paras Vishwakarma | Roll No: 48 | Class: C | BE SEM-II | 2025-26*
