# Practical 1 — Tokenization, Stemming \& Lemmatization

**Name:** Paras Vishwakarma
**Class:** C  
**Roll No:** 48  
**Subject:** (410256) Laboratory Practice VI — NLP  
**Academic Year:** 2025-26 / SEM-II  
**College:** Genba Sopanrao Moze College of Engineering

\---

## Problem Statement

Write a Python program to perform the following NLP preprocessing tasks on a sample sentence using the NLTK library:

* Apply five tokenization techniques: Whitespace, Punctuation-based (WordPunct), Treebank, Tweet, and MWE Tokenizer.
* Perform stemming using Porter Stemmer and Snowball Stemmer.
* Perform lemmatization using WordNet Lemmatizer with POS tagging (verb and noun).
* Compare outputs of all techniques and observe the differences.

**Input:** Sample sentence —  
`"The electric cars are running faster than ever! U.S. auto-makers can't ignore #EV trends in 2024 :)"`

\---

## Libraries Used

|Library|Purpose|
|-|-|
|`nltk`|Natural Language Toolkit for tokenization, stemming, lemmatization|
|`pandas`|Display results as DataFrames|

Install command:

```bash
pip install nltk
```

NLTK downloads required:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt\_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged\_perceptron\_tagger')
```

\---

## Explanation

### 1\. Tokenization

Tokenization is the process of splitting text into smaller units called **tokens**. It is the first step in any NLP pipeline. Different tokenizers handle punctuation, contractions, and special characters differently.

|Tokenizer|Description|
|-|-|
|**Whitespace**|Splits only on spaces/tabs. Punctuation stays attached to words.|
|**WordPunct (Punct-based)**|Splits on whitespace AND punctuation. Each symbol becomes a separate token.|
|**Treebank**|Penn Treebank rules. Handles contractions correctly (can't → ca, n't).|
|**Tweet**|Designed for social media. Handles #hashtags, emoticons :), @mentions.|
|**MWE**|Merges predefined multi-word expressions (electric cars → electric\_cars).|

### 2\. Stemming

Stemming reduces a word to its root by removing suffixes using heuristic rules. The result may not be a real word.

* **Porter Stemmer** — Oldest and most commonly used. e.g., `running → run`, `studies → studi`
* **Snowball Stemmer** — Improved version of Porter (Porter2). Slightly more accurate. e.g., `running → run`, `happily → happili`

### 3\. Lemmatization

Lemmatization returns the actual dictionary base form (lemma) using vocabulary and morphological analysis. Unlike stemming, it always returns a valid word.

* Uses **WordNet Lemmatizer** from NLTK.
* The `pos` parameter specifies part-of-speech (`v` = verb, `n` = noun).
* e.g., `running (verb) → run`, `feet (noun) → foot`, `better (adj) → good`

\---

## Conclusion

In this practical, we successfully implemented five tokenization techniques using NLTK. We observed that:

* **Whitespace tokenizer** is the simplest but leaves punctuation attached to words.
* **WordPunct tokenizer** splits every punctuation into a separate token.
* **Treebank tokenizer** is best for formal English and handles contractions well.
* **Tweet tokenizer** correctly handles hashtags, emoticons, and social media text.
* **MWE tokenizer** is useful when domain-specific phrases must be kept together.

For stemming, both Porter and Snowball produce similar results, but Snowball is slightly more refined. Lemmatization is more accurate than stemming as it returns real dictionary words, but requires the correct POS tag for best results.

\---

*Submitted by: Paras Vishwakarma | Roll No: 48 | Class: C | BE SEM-II | 2025-26*

