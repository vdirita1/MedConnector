
# Matching Service – Deep‑Dive Documentation (rev 2025-05-19)
_Authors: VWD + o3 2025_

> **Goal:** Give a transparent walk‑through of every matching function **and** instant jump‑links to a glossary of their underlying algorithms.  
> Click any blue term (e.g. **[TF‑IDF](#tfidf)**) to jump to its mini‑explanation.

---
## Table of Contents
1. [Component matchers](#component-matchers)
2. [Overall match index](#overall-match-index)
3. [Algorithm glossary](#algorithm-glossary)

---

## Component matchers
Each subsection explains the scoring logic, **lists exact weights**, and shows mini‑examples.

### 1 · match_by_year
*Exact vs. adjacent M‑year logic.*

| Student year | Score |
|--------------|-------|
| Same year    | **1.0** |
| ±1 year      | 0.5 |
| Otherwise    | 0.0 |

---

### 2 · match_by_gap_year
Same three‑level scoring after mapping  
`"None"→0`, `"1"→1`, `"2"→2`, `"More than 2"→3`.

---

### 3 · match_by_student_orgs
*Exact case‑insensitive set overlap.*

| Overlap? | Score |
|----------|-------|
| ≥1 shared org | **1.0** |
| None          | 0.0 |

---

### 4 · match_by_undergrad_degree

**Base blend**

| Component | Weight |
|-----------|--------|
| **RapidFuzz** `token_set_ratio` | 20 % |
| **BiomedBERT** cosine sim. | 55 % |
| Direct substring bonus | 25 % |

**Rule‑based boosts / penalties** (applied *after* the blend, then clipped to \[0, 1\])  

* +0 – 0.30 for shared biology keywords  
* +0 – 0.20 for shared medical/health keywords  
* +0.25 if both degrees fall in the *same* broad domain group  
* +0.30 for the specific *human biology ↔ physiology* pair  
* −0.40 penalty if sociology mismatches with biology‑heavy terms  

---

### 5 · match_by_clinical_interests

Steps  
1. Normalise checkbox + free‑text into sets  
2. Compute the three similarities below  
3. **Return their average**  

| Component | Weight |
|-----------|--------|
| Jaccard overlap | 33 % |
| RapidFuzz `token_set_ratio` | 33 % |
| **BiomedBERT** similarity | 33 % |

*(No additional boosts or penalties.)*

---

### 6 · match_by_research_interests

| Component | Weight |
|-----------|--------|
| **BiomedBERT** embeddings | 60 % |
| Jaccard keywords | 10 % |
| RapidFuzz token‑set | 10 % |
| **[TF‑IDF](#tfidf)** cosine | 20 % |

*A relevance filter down‑weights pairs with **zero** lexical overlap (see code comment).*  

---

### 7 · match_by_motivation_essay

| Component | Weight |
|-----------|--------|
| `all‑mpnet‑base‑v2` embeddings | 50 % |
| Motivation‑keyword Jaccard | 30 % |
| **TF‑IDF** cosine | 20 % |

---

<a id="overall-match-index"></a>
## Overall match index

1. Compute each component score.  
2. Combine with weights (defaults below; override via `MATCH_WEIGHTS` env var).  
3. `_safe_weighted_sum` divides by the sum of *present* weights so missing data don’t inflate the index.

| Component  | Default weight |
|------------|----------------|
| year       | 0.20 |
| gap        | 0.10 |
| degree     | 0.15 |
| clinical   | 0.25 |
| research   | 0.15 |
| motivation | 0.10 |
| orgs       | 0.05 |

---

<a id="algorithm-glossary"></a>
## Algorithm glossary

### <a id="rapidfuzz"></a>RapidFuzz – Fuzzy string matching
* Computes normalised Levenshtein distance; `token_set_ratio` first converts each string to a *set* of unique tokens, so word order and duplicates don’t matter.  
* Score range 0 – 100 → scaled to 0 – 1 in code.

**Toy example**

| s1 | s2 | token_set_ratio |
|----|----|-----------------|
| "Human Biology" | "Biology, Human" | 100 |
| "Biology" | "Biomedical Engineering" | 32 |

```python
from rapidfuzz import fuzz
fuzz.token_set_ratio("Human Biology", "Biology, Human")  # 100
```

---

### <a id="tfidf"></a>TF‑IDF + Cosine similarity
* **TF** = term frequency, **IDF** = inverse document frequency.  
* Multiplying them down‑weights stop‑words and common terms.  
* Cosine similarity of non‑negative TF‑IDF vectors is therefore in **\[0, 1\]**.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["microglia activation", "activation of microglia in AD"]
tfidf = TfidfVectorizer().fit_transform(docs)
cosine_similarity(tfidf[0], tfidf[1])  # ≈ 0.86
```

---

### <a id="jaccard"></a>Jaccard overlap
|A ∩ B| / |A ∪ B|. Great for yes/no checkbox sets.

---

### <a id="sentencebert"></a>Sentence‑/BiomedBERT embeddings
* **Sentence‑BERT:** fine‑tuned to map sentences into 768‑dim vectors where cosine ≈ semantic similarity.  
* **BiomedBERT** (`microsoft/BiomedNLP‑BiomedBERT‑base‑uncased‑abstract`): pre‑trained on PubMed; better for biomedical terms.

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
util.cos_sim(model.encode("brain tumor"),
             model.encode("glioblastoma")).item()  # ≈ 0.71
```

---

### Safe weighted average
`_safe_weighted_sum(scores, weights)` divides by the sum of **present** weights, so if a component is missing (NaN), the result stays in the 0‑1 range.

---
