
# Matching Service – Deep‑Dive Documentation
_Authors: VWD + o3 2025

> **Goal of this file:** give both a *walk‑through* of each matching function **and** instant jump‑links to a mini‑glossary of the underlying algorithms (RapidFuzz, TF‑IDF, etc.).  
> Click a blue link like **[TF‑IDF](#tfidf)** any time it appears to read what it means and see a toy example.

---

## Table of Contents
1. [Component matchers](#component-matchers)  
2. [Overall match index](#overall-match-index)  
3. [Algorithm glossary](#algorithm-glossary)  

---

## Component matchers
Each subsection explains the logic and shows two mini‑examples.  
Algorithm names link to their glossary entries.

### 1 · match_by_year  
*Exact / adjacent M‑year logic.*

| Student year | Score rationale |
|--------------|-----------------|
| same year    | **1.0** |
| ±1 year      | 0.5 |
| otherwise    | 0.0 |

<details><summary>Examples</summary>

| Premed wants | Student year | Score |
|--------------|--------------|-------|
| M1 | M1 | **1.0** |
| M1 | M2 | 0.5 |
| M3 | M2 | 0.5 |
| M3 | M4 | 0.5 |
| M1 | M3 | 0.0 |
</details>

---

### 2 · match_by_gap_year  
Same scoring scheme, after mapping `"None"→0`, `"1"→1`, `"2"→2`, `"More than 2"→3`.

---

### 3 · match_by_student_orgs  
Uses **exact set overlap** – if any organisation name matches (case‑insensitive) score = 1, else 0.

---

### 4 · match_by_undergrad_degree  
Hybrid score:

* **[RapidFuzz](#rapidfuzz)** `token_set_ratio` (20 %)  
* **[Bio‑BERT embeddings](#sentencebert)** cosine (55 %)  
* **Direct substring bonus** (25 %)  
* Domain boosts / penalties

Why a hybrid? RapidFuzz catches near‑string matches, embeddings catch conceptual matches (“Physiology” ↔ “Human Biology”).

---

### 5 · match_by_clinical_interests  
Steps  

1. Normalise checkbox + free‑text into sets.  
2. Compute  
   * Jaccard overlap  
   * RapidFuzz fuzzy ratio  
   * **[Bio‑BERT](#sentencebert)** similarity  
3. Return their average.

---

### 6 · match_by_research_interests  
Four‑way blend:

| Component | Weight |
|-----------|--------|
| **Bio‑BERT embeddings** | 0.60 |
| **Jaccard** keywords | 0.10 |
| **RapidFuzz** token‑set | 0.10 |
| **[TF‑IDF](#tfidf)** cosine | 0.20 |

A relevance filter down‑weights cases with zero lexical overlap.

---

### 7 · match_by_motivation_essay  
Uses a *general* Sentence‑BERT (`all‑mpnet‑base‑v2`) plus keyword Jaccard and **TF‑IDF**.

---

<a id="overall-match-index"></a>
## Overall match index
1. Compute all component scores.  
2. Combine with weights (defaults shown; override with `MATCH_WEIGHTS` env var).  
3. Use `_safe_weighted_sum` to stay in 0–1 range even if some scores are missing.

| Component | Default weight |
|-----------|----------------|
| year | 0.20 |
| gap  | 0.10 |
| degree | 0.15 |
| clinical | 0.25 |
| research | 0.15 |
| motivation | 0.10 |
| orgs | 0.05 |

---

<a id="algorithm-glossary"></a>
## Algorithm glossary

### <a id="rapidfuzz"></a>RapidFuzz – Fuzzy string matching
* **Core idea:** computes Levenshtein edit‑distance but tokenises first; order of words doesn’t matter.  
* **Token‑set ratio:**  
  1. Split both strings on whitespace/punctuation.  
  2. Remove duplicate tokens.  
  3. Join tokens and compute edit similarity → score in \[0–100\].  
* **Toy example**

| s1 | s2 | token_set_ratio |
|----|----|-----------------|
| "Human Biology" | "Biology, Human" | 100 |
| "Biology" | "Biomedical Engineering" | 32 |

Python snippet  
```python
from rapidfuzz import fuzz
fuzz.token_set_ratio("Human Biology", "Biology, Human")  # 100
```

---

### <a id="tfidf"></a>TF‑IDF + Cosine similarity
* **TF (term‑frequency):** how often each word appears in a document.  
* **IDF (inverse document frequency):** log(N / df) down‑weights very common words like “the”.  
* Multiply TF × IDF → TF‑IDF vector.  
* **Cosine similarity** between two TF‑IDF vectors ≈ angle between them. Range \[-1,1\] but non‑negative with typical TF‑IDF.

Why here: highlights uncommon biomedical terms shared between texts without heavy compute.

*Toy example* (sklearn):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
docs = ["microglia activation", "activation of microglia in AD"]
tfidf = TfidfVectorizer().fit_transform(docs)
cosine_similarity(tfidf[0], tfidf[1])  # 0.86
```

---

### <a id="jaccard"></a>Jaccard overlap
Exact‑set similarity = |intersection| / |union|.  
Useful for yes/no checkbox lists.

Example:  
*Set A* = {neurosurgery, surgery}  
*Set B* = {neurosurgery, anesthesiology}  
Jaccard = 1 / 3 ≈ 0.33

---

### <a id="sentencebert"></a>Sentence‑/Bio‑BERT embeddings
* **Sentence‑BERT:** fine‑tunes a transformer to map sentences into 768‑dim vectors where cosine ≈ semantic similarity.  
* **Bio‑BERT:** pre‑trained on PubMed abstracts; better for biomedical terms.

Example:

```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
sim = util.cos_sim(model.encode("brain tumor"), model.encode("glioblastoma"))
print(float(sim))  # ≈ 0.71
```

---

### Safe weighted average
`_safe_weighted_sum(scores, weights)` divides by the sum of *present* weights, avoiding inflation when some component returns 0 because data are missing.

---

*End of file.*
