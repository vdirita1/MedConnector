# Matching Service – Deep‑Dive Documentation (rev 2025-05-19 v3)

*Authors: VWD + o3 2025*

> **Goal:** Provide a transparent walk‑through of every matching function **plus** two step‑by‑step worked examples for each one, with instant jump‑links to a glossary of the underlying algorithms (RapidFuzz, TF‑IDF, etc.).
> Click any blue term (e.g. **[TF‑IDF](#tfidf)**) to jump to its mini‑explanation.

---

## Table of Contents

1. [Component matchers](#component-matchers)

   1. [match\_by\_year](#match_by_year)
   2. [match\_by\_gap\_year](#match_by_gap_year)
   3. [match\_by\_undergrad\_degree](#match_by_undergrad_degree)
   4. [match\_by\_clinical\_interests](#match_by_clinical_interests)
   5. [match\_by\_student\_orgs](#match_by_student_orgs)
   6. [match\_by\_research\_interests](#match_by_research_interests)
   7. [match\_by\_motivation\_essay](#match_by_motivation_essay)
2. [Overall match index](#overall-match-index)
3. [Algorithm glossary](#algorithm-glossary)

---

## Component matchers

### <a id="match_by_year"></a>1 · match\_by\_year

**What it checks:** Does the med‑student’s academic year (M1–M4) satisfy the pre‑med’s preference?
**Scoring rule**

| Year difference | Score |
| --------------- | ----- |
| 0 (exact)       | 1.0   |
| 1 (adjacent)    | 0.5   |
| ≥2              | 0.0   |

#### Worked examples

| Premed wants | Med student is | Output score |
| ------------ | -------------- | ------------ |
| M2           | M2             | 1.00         |
| M2           | M1             | 0.50         |

### <a id="match_by_gap_year"></a>2 · match\_by\_gap\_year

**What it checks:** Preferred *gap year* count (“None”, “1”, “2”, “More than 2”).

#### Mapping to integers

| Label       | Internal code |
| ----------- | ------------- |
| None        | 0             |
| 1           | 1             |
| 2           | 2             |
| More than 2 | 3             |

Score is computed exactly like *match\_by\_year* on the converted codes.

#### Worked examples

| Premed wants | Med student gap | Score |
| ------------ | --------------- | ----- |
| None         | None            | 1.00  |
| 1            | More than 2     | 0.00  |

### <a id="match_by_undergrad_degree"></a>3 · match\_by\_undergrad\_degree

**What it checks:** Similarity between undergraduate degree descriptions using a hybrid of
• [RapidFuzz](#rapidfuzz) token‑set ratio,
• [BiomedBERT](#biomedbert) semantic similarity,
• direct substring match, and
• domain‑specific boosts / penalties.

Weights (default): `0.2 fuzzy + 0.55 semantic + 0.25 direct`.

#### Worked examples

| Premed degree          | Med degree | Approx. score |
| ---------------------- | ---------- | ------------- |
| Human Biology          | Physiology | ≈ 0.83        |
| Mechanical Engineering | Sociology  | ≈ 0.12        |

### <a id="match_by_clinical_interests"></a>4 · match\_by\_clinical\_interests

**What it checks:** Overlap between checkbox clinical interests **and** free‑text entries.

| Component                                 | Share |
| ----------------------------------------- | ----- |
| Jaccard overlap                           | ⅓     |
| [RapidFuzz](#rapidfuzz) `token_set_ratio` | ⅓     |
| **[BiomedBERT](#biomedbert)** similarity  | ⅓     |

#### Worked examples

| Premed set            | Med set              | Resulting score |
| --------------------- | -------------------- | --------------- |
| Cardiology, Neurology | Cardiology, Oncology | ≈ 0.55          |
| Dermatology           | Orthopaedics         | 0.00            |

### <a id="match_by_student_orgs"></a>5 · match\_by\_student\_orgs

**What it checks:** Exact case‑insensitive overlap of student‑organisation names.
Returns **1.0** if *any* org intersects, else **0.0**.

#### Worked examples

| Premed orgs     | Med orgs                  | Score |
| --------------- | ------------------------- | ----- |
| AMSA, Red Cross | Red Cross / Global Health | 1.00  |
| AMSA            | SNMA                      | 0.00  |

### <a id="match_by_research_interests"></a>6 · match\_by\_research\_interests

**Hybrid recipe**

| Component                                | Weight |
| ---------------------------------------- | ------ |
| **[BiomedBERT](#biomedbert)** similarity | 0.60   |
| Keyword Jaccard                          | 0.10   |
| [RapidFuzz](#rapidfuzz) fuzzy            | 0.10   |
| **[TF‑IDF](#tfidf)** cosine              | 0.20   |

Plus a *direct‑substring boost* (+0.3) if one text fully contains the other.

#### Worked examples

| Premed research            | Med research               | Approx. score |
| -------------------------- | -------------------------- | ------------- |
| Glioblastoma immunotherapy | Brain tumour immunotherapy | ≈ 0.80        |
| Renewable energy policy    | Cardiac stem‑cell therapy  | ≈ 0.10        |

### <a id="match_by_motivation_essay"></a>7 · match\_by\_motivation\_essay

**What it checks:** Similarity of free‑text motivation essays using

* general [Sentence‑BERT](#sentencebert) embeddings,
* motivation‑keyword Jaccard, and
* **[TF‑IDF](#tfidf)** cosine.

Weights: `0.5 embedding + 0.3 keywords + 0.2 tfidf`.

#### Worked examples

| Scenario                                                            | Explanation                             | Score  |
| ------------------------------------------------------------------- | --------------------------------------- | ------ |
| Both emphasise *service* and *community health*                     | High keyword overlap ➜ embedding ≈ 0.75 | ≈ 0.75 |
| Premed focuses on *basic research*; Med focuses on *global surgery* | Low overlap                             | ≈ 0.25 |

## <a id="overall-match-index"></a>Overall match index

`overall_match_index()` takes the 7 component scores above and combines them with **\_active\_weights()**
(default weights shown below; runtime overrides allowed via `MATCH_WEIGHTS`).

| Component  | Default weight |
| ---------- | -------------- |
| year       | 0.20           |
| gap        | 0.10           |
| degree     | 0.15           |
| clinical   | 0.25           |
| research   | 0.15           |
| motivation | 0.10           |
| orgs       | 0.05           |

The helper **\_safe\_weighted\_sum** divides by the sum of weights actually present, keeping the final index safely between 0‑1 even if some components are missing.

#### Worked example

Suppose the component scores for a given med‑student are

| year | gap | degree | clinical | research | motivation | orgs |
| ---- | --- | ------ | -------- | -------- | ---------- | ---- |
| 1.0  | 0.5 | 0.83   | 0.55     | 0.80     | 0.75       | 1.0  |

Weighted sum =
`(0.2·1.0 + 0.1·0.5 + 0.15·0.83 + 0.25·0.55 + 0.15·0.80 + 0.10·0.75 + 0.05·1.0) / 1.0` ≈ **0.78**

This med‑student would appear near the top of the recommendations.

---

## <a id="algorithm-glossary"></a>Algorithm glossary

### <a id="rapidfuzz"></a>RapidFuzz – fuzzy string matching

* Computes normalised Levenshtein distance; `token_set_ratio` first splits each string into the **set** of unique tokens, so word order and duplicates don’t matter.
* Raw score range 0 – 100 → divided by 100 in code to give 0 – 1.

**Toy example**

| s1              | s2                       | token\_set\_ratio |
| --------------- | ------------------------ | ----------------- |
| "Human Biology" | "Biology, Human"         | 100               |
| "Biology"       | "Biomedical Engineering" | 32                |

### <a id="tfidf"></a>TF‑IDF – term‑frequency / inverse‑document‑frequency

* Converts every document into a sparse vector of term weights.
* High weight for words frequent in the document **but rare** across the corpus.
* Cosine similarity on these vectors measures lexical overlap beyond exact matches.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(["brain tumour immunotherapy",
                       "glioblastoma immunotherapy"])
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(X[0], X[1]).item()  # ≈ 0.67
```

### <a id="jaccard"></a>Jaccard overlap

* For two sets *A* and *B*: **|A ∩ B| / |A ∪ B|**
* Returns 1 when sets identical, 0 when disjoint.

### <a id="biomedbert"></a>BiomedBERT – domain‑tuned sentence embeddings

* `microsoft/BiomedNLP‑BiomedBERT‑base‑uncased‑abstract` fine‑tuned on PubMed abstracts.
* Provides dense 768‑dim vectors well‑suited for biomedical semantics.
* Similarity via cosine – see `util.cos_sim`.

### <a id="sentencebert"></a>Sentence‑BERT (all‑mpnet‑base‑v2)

* General‑purpose 768‑dim embeddings for any English text.
* Good for longer free‑text such as motivation essays.

### Safe weighted average

`_safe_weighted_sum(scores, weights)` divides by the sum of **provided** weights, so if a component is missing the result stays safely in 0‑1.
