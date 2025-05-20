# Matching Service – Deep‑Dive Documentation (rev 2025-05-19 v3)

*Authors: VWD + Claude Sonnet 3.7 2025*

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

**What it checks:** Does the med‑student's academic year (M1–M4) satisfy the pre‑med's preference?
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

**What it checks:** Preferred *gap year* count ("None", "1", "2", "More than 2").

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

* Computes normalised Levenshtein distance; `token_set_ratio` first splits each string into the **set** of unique tokens, so word order and duplicates don't matter.
* Raw score range 0 – 100 → divided by 100 in code to give 0 – 1.

**Toy example**

| s1              | s2                       | token\_set\_ratio |
| --------------- | ------------------------ | ----------------- |
| "Human Biology" | "Biology, Human"         | 100               |
| "Biology"       | "Biomedical Engineering" | 32                |

### <a id="tfidf"></a>TF‑IDF – term‑frequency / inverse‑document‑frequency

* Converts every document into a sparse vector of term weights.
* High weight for words frequent in the document **but rare** across the corpus.
* Cosine similarity on these vectors measures lexical overlap beyond exact matches.

**Detailed explanation with worked example:**

TF-IDF consists of two components:
1. **Term Frequency (TF)**: How often a word appears in a document (normalized)
2. **Inverse Document Frequency (IDF)**: How rare or common a word is across all documents

The formula is: TF-IDF = TF × IDF

Let's work through a complete example with a small corpus about medical research interests:

```
Document 1: "cancer immunotherapy research"
Document 2: "brain tumor immunotherapy"
Document 3: "cardiovascular disease research"
```

**Step 1: Calculate Term Frequency (TF)**
For each term in each document:
TF = (Number of times term appears in document) / (Total number of terms in document)

For Document 1:
- TF(cancer) = 1/3 = 0.33
- TF(immunotherapy) = 1/3 = 0.33
- TF(research) = 1/3 = 0.33

For Document 2:
- TF(brain) = 1/3 = 0.33
- TF(tumor) = 1/3 = 0.33
- TF(immunotherapy) = 1/3 = 0.33

For Document 3:
- TF(cardiovascular) = 1/3 = 0.33
- TF(disease) = 1/3 = 0.33
- TF(research) = 1/3 = 0.33

**Step 2: Calculate Inverse Document Frequency (IDF)**
IDF = log(Total number of documents / Number of documents containing the term)

- IDF(cancer) = log(3/1) = log(3) ≈ 1.10
- IDF(immunotherapy) = log(3/2) = log(1.5) ≈ 0.41
- IDF(research) = log(3/2) = log(1.5) ≈ 0.41
- IDF(brain) = log(3/1) = log(3) ≈ 1.10
- IDF(tumor) = log(3/1) = log(3) ≈ 1.10
- IDF(cardiovascular) = log(3/1) = log(3) ≈ 1.10
- IDF(disease) = log(3/1) = log(3) ≈ 1.10

**Step 3: Calculate TF-IDF for each term in each document**

Document 1:
- TF-IDF(cancer) = 0.33 × 1.10 = 0.36
- TF-IDF(immunotherapy) = 0.33 × 0.41 = 0.14
- TF-IDF(research) = 0.33 × 0.41 = 0.14

Document 2:
- TF-IDF(brain) = 0.33 × 1.10 = 0.36
- TF-IDF(tumor) = 0.33 × 1.10 = 0.36
- TF-IDF(immunotherapy) = 0.33 × 0.41 = 0.14

Document 3:
- TF-IDF(cardiovascular) = 0.33 × 1.10 = 0.36
- TF-IDF(disease) = 0.33 × 1.10 = 0.36
- TF-IDF(research) = 0.33 × 0.41 = 0.14

**Step 4: Create document vectors**

Document 1: [0.36, 0.14, 0.14, 0, 0, 0, 0]
Document 2: [0, 0.14, 0, 0.36, 0.36, 0, 0]
Document 3: [0, 0, 0.14, 0, 0, 0.36, 0.36]

**Step 5: Calculate cosine similarity**

Cosine similarity between Doc1 and Doc2:
= (0 × 0 + 0.14 × 0.14 + 0.14 × 0 + 0.36 × 0.36 + 0.36 × 0.36 + 0 × 0 + 0 × 0) / 
  (√(0.36²+0.14²+0.14²) × √(0²+0.14²+0²+0.36²+0.36²+0²+0²))
= 0.02 / (0.41 × 0.52)
≈ 0.09

This means Documents 1 and 2 have some similarity (they share "immunotherapy"), but are mostly different.

**In our matching system:**
TF-IDF helps identify terms that are distinctive and important in research descriptions. For example, "glioblastoma" would get a higher weight than common words like "study" or "analysis" when matching research interests.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(["brain tumour immunotherapy",
                       "glioblastoma immunotherapy"])
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(X[0], X[1]).item()  # ≈ 0.67
```

### <a id="jaccard"></a>Jaccard overlap

* For two sets *A* and *B*: **|A ∩ B| / |A ∪ B|**
* Returns 1 when sets identical, 0 when disjoint.

**Detailed explanation with worked example:**

Jaccard similarity measures how similar two sets are by calculating the ratio of their intersection to their union.

Formula: J(A,B) = |A ∩ B| / |A ∪ B|

Let's walk through examples relevant to our matching system:

**Example 1: Clinical Interests**

Premed student interests: {"Cardiology", "Neurology", "Pediatrics"}
Med student interests: {"Neurology", "Pediatrics", "Oncology"}

Intersection: {"Neurology", "Pediatrics"} - 2 elements
Union: {"Cardiology", "Neurology", "Pediatrics", "Oncology"} - 4 elements

Jaccard similarity = 2/4 = 0.5 or 50% overlap

**Example 2: Student Organizations** 

Premed orgs: {"AMSA", "Red Cross", "Global Health Initiative"}
Med student orgs: {"SNMA", "AMA", "Gold Humanism"}

Intersection: {} (empty set) - 0 elements
Union: {"AMSA", "Red Cross", "Global Health Initiative", "SNMA", "AMA", "Gold Humanism"} - 6 elements

Jaccard similarity = 0/6 = 0 or 0% overlap

**Example 3: Research Keywords**

After keyword extraction:
Premed research keywords: {"cancer", "immunotherapy", "t-cell", "lymphoma"}
Med student research keywords: {"cancer", "immunotherapy", "checkpoint", "inhibitors"}

Intersection: {"cancer", "immunotherapy"} - 2 elements
Union: {"cancer", "immunotherapy", "t-cell", "lymphoma", "checkpoint", "inhibitors"} - 6 elements

Jaccard similarity = 2/6 ≈ 0.33 or 33% overlap

**Implementation detail:** In our matching system, we often transform free text into sets of keywords or tokens before calculating Jaccard similarity. This allows us to match on meaningful terms while filtering out common words like "the" or "and".

### <a id="biomedbert"></a>BiomedBERT – domain‑tuned sentence embeddings

* `microsoft/BiomedNLP‑BiomedBERT‑base‑uncased‑abstract` fine‑tuned on PubMed abstracts.
* Provides dense 768‑dim vectors well‑suited for biomedical semantics.
* Similarity via cosine – see `util.cos_sim`.

### <a id="sentencebert"></a>Sentence‑BERT (now using BAAI/bge-large-en-v1.5)

* General‑purpose embeddings for any English text.
* Good for longer free‑text such as motivation essays.

**Detailed explanation with worked example:**

Sentence transformers convert text into dense vector representations (embeddings) where semantic similarity is preserved. Similar meanings are close in vector space, even if they use different words.

**How Sentence Transformers Work:**

1. **Tokenization**: Text is split into tokens (subwords)
2. **Encoding**: Tokens are processed through transformer layers
3. **Pooling**: Token representations are combined into a single sentence embedding
4. **Comparison**: Cosine similarity measures how aligned two embeddings are

**Worked Example:**

Let's compare two motivation statements using BAAI/bge-large-en-v1.5:

```
Statement 1: "I am passionate about serving underserved communities through medicine. My volunteer work in rural clinics showed me the impact a dedicated physician can have."

Statement 2: "My goal is to help disadvantaged populations access quality healthcare. Working at community clinics demonstrated the difference doctors make in vulnerable areas."
```

These statements express similar motivations but use different phrasing.

**Embedding Process (simplified):**

1. Each statement is tokenized and encoded through the transformer model
2. The model outputs a 1024-dimensional vector for each statement
3. These vectors capture the semantic meaning of each text

**Visualization (high-dimensional vectors simplified to 2D):**
```
         Statement 1 ●
                     \
                      \ (small angular distance)
                       \
                        ● Statement 2


         Statement 3 ●
                      \
                       \
                        \ (large angular distance)
                         \
                          ● Statement 1
```

Where Statement 3 might be: "I'm interested in medical research to develop new cancer therapies" - a different motivation.

**Cosine Similarity Calculation:**
If our simplified vectors were:
- Statement 1: [0.8, 0.6]
- Statement 2: [0.7, 0.7]

Cosine similarity = (0.8×0.7 + 0.6×0.7) / (√(0.8²+0.6²) × √(0.7²+0.7²))
                  = 1.12 / (1.0 × 0.99)
                  ≈ 0.94

This high similarity score (close to 1.0) indicates the statements express very similar motivations despite different wording.

In our matching system, this allows us to match motivation essays that express the same values and goals even when they use completely different vocabulary, capturing deeper semantic relationships than keyword matching alone.

### Safe weighted average

`_safe_weighted_sum(scores, weights)` divides by the sum of **provided** weights, so if a component is missing the result stays safely in 0‑1.
