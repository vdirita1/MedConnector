# Matching Service – Deep-Dive Documentation (rev 2025-05-19)
_Authors · VWD + o3 2025_

> **Goal** – Explain every matching function, list exact weights, show two worked examples **and** embed mini-glossary definitions (with runnable code snippets) for every algorithm term.  
> Click any blue link such as **[TF-IDF](#tf-idf)** or **[RapidFuzz](#rapidfuzz)** to jump straight to its definition.

---

## Table of Contents
1. [Component matchers](#component-matchers)  
2. [Overall match index](#overall-match-index)  
3. [Algorithm glossary](#algorithm-glossary)

---

<a name="component-matchers"></a>
## Component matchers  
*(Each subsection shows logic, exact weights, and two worked examples.)*

### 1 · `match_by_year`
| Relationship | Score |
|--------------|-------|
| Same year    | **1.0** |
| ±1 year     | 0.5 |
| Otherwise   | 0.0 |

**Examples**

| Premed wants | Med year | Score |
|--------------|----------|-------|
| M2           | M2       | 1.0 |
| M2           | M3       | 0.5 |

---

### 2 · `match_by_gap_year`
_Map “None → 0”, “1 → 1”, “2 → 2”, “More than 2 → 3”._

| Gap-year diff. | Score |
|----------------|-------|
| 0              | **1.0** |
| 1              | 0.5 |
| ≥ 2            | 0.0 |

**Examples**

| Premed pref. | Med actual | Diff | Score |
|--------------|------------|------|-------|
| “1”          | “1”        | 0   | 1.0 |
| “>2”         | “1”        | 2   | 0.0 |

---

### 3 · `match_by_student_orgs`
*Exact, case-insensitive set overlap.*

| ≥ 1 shared org? | Score |
|-----------------|-------|
| Yes             | **1.0** |
| No              | 0.0 |

**Examples**

| Premed orgs                | Med orgs                     | Score |
|----------------------------|------------------------------|-------|
| AMSA, Neurology Club       | Neurology Club               | 1.0 |
| Surgery Society            | Public Health Association    | 0.0 |

---

### 4 · `match_by_undergrad_degree`

| Component                                         | Weight |
|---------------------------------------------------|--------|
| [RapidFuzz](#rapidfuzz) `token_set_ratio`         | 20 % |
| [BiomedBERT](#biomedbert) cosine similarity       | 55 % |
| Direct substring bonus                            | 25 % |

*Post-blend boosts/penalties (clipped to \[0, 1\])*  
+0–0.30 shared biology keywords · +0–0.20 shared medical keywords  
+0.25 same domain group · +0.30 **human biology ↔ physiology**  
−0.40 sociology ↔ biology mismatch

**Examples**

| Premed degree | Med degree        | Notes                     | Score (≈) |
|---------------|-------------------|---------------------------|-----------|
| Human Biology | Physiology        | Pair boost +0.30          | 0.92 |
| Sociology     | Molecular Biology | Sociology penalty −0.40   | 0.28 |

---

### 5 · `match_by_clinical_interests`

| Component                              | Weight |
|----------------------------------------|--------|
| Jaccard overlap                        | 33 % |
| [RapidFuzz](#rapidfuzz) token-set      | 33 % |
| [BiomedBERT](#biomedbert) similarity   | 33 % |

**Examples**

| Premed list              | Med list                      | Score (≈) |
|--------------------------|-------------------------------|-----------|
| Neurosurgery, Oncology   | Oncology, Neurosurgery        | 1.0 |
| Dermatology, Plastic Sx  | Family Med, Emergency Med     | 0.11 |

---

### 6 · `match_by_research_interests`

| Component                         | Weight |
|-----------------------------------|--------|
| [BiomedBERT](#biomedbert) embed.  | 60 % |
| Jaccard keywords                  | 10 % |
| RapidFuzz token-set               | 10 % |
| **[TF-IDF](#tf-idf)** cosine     | 20 % |

*Pairs with **zero** lexical overlap are down-weighted.*

**Examples**

| Premed statement                     | Med statement                              | Score (≈) |
|--------------------------------------|--------------------------------------------|-----------|
| Alzheimer’s biomarkers in CSF        | CSF biomarkers for early Alzheimer’s       | 0.88 |
| Global health policy on malaria      | Neural stem-cell differentiation in mice   | 0.12 |

---

### 7 · `match_by_motivation_essay`

| Component                | Weight |
|--------------------------|--------|
| `all-mpnet-base-v2` embeddings | 50 % |
| Motivation-keyword Jaccard    | 30 % |
| TF-IDF cosine                 | 20 % |

**Examples**

| Premed snippet                                         | Med snippet                                     | Score (≈) |
|--------------------------------------------------------|-------------------------------------------------|-----------|
| Compassionate care + translational research            | Blend bedside care with bench discoveries       | 0.79 |
| Fascinated by hospital admin & policy                  | Rural family-medicine outreach                  | 0.27 |

---

<a name="overall-match-index"></a>
## Overall match index

| Component  | Default weight |
|------------|----------------|
| year       | 0.20 |
| gap        | 0.10 |
| degree     | 0.15 |
| clinical   | 0.25 |
| research   | 0.15 |
| motivation | 0.10 |
| orgs       | 0.05 |

`_safe_weighted_sum` divides by the sum of **present** weights so missing data never inflate the index.

---

<a name="algorithm-glossary"></a>
## Algorithm glossary

### <a name="rapidfuzz"></a>RapidFuzz – fuzzy string matching
Normalised Levenshtein distance; `token_set_ratio` first converts each string to a **set** of unique tokens, ignoring order and duplicates.  
Range 0 – 100 → scaled to 0 – 1.

```python
from rapidfuzz import fuzz
fuzz.token_set_ratio("Human Biology", "Biology, Human")  # 100
