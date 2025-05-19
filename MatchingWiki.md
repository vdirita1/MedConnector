# Matching Service – Deep-Dive Documentation (rev 2025-05-19)
_Authors: VWD + o3 2025_

> **Goal** – Give a transparent walk-through of every matching function **and** instant jump-links to a glossary of their underlying algorithms.  
> Click any blue term such as **[TF-IDF](#tf-idf)** or **[RapidFuzz](#rapidfuzz)** to jump to its mini-explanation.

---

## Table of Contents
1. [Component matchers](#component-matchers)  
2. [Overall match index](#overall-match-index)  
3. [Algorithm glossary](#algorithm-glossary)

---

<a name="component-matchers"></a>
## Component matchers
Each subsection explains the scoring logic **with exact weights** and provides **two worked examples**.

### 1 · `match_by_year`
*Exact vs. adjacent M-year logic.*

| Relationship | Score |
|--------------|-------|
| Same year    | **1.0** |
| ±1 year      | 0.5 |
| Otherwise    | 0.0 |

**Worked examples**

| Premed request | Med student year | Result |
|----------------|------------------|--------|
| Wants **M2**   | Is **M2**        | 1.0 |
| Wants **M2**   | Is **M3**        | 0.5 |

---

### 2 · `match_by_gap_year`
After mapping `"None" → 0`, `"1" → 1`, `"2" → 2`, `"More than 2" → 3`.

| Gap-year diff. | Score |
|----------------|-------|
| 0              | **1.0** |
| 1              | 0.5 |
| ≥ 2            | 0.0 |

**Worked examples**

| Premed pref.     | Med actual | Diff | Result |
|------------------|------------|------|--------|
| “1”              | “1”        | 0    | 1.0 |
| “More than 2”    | “1”        | 2    | 0.0 |

---

### 3 · `match_by_student_orgs`
*Exact, case-insensitive set overlap.*

| Overlap?       | Score |
|----------------|-------|
| ≥ 1 shared org | **1.0** |
| None           | 0.0 |

**Worked examples**

| Premed orgs                    | Med orgs                        | Result |
|--------------------------------|---------------------------------|--------|
| “AMSA, Neurology Club”         | “Neurology Club”                | 1.0 |
| “Surgery Society”              | “Public Health Association”     | 0.0 |

---

### 4 · `match_by_undergrad_degree`

**Base blend**

| Component | Weight |
|-----------|--------|
| [RapidFuzz](#rapidfuzz) `token_set_ratio` | 20 % |
| [BiomedBERT](#biomedbert) cosine sim.     | 55 % |
| Direct substring bonus                    | 25 % |

**Rule-based adjustments** (applied *after* the blend, then clipped to \[0, 1])  

* +0 – 0.30 for shared biology keywords  
* +0 – 0.20 for shared medical/health keywords  
* +0.25 if both degrees fall in the **same** broad domain group  
* +0.30 for the specific **human biology ↔ physiology** pair  
* −0.40 penalty if sociology mismatches with biology-heavy terms  

**Worked examples**

| Premed degree  | Med degree            | Notes                                   | Result (≈) |
|----------------|-----------------------|-----------------------------------------|------------|
| “Human Biology”| “Physiology”          | Specific-pair boost +0.30               | 0.92 |
| “Sociology”    | “Molecular Biology”   | Sociology penalty −0.40                 | 0.28 |

---

### 5 · `match_by_clinical_interests`

Three-way average:

| Component | Weight |
|-----------|--------|
| Jaccard overlap            | 33 % |
| [RapidFuzz](#rapidfuzz)    | 33 % |
| [BiomedBERT](#biomedbert)  | 33 % |

**Worked examples**

| Premed list                        | Med list                                   | Result (≈) |
|------------------------------------|--------------------------------------------|------------|
| “Neurosurgery, Oncology”           | “Oncology, Neurosurgery”                   | 1.0 |
| “Dermatology, Plastic Surgery”     | “Family Medicine, Emergency Medicine”      | 0.11 |

---

### 6 · `match_by_research_interests`

| Component                       | Weight |
|---------------------------------|--------|
| [BiomedBERT](#biomedbert) embed.| 60 % |
| Jaccard keywords                | 10 % |
| RapidFuzz token-set             | 10 % |
| **[TF-IDF](#tf-idf)** cosine    | 20 % |

*Pairs with **zero** lexical overlap are down-weighted by a relevance filter.*

**Worked examples**

| Premed statement                          | Med statement                                 | Result (≈) |
|-------------------------------------------|-----------------------------------------------|------------|
| “Alzheimer’s biomarkers in CSF”           | “CSF biomarkers for early Alzheimer’s disease”| 0.88 |
| “Global health policy on malaria”         | “Neural stem-cell differentiation in mice”    | 0.12 |

---

### 7 · `match_by_motivation_essay`

| Component                     | Weight |
|-------------------------------|--------|
| `all-mpnet-base-v2` embeddings| 50 % |
| Motivation-keyword Jaccard    | 30 % |
| TF-IDF cosine                 | 20 % |

**Worked examples**

| Premed essay snippet                                     | Med essay snippet                                    | Result (≈) |
|----------------------------------------------------------|------------------------------------------------------|------------|
| “I seek to combine compassionate patient care with translational research.” | “My passion lies in blending bedside care with bench discoveries.” | 0.79 |
| “I’m fascinated by hospital administration and policy.”  | “I find purpose in rural family-medicine outreach.”  | 0.27 |

---

<a name="overall-match-index"></a>
## Overall match index

1. Compute each component score.  
2. Weight them as below (defaults; override via `MATCH_WEIGHTS`).  
3. `_safe_weighted_sum` divides by the sum of **present** weights so missing fields never inflate the index.

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

<a name="algorithm-glossary"></a>
## Algorithm glossary

### <a name="rapidfuzz"></a>RapidFuzz – fuzzy string matching
Normalised Levenshtein distance; `token_set_ratio` first converts each string to a **set** of unique tokens, ignoring order and duplicates.  
Score range 0 – 100 → scaled to 0 – 1 in code.

```python
from rapidfuzz import fuzz
fuzz.token_set_ratio("Human Biology", "Biology, Human")  # 100
