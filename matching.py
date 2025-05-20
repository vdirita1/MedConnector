from typing import List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util, models
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import os, json
from functools import lru_cache

# ── Static weights (edit here OR override via env var MATCH_WEIGHTS) ──
DEFAULT_WEIGHTS: dict[str, float] = {
    "year":       0.20,
    "gap":        0.10,
    "degree":     0.15,
    "clinical":   0.25,
    "research":   0.15,
    "motivation": 0.10,
    "orgs":       0.05,
}

def _active_weights() -> dict[str, float]:
    """
    Return weight dict, letting users hot-swap via:
        export MATCH_WEIGHTS='{"clinical":0.30,"orgs":0.02}'
    """
    raw = os.getenv("MATCH_WEIGHTS")
    if raw:
        try:
            env_w = json.loads(raw)
            return {**DEFAULT_WEIGHTS, **env_w}
        except Exception:
            pass
    return DEFAULT_WEIGHTS

def _safe_weighted_sum(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average while preserving 0-1 range even if some fields are missing."""
    num   = sum(weights[k] * v for k, v in scores.items())
    denom = sum(weights[k]     for k in scores)
    return num / denom if denom else 0.0

class MatchingService:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.med_students_df = None
        self.premed_df = None
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # For TF-IDF calculations
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self.tfidf_matrix = None
        self.tfidf_docs = []
        
        # Explicitly define BiomedBERT with mean pooling
        word_embed = models.Transformer(
            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
            max_seq_length=512
        )
        pooling = models.Pooling(
            word_embed.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        self.bio_embedder = SentenceTransformer(modules=[word_embed, pooling])

    def load_data(self, med_students_path: str, premed_path: str = None):
        """
        Load data from Excel files.
        
        Args:
            med_students_path: Path to the medical students Excel file
            premed_path: Path to the pre-med student Excel file (optional)
        """
        self.med_students_df = pd.read_excel(med_students_path)
        if premed_path:
            self.premed_df = pd.read_excel(premed_path)
            self.premed_df.columns = self.premed_df.columns.str.strip()
            
        # Initialize TF-IDF with research interests from the dataset
        self._initialize_tfidf_model()

    def _initialize_tfidf_model(self):
        """
        Initialize the TF-IDF model with all research interests in the dataset.
        This allows for better domain-specific term weighting.
        """
        # Extract all research interests
        research_docs = []
        
        # Add med student research
        if self.med_students_df is not None:
            for _, row in self.med_students_df.iterrows():
                research = self._safe_get(row, 'Q5', "")
                if research and len(research.strip()) > 0:
                    research_docs.append(research)
                    
        # Add premed research if available
        if self.premed_df is not None:
            for _, row in self.premed_df.iterrows():
                research = self._safe_get(row, 'Q5', "")
                if research and len(research.strip()) > 0:
                    research_docs.append(research)
        
        # Fit the TF-IDF vectorizer on all research texts
        if research_docs:
            self.tfidf_docs = research_docs
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(research_docs)
        
    def _compute_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity between two research interest statements.
        If the TF-IDF model hasn't been initialized, will return 0.
        """
        if not self.tfidf_vectorizer or not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
            
        # Clean the texts
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        if not text1 or not text2:
            return 0.0
            
        try:
            # Transform using the fitted vectorizer
            vec1 = self.tfidf_vectorizer.transform([text1])
            vec2 = self.tfidf_vectorizer.transform([text2])
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim = float(cosine_similarity(vec1, vec2)[0][0])
            return sim
        except Exception as e:
            print(f"Error computing TF-IDF similarity: {e}")
            return 0.0

    def _convert_year(self, year_str: str) -> int:
        """
        Convert year string (M1, M2, M3, M4) to integer (1, 2, 3, 4).
        """
        mapping = {'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4}
        return mapping.get(str(year_str).strip().upper(), 0)  # 0 if not found

    def _convert_gap_year(self, gap_str: str) -> int:
        """
        Convert gap year string to integer for comparison.
        Maps: "None" -> 0, "1" -> 1, "2" -> 2, "More than 2" -> 3
        """
        mapping = {
            'None': 0,
            '1': 1,
            '2': 2,
            'More than 2': 3
        }
        return mapping.get(str(gap_str).strip(), 0)  # 0 if not found

    def gap_year_match(self, premed_val: str, med_val: str) -> float:
        """
        Compare pre-med's preferred gap years with med student's actual gap years.
        
        Args:
            premed_val: Pre-med's preferred gap years (Q2)
            med_val: Med student's actual gap years (Q2)
            
        Returns:
            float: Match score where:
            - 1.0 = exact match
            - 0.5 = one step apart
            - 0.0 = more than one step apart
        """
        premed_gap = self._convert_gap_year(premed_val)
        med_gap = self._convert_gap_year(med_val)
        
        gap_diff = abs(premed_gap - med_gap)
        if gap_diff == 0:
            return 1.0
        elif gap_diff == 1:
            return 0.5
        else:
            return 0.0

    def calculate_year_compatibility(self, premed_year: int, med_student_year: int) -> float:
        """
        Score is 1.0 for exact match (user request satisfied), 
        0.5 for adjacent years (fallback), and 0.0 otherwise.
        """
        year_diff = abs(premed_year - med_student_year)
        if year_diff == 0:
            return 1.0
        elif year_diff == 1:
            return 0.5
        else:
            return 0.0

    def match_by_year(self, premed_year: int) -> List[Dict[str, Any]]:
        """
        Match a pre-med student with medical students based on year preference.

        Args:
            premed_year: Desired year of the medical student, as specified by the pre-med

        Returns:
            List of dictionaries containing medical students and match scores, where:
            - 1.0 = exact match to desired year
            - 0.5 = adjacent year (fallback)
            - 0.0 = distant years
        """
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")

        matches = []
        for _, med_student in self.med_students_df.iterrows():
            med_year = self._convert_year(med_student['Q1'])
            score = self.calculate_year_compatibility(
                premed_year,
                med_year
            )

            matches.append({
                "med_student_id": med_student.name,  # Using index as ID
                "score": score,
                "med_student_year": med_student['Q1'],
                # Add more fields as needed
            })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def match_by_gap_year(self, premed_gap: str) -> List[Dict[str, Any]]:
        """
        Match a pre-med student with medical students based on gap year preference.

        Args:
            premed_gap: Desired gap years of the medical student, as specified by the pre-med (Q2)

        Returns:
            List of dictionaries containing medical students and match scores, where:
            - 1.0 = exact match to desired gap years
            - 0.5 = one step apart
            - 0.0 = more than one step apart
        """
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")

        matches = []
        for _, med_student in self.med_students_df.iterrows():
            score = self.gap_year_match(
                premed_gap,
                med_student['Q2']  # Med student's actual gap years
            )

            matches.append({
                "med_student_id": med_student.name,  # Using index as ID
                "score": score,
                "med_student_gap": med_student['Q2'],
                # Add more fields as needed
            })

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def match_student_orgs(self, premed_orgs: str, med_orgs: str) -> float:
        """
        Compare Q6 "student organizations" between a pre-med and a med student.
        - Treat each input string as a comma- or newline-separated list of org names.
        - Normalize by splitting on commas or newlines, stripping whitespace, and lowercasing.
        - If any org appears in both lists exactly, return 1.0; otherwise return 0.0.
        """
        def normalize(orgs):
            if not isinstance(orgs, str):
                return set()
            # Split on commas or newlines, strip whitespace, lowercase
            return set([o.strip().lower() for o in re.split(r'[\n,]', orgs) if o.strip()])
        
        premed_set = normalize(premed_orgs)
        med_set = normalize(med_orgs)
        
        # Use len() on common items instead of set operations in boolean context
        common_items = premed_set.intersection(med_set)
        if len(common_items) > 0:
            return 1.0
        return 0.0

    def match_by_student_orgs(self, premed_orgs: str) -> List[Dict[str, Any]]:
        """
        Loop through all med students, compute match_student_orgs for each, and return
        a sorted list of {med_student_id, score, med_student_orgs}.
        """
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")
        matches = []
        for _, med_student in self.med_students_df.iterrows():
            med_orgs = med_student.get('Q6', "")
            score = self.match_student_orgs(premed_orgs, med_orgs)
            matches.append({
                "med_student_id": med_student.name,
                "score": score,
                "med_student_orgs": med_orgs
            })
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def _normalize_clinical_fields(self, row):
        # Always work on a copy to avoid pandas SettingWithCopyWarning
        row = row.copy() if hasattr(row, 'copy') else dict(row)
        q4 = row.get("Q4", "")
        if isinstance(q4, str):
            row["Q4_list"] = [i.strip().lower() for i in q4.split(",") if i.strip()]
        elif isinstance(q4, list):
            row["Q4_list"] = [str(i).strip().lower() for i in q4 if str(i).strip()]
        else:
            row["Q4_list"] = []
        q4_other_raw = row.get("Q4_18_TEXT", "")
        # Always coerce to string to avoid ambiguous truth-value of a Series
        row["Q4_other"] = str(q4_other_raw).strip().lower()
        return row

    def match_single_premed(self, premed_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Match a single pre-med student against all medical students.
        Args:
            premed_path: Path to the pre-med student's Excel file
        Returns:
            Dictionary containing different types of matches
        """
        # ——— load and normalize the single pre-med row ———
        premed_df = pd.read_excel(premed_path)
        premed_df.columns = premed_df.columns.str.strip()
        raw_row = premed_df.iloc[0]
        premed_row = self._normalize_clinical_fields(raw_row)

        # ——— extract each field via _safe_get so we only ever deal with scalars or lists ———
        # year
        premed_year_str = self._safe_get(premed_row, "Q1")
        premed_year     = self._convert_year(premed_year_str)
        year_matches    = self.match_by_year(premed_year)

        # gap year
        premed_gap_str     = self._safe_get(premed_row, "Q2")
        gap_year_matches   = self.match_by_gap_year(premed_gap_str)

        # undergrad degree
        premed_degree_str  = self._safe_get(premed_row, "Q3")
        undergrad_matches  = self.match_by_undergrad_degree(premed_degree_str)

        # clinical interests
        premed_interests_list = self._safe_get(premed_row, "Q4", to_list=True)
        premed_other_str      = self._safe_get(premed_row, "Q4_18_TEXT")
        clinical_matches      = self.match_by_clinical_interests(premed_interests_list, premed_other_str)

        # student organizations
        premed_orgs_str       = self._safe_get(premed_row, "Q6")
        student_org_matches   = self.match_by_student_orgs(premed_orgs_str)

        # normalize all med-student rows for global index
        self.med_students_df = self.med_students_df.copy()
        for idx, med_row in self.med_students_df.iterrows():
            self.med_students_df.loc[idx] = self._normalize_clinical_fields(med_row)

        global_matches = self.overall_match_index(premed_row)
        return {
            "year_matches": year_matches,
            "gap_year_matches": gap_year_matches,
            "undergrad_matches": undergrad_matches,
            "clinical_matches": clinical_matches,
            "student_org_matches": student_org_matches,
            "global_matches": global_matches
        }

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two strings using sentence embeddings.
        Returns a float between 0.0 and 1.0.
        """
        if not isinstance(text1, str) or not isinstance(text2, str):
            return 0.0
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        sim = util.cos_sim(emb1, emb2)
        return float(sim[0][0])

    def match_undergrad_degree(self, premed_val: str, med_val: str, fuzzy_weight: float = 0.2, semantic_weight: float = 0.55, direct_match_weight: float = 0.25) -> float:
        """
        Compute combined similarity between pre-med and med student's undergraduate degrees.
        Uses fuzzy token_set_ratio, biomedical semantic similarity, and direct substring matching.
        Returns a float between 0.0 and 1.0.
        """
        if not isinstance(premed_val, str) or not isinstance(med_val, str):
            return 0.0
            
        # Normalize inputs
        premed_clean = premed_val.strip().lower()
        med_clean = med_val.strip().lower()
        
        if not premed_clean or not med_clean:
            return 0.0
            
        # 1. Fuzzy string matching score
        fuzzy_score = fuzz.token_set_ratio(premed_clean, med_clean) / 100.0
            
        # 2. Biology-focused semantic similarity (using biomedical embedder instead of general one)
        try:
            emb1 = self.bio_embedder.encode(premed_clean, convert_to_tensor=True)
            emb2 = self.bio_embedder.encode(med_clean, convert_to_tensor=True)
            semantic_score = float(util.cos_sim(emb1, emb2)[0][0])
        except Exception:
            # Fallback to general embedder if bio embedder fails
            semantic_score = self.semantic_similarity(premed_clean, med_clean)
        
        # 3. Direct substring match component
        direct_match_score = 0.0
        
        # Check for direct substring match (either way)
        if premed_clean in med_clean or med_clean in premed_clean:
            direct_match_score = 0.8  # Strong bonus but not perfect
            
        # Check for exact match 
        if premed_clean == med_clean:
            direct_match_score = 1.0  # Perfect match
        
        # 4. Domain specific boosting for science/medicine fields
        # Define related fields that should score higher
        bio_related_keywords = [
            "biology", "physiology", "neuroscience", "biochemistry", 
            "molecular", "cellular", "anatomy", "physio", "biomedical",
            "health science", "life science", "human biology", "bioscience",
            "biological", "biophysics", "biotechnology"
        ]
        
        medicine_related_keywords = [
            "pre-med", "pre med", "premed", "medicine", "medical", "health",
            "nursing", "pharmacy", "public health", "kinesiology", "nutrition",
            "physician", "clinical"
        ]
        
        # More robust tokenization - get both individual words and full phrases
        def get_tokens(text):
            # Single words
            single_words = set(text.split())
            # Add the full text as a token too
            all_tokens = single_words.copy()
            all_tokens.add(text)
            # Add common 2-word and 3-word phrases
            words = text.split()
            for i in range(len(words)-1):
                all_tokens.add(f"{words[i]} {words[i+1]}")
            for i in range(len(words)-2):
                all_tokens.add(f"{words[i]} {words[i+1]} {words[i+2]}")
            return all_tokens
        
        premed_tokens = get_tokens(premed_clean)
        med_tokens = get_tokens(med_clean)
        
        bio_boost = 0.0
        for keyword in bio_related_keywords:
            # Check if keyword appears in either degree
            if keyword in premed_clean and keyword in med_clean:
                bio_boost = max(bio_boost, 0.3)  # Bonus for shared biology keywords
                
        med_boost = 0.0
        for keyword in medicine_related_keywords:
            if keyword in premed_clean and keyword in med_clean:
                med_boost = max(med_boost, 0.2)  # Bonus for shared medical keywords
  
        # ---- Domain‑group mapping boost ----
        DOMAIN_GROUPS = {
            "life_sci": {
                "biology", "biological", "biomed", "biomedical", "physiology",
                "neuroscience", "biochemistry", "molecular", "cellular",
                "anatomy", "health", "kinesiology", "nutrition", "biophysics",
                "human", "life", "bioscience", "human biology"
            },
            "social_sci": {
                "sociology", "psychology", "anthropology", "social"
            },
            "physical_sci": {
                "physics", "chemistry", "mathematics", "math", "statistics", "statistical"
            },
            "engineering": {
                "engineering", "computer", "electrical", "mechanical", "civil"
            }
        }

        premed_domain = None
        med_domain = None
        
        # Check against all tokens including multi-word phrases
        for dom, keywords in DOMAIN_GROUPS.items():
            if any(any(tok in keyword or keyword in tok for keyword in keywords) for tok in premed_tokens):
                premed_domain = dom
                    
            if any(any(tok in keyword or keyword in tok for keyword in keywords) for tok in med_tokens):
                med_domain = dom

        domain_boost = 0.0
        if premed_domain and premed_domain == med_domain:
            # Strong boost when both degrees fall in the same broad domain
            domain_boost = 0.25
        elif premed_domain and med_domain and {premed_domain, med_domain}.issubset({"life_sci", "physical_sci"}):
            # Moderate boost when both are sciences but different branches
            domain_boost = 0.15
                
        # Add explicit check for 'human biology' vs 'physiology' - ensure this specific pair gets a good match
        human_bio_physio_boost = 0.0
        if ("human biology" in premed_clean and "physiology" in med_clean) or \
           ("physiology" in premed_clean and "human biology" in med_clean):
            human_bio_physio_boost = 0.3  # Strong boost for this specific pair
                
        # Add explicit penalty for 'sociology' when matching with biology terms
        sociology_penalty = 0.0
        if ("sociology" in premed_clean or "sociology" in med_clean) and \
           any(bio_term in premed_clean or bio_term in med_clean 
               for bio_term in ["biology", "physiology", "biomedical", "health science"]):
            sociology_penalty = -0.4  # Strong penalty for this mismatch
        
        # Apply all scores and adjustments
        score = (fuzzy_weight * fuzzy_score + 
                semantic_weight * semantic_score + 
                direct_match_weight * direct_match_score)
                
        # Add boosts (including the new domain_boost), but cap at 1.0
        final_score = min(1.0, max(0.0, score + bio_boost + med_boost + domain_boost + human_bio_physio_boost + sociology_penalty))
        
        # Print component scores
        print(f"\nUndergrad Degree Component Scores:")
        print(f"  • Fuzzy Score: {fuzzy_score:.3f}")
        print(f"  • Semantic Score: {semantic_score:.3f}")
        print(f"  • Direct Match Score: {direct_match_score:.3f}")
        print(f"  • Bio Boost: {bio_boost:.3f}")
        print(f"  • Med Boost: {med_boost:.3f}")
        print(f"  • Domain Boost: {domain_boost:.3f}")
        print(f"  • Human Bio/Physio Boost: {human_bio_physio_boost:.3f}")
        print(f"  • Sociology Penalty: {sociology_penalty:.3f}")
        print(f"  • Final Weighted Score: {final_score:.3f}")
        
        return final_score

    def match_by_undergrad_degree(self, premed_degree: str) -> List[Dict[str, any]]:
        """
        Match a pre-med student with medical students based on undergrad degree similarity (Q3).
        Returns a sorted list of matches with med_student_id, med_student_degree, and score.
        """
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")
        matches = []
        for _, med_student in self.med_students_df.iterrows():
            med_degree = med_student['Q3']
            score = self.match_undergrad_degree(premed_degree, med_degree)
            matches.append({
                "med_student_id": med_student.name,
                "med_student_degree": med_degree,
                "score": score
            })
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def _normalize_to_set(self, values: List[str], other: str) -> set:
        import re
        import pandas as pd
        if isinstance(values, pd.Series):
            values = values.tolist()
        if not isinstance(values, list):
            values = []
        if not isinstance(other, str):
            other = ""
        cleaned_terms = set()
        # Only include 'other' free text if it is not empty and not just a placeholder
        other = other.strip()
        if other and not re.match(r'^other[:\-\s\.]*$', other, flags=re.I):
            values = values + [other]
        for raw_term in values:
            if not isinstance(raw_term, str):
                continue
            for t in raw_term.split(","):
                t = t.strip().casefold()
                # Remove if empty, just dots, or any form of 'other' placeholder
                if not t or t == '...' or re.match(r'^other[:\-\s\.]*$', t, flags=re.I):
                    continue
                cleaned_terms.add(t)
        return cleaned_terms

    def match_clinical_interests(self, premed_vals: List[str], premed_other: str, med_vals: List[str], med_other: str) -> float:
        """
        Compare clinical interests between pre-med and med student using a three-part blend: exact term overlap (Jaccard), fuzzy match, and BiomedBERT semantic similarity.
        """
        premed_set = self._normalize_to_set(premed_vals, premed_other)
        med_set = self._normalize_to_set(med_vals, med_other)
        if len(premed_set) == 0 or len(med_set) == 0:
            return 0.0
        # Exact term overlap via Jaccard
        intersection = premed_set.intersection(med_set)
        union = premed_set.union(med_set)
        jaccard_score = len(intersection) / len(union) if union else 0.0
        # Convert sets to strings
        premed_text = " ".join(sorted(premed_set))
        med_text = " ".join(sorted(med_set))
        # Fuzzy token-set ratio
        fuzzy_score = fuzz.token_set_ratio(premed_text, med_text) / 100.0
        # Semantic similarity with BiomedBERT
        emb1 = self.bio_embedder.encode(premed_text, convert_to_tensor=True)
        emb2 = self.bio_embedder.encode(med_text, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(emb1, emb2)[0][0])
        # Blend equally
        final_score = (jaccard_score + fuzzy_score + semantic_score) / 3.0

        # Print component scores
        print(f"\nClinical Interests Component Scores:")
        print(f"  • Jaccard Score: {jaccard_score:.3f}")
        print(f"  • Fuzzy Score: {fuzzy_score:.3f}")
        print(f"  • Semantic Score: {semantic_score:.3f}")
        print(f"  • Final Weighted Score: {final_score:.3f}")

        return final_score

    def match_by_clinical_interests(self, premed_interests: List[str], premed_other: str) -> List[Dict[str, Any]]:
        """
        Match a pre-med student with medical students based on clinical interests.
        
        Args:
            premed_interests: List of pre-med's selected clinical interests (Q4)
            premed_other: Pre-med's other clinical interests (Q4_18_TEXT)
            
        Returns:
            List of dictionaries containing medical students and match scores
        """
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")
            
        matches = []
        for _, med_student in self.med_students_df.iterrows():
            # Get med student's clinical interests
            med_interests = med_student['Q4']
            med_other_raw = med_student.get('Q4_18_TEXT', "")
            med_other = str(med_other_raw).strip().lower()
            # Convert med_interests to list if it's a string (comma-separated)
            if isinstance(med_interests, str):
                med_interests = [i.strip() for i in med_interests.split(',') if i.strip()]
            elif not isinstance(med_interests, list):
                med_interests = []
            else:
                med_interests = [str(i).strip() for i in med_interests if pd.notna(i) and str(i).strip()]
            # Normalize and combine med_interests and med_other for display
            normalized_set = self._normalize_to_set(med_interests, med_other)
            display_set = sorted(normalized_set)
            # Calculate match score
            score = self.match_clinical_interests(
                premed_interests,
                premed_other,
                med_interests,
                med_other
            )
            matches.append({
                "med_student_id": med_student.name,
                "score": score,
                "med_student_interests": display_set,  # Show only real interests
                "med_student_other": med_other
            })
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def _extract_keywords(self, text: str) -> set:
        """
        Extracts keywords from biomedical text by removing stopwords and punctuation.
        """
        if not isinstance(text, str):
            return set()
        # Lowercase and remove punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        # Simple stopword list (expand as needed)
        stopwords = set([
            'the', 'and', 'or', 'in', 'on', 'at', 'for', 'to', 'of', 'a', 'an', 'is', 'are', 'with', 'by', 'as', 'but', 'if', 'i', 'am', 'not', 'be', 'do', 'does', 'was', 'were', 'this', 'that', 'it', 'from', 'so', 'my', 'we', 'you', 'your', 'our', 'their', 'they', 'he', 'she', 'his', 'her', 'have', 'has', 'had', 'will', 'would', 'can', 'could', 'should', 'may', 'might', 'looking', 'moment', 'currently', 'opportunities', 'involved', 'research', 'interested', 'interest', 'related', 'field', 'area', 'areas', 'etc', 'etc.'
        ])
        tokens = [w for w in text.split() if w not in stopwords and len(w) > 2]
        return set(tokens)

    def _safe_get(self, row, key, default="", to_list=False):
        """
        Safely get a value from a row, handling all edge cases with
        pandas Series, missing values, etc.
        """
        import pandas as pd
        
        # Handle all possible ways to get a value (dot vs bracket vs get)
        try:
            if hasattr(row, 'get'):
                val = row.get(key, None)
            elif hasattr(row, key):
                val = getattr(row, key)
            else:
                try:
                    val = row[key]
                except (KeyError, TypeError):
                    val = default
        except Exception:
            val = default
            
        # Handle Series - CRITICAL for avoiding ambiguous truth value errors
        if isinstance(val, pd.Series):
            print(f"DEBUG: Series encountered for key '{key}': {val}")
            val = val.iloc[0] if not val.empty else default
            
        # Handle None/NaN
        if pd.isna(val):
            val = default
            
        # Return as list or string
        if to_list:
            if isinstance(val, str):
                return [i.strip().lower() for i in val.split(',') if i.strip()]
            elif isinstance(val, list):
                return [str(i).strip().lower() for i in val if pd.notna(i) and str(i).strip()]
            else:
                return []
        else:
            return str(val).strip().lower() if val is not None else default

    def match_by_research_interests(self, premed_research: str) -> List[Dict[str, Any]]:
        """
        Hybrid: Combine BiomedBERT embedding similarity, Jaccard keyword overlap, fuzzy matching, and TF-IDF similarity,
        with a direct substring match boost (case-insensitive).
        """
        # More robust handling of input
        if not isinstance(premed_research, str):
            premed_research = str(premed_research)
        
        premed_clean = premed_research.strip().lower()
        
        # Fix this boolean check to be defensive
        if len(premed_clean) == 0 or "n/a" in premed_clean:
            return []
        
        if self.med_students_df is None:
            raise ValueError("Medical students data not loaded. Call load_data() first.")
            
        matches = []
        premed_keywords = self._extract_keywords(premed_clean)
        
        try:
            emb1 = self.bio_embedder.encode(premed_clean, convert_to_tensor=True)
        except Exception as e:
            print(f"Error encoding premed research: {e}")
            return []
            
        for _, med_student in self.med_students_df.iterrows():
            # Safe extraction
            med_research = self._safe_get(med_student, 'Q5', "")
            
            # No need for boolean checks on med_clean
            med_clean = med_research.strip().lower()
            if len(med_clean) == 0 or med_clean == "n/a":
                continue
                
            med_keywords = self._extract_keywords(med_clean)
            
            # Jaccard keyword overlap - avoid boolean checks
            if len(premed_keywords) > 0 or len(med_keywords) > 0:
                all_keywords = premed_keywords.union(med_keywords)
                common_keywords = premed_keywords.intersection(med_keywords)
                jaccard = len(common_keywords) / len(all_keywords) if len(all_keywords) > 0 else 0.0
            else:
                jaccard = 0.0
                
            # BiomedBERT embedding similarity
            try:
                emb2 = self.bio_embedder.encode(med_clean, convert_to_tensor=True)
                embedding_score = float(util.cos_sim(emb1, emb2)[0][0])
            except Exception:
                embedding_score = 0.0
            
            # Fuzzy matching score
            fuzzy_score = fuzz.token_set_ratio(premed_clean, med_clean) / 100.0
            
            # TF-IDF similarity 
            tfidf_score = self._compute_tfidf_similarity(premed_clean, med_clean)
                
            # Weighted hybrid score - match weights in match_research_interest_pair
            score = 0.6 * embedding_score + 0.1 * jaccard + 0.1 * fuzzy_score + 0.2 * tfidf_score
            
            # Relevance filter: require some keyword overlap OR a reasonable fuzzy match
            # This prevents completely unrelated fields from matching just because of embedding similarity
            if jaccard == 0.0 and fuzzy_score < 0.4 and tfidf_score < 0.3:
                score = score * 0.2  # Apply a stronger penalty when no meaningful overlap
            
            # Direct substring match boost (case-insensitive)
            if premed_clean and premed_clean in med_clean:
                score += 0.3  # Significant boost for direct match
                
            score = min(score, 1.0)  # Cap at 1.0
            
            matches.append({
                "med_student_id": med_student.name,
                "med_student_research": med_research,
                "score": score
            })
            
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def _motivation_keywords(self, text: str) -> set:
        # Small domain-specific lexicon
        lexicon = [
            'service', 'research', 'patient', 'advocacy', 'discovery', 'community',
            'science', 'help', 'care', 'equity', 'diversity', 'leadership', 'education',
            'curiosity', 'healing', 'compassion', 'innovation', 'teamwork', 'impact',
            'medicine', 'health', 'wellness', 'support', 'family', 'experience', 'challenge',
            'growth', 'learning', 'change', 'difference', 'doctor', 'physician', 'mentor',
            'role model', 'volunteer', 'underserved', 'access', 'prevention', 'treatment',
            'diagnosis', 'empathy', 'motivation', 'drive', 'passion', 'calling', 'purpose'
        ]
        # Preprocess
        if not isinstance(text, str):
            return set()
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = set(text.split())
        return set([word for word in tokens if word in lexicon])

    def match_by_motivation_essay(self, premed_essay: str) -> List[Dict[str, Any]]:
        """
        Hybrid: Combine general-purpose embedding similarity (all-mpnet-base-v2), Jaccard overlap on motivation keywords, and TF-IDF cosine similarity.
        """
        if not isinstance(premed_essay, str) or not premed_essay.strip():
            return []
        premed_clean = premed_essay.strip().lower()
        # 1. Motivation keywords
        premed_keywords = self._motivation_keywords(premed_clean)
        # 2. Embedding (general-purpose)
        emb1 = self.embedding_model.encode(premed_clean, convert_to_tensor=True)
        # 3. Prepare for TF-IDF
        med_essays = []
        med_ids = []
        for _, med_student in self.med_students_df.iterrows():
            med_essay = med_student.get('Q7', None)
            if isinstance(med_essay, str) and med_essay.strip():
                med_essays.append(med_essay.strip().lower())
                med_ids.append(med_student.name)
        # Fit TF-IDF on all essays (including premed)
        tfidf = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf_matrix = tfidf.fit_transform([premed_clean] + med_essays)
        premed_tfidf = tfidf_matrix[0]
        med_tfidfs = tfidf_matrix[1:]
        matches = []
        for idx, med_essay in enumerate(med_essays):
            med_id = med_ids[idx]
            # 1. Jaccard on motivation keywords
            med_keywords = self._motivation_keywords(med_essay)
            if premed_keywords or med_keywords:
                jaccard = len(premed_keywords & med_keywords) / len(premed_keywords | med_keywords) if (premed_keywords | med_keywords) else 0.0
            else:
                jaccard = 0.0
            # 2. Embedding similarity (general-purpose)
            emb2 = self.embedding_model.encode(med_essay, convert_to_tensor=True)
            embedding_score = float(util.cos_sim(emb1, emb2)[0][0])
            # 3. TF-IDF cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_score = float(cosine_similarity(premed_tfidf, med_tfidfs[idx])[0][0])
            # Hybrid score
            score = 0.5 * embedding_score + 0.3 * jaccard + 0.2 * tfidf_score
            matches.append({
                "med_student_id": med_id,
                "med_student_essay": med_essay,
                "score": score
            })
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

    def _component_scores(self, premed_row, med_row) -> dict[str, float]:
        premed_row = self._normalize_clinical_fields(premed_row)
        med_row = self._normalize_clinical_fields(med_row)
        return {
            "year":       self.calculate_year_compatibility(
                              self._convert_year(self._safe_get(premed_row, "Q1")),
                              self._convert_year(self._safe_get(med_row, "Q1"))
                          ),
            "gap":        self.gap_year_match(self._safe_get(premed_row, "Q2"), self._safe_get(med_row, "Q2")),
            "degree":     self.match_undergrad_degree(self._safe_get(premed_row, "Q3"), self._safe_get(med_row, "Q3")),
            "clinical":   self.match_clinical_interests(
                              self._safe_get(premed_row, "Q4", to_list=True), self._safe_get(premed_row, "Q4_18_TEXT"),
                              self._safe_get(med_row, "Q4", to_list=True),   self._safe_get(med_row, "Q4_18_TEXT")
                          ),
            "research":   self.match_research_interest_pair(
                              self._safe_get(premed_row, "Q5"), self._safe_get(med_row, "Q5")
                          ),
            "motivation": self.semantic_similarity(self._safe_get(premed_row, "Q7"), self._safe_get(med_row, "Q7")),
            "orgs":       self.match_student_orgs(self._safe_get(premed_row, "Q6"), self._safe_get(med_row, "Q6")),
        }

    def overall_match_index(self, premed_row) -> list[dict]:
        """
        Return med students sorted by global match index (0-1).
        Each entry also contains component scores for transparency.
        """
        if self.med_students_df is None:
            raise ValueError("Call load_data() first.")
        W = _active_weights()
        results = []
        for _, med_row in self.med_students_df.iterrows():
            scores = self._component_scores(premed_row, med_row)
            index  = round(_safe_weighted_sum(scores, W), 4)
            results.append({"med_student_id": med_row.name, "index": index, **scores})
        results.sort(key=lambda x: x["index"], reverse=True)
        return results

    def match_research_interest_pair(self, premed_research: str, med_research: str) -> float:
        """
        Compute similarity score between a premed and med student's research interests using the same logic as match_by_research_interests, but for a single pair.
        """
        # Defensive: always convert to string if not already
        if not isinstance(premed_research, str):
            premed_clean = str(premed_research).strip().lower()
        else:
            premed_clean = premed_research.strip().lower()
        if not isinstance(med_research, str):
            med_clean = str(med_research).strip().lower()
        else:
            med_clean = med_research.strip().lower()
        if not premed_clean or not med_clean or premed_clean == "n/a" or med_clean == "n/a":
            return 0.0
        premed_keywords = self._extract_keywords(premed_clean)
        med_keywords = self._extract_keywords(med_clean)
        # Jaccard keyword overlap
        if premed_keywords or med_keywords:
            jaccard = len(premed_keywords & med_keywords) / len(premed_keywords | med_keywords) if (premed_keywords | med_keywords) else 0.0
        else:
            jaccard = 0.0
        # BiomedBERT embedding similarity
        emb1 = self.bio_embedder.encode(premed_clean, convert_to_tensor=True)
        emb2 = self.bio_embedder.encode(med_clean, convert_to_tensor=True)
        embedding_score = float(util.cos_sim(emb1, emb2)[0][0])
        # Fuzzy string matching score
        fuzzy_score = fuzz.token_set_ratio(premed_clean, med_clean) / 100.0
        # TF-IDF similarity
        tfidf_score = self._compute_tfidf_similarity(premed_clean, med_clean)
        
        # Weighted hybrid score - now with 4 components
        score = 0.6 * embedding_score + 0.1 * jaccard + 0.1 * fuzzy_score + 0.2 * tfidf_score
        
        # Relevance filter: require some keyword overlap OR a reasonable fuzzy match
        # This prevents completely unrelated fields from matching just because of embedding similarity
        if jaccard == 0.0 and fuzzy_score < 0.4 and tfidf_score < 0.3:
            score = score * 0.2  # Apply a stronger penalty when no meaningful overlap
        
        # Direct substring match boost (case-insensitive)
        if premed_clean and premed_clean in med_clean:
            score += 0.3  # Significant boost for direct match
        score = min(score, 1.0)  # Cap at 1.0

        # Print component scores
        print(f"\nResearch Interest Component Scores:")
        print(f"  • Jaccard Score: {jaccard:.3f}")
        print(f"  • Embedding Score: {embedding_score:.3f}")
        print(f"  • Fuzzy Score: {fuzzy_score:.3f}")
        print(f"  • TF-IDF Score: {tfidf_score:.3f}")
        print(f"  • Final Weighted Score: {score:.3f}")

        return score
