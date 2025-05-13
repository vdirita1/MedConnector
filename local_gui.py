import sys
import subprocess
import pandas as pd

REQUIRED_PACKAGES = [
    "sentence_transformers",
    "rapidfuzz",
    "pandas",
    "scikit-learn"
]

missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        if pkg == "sentence_transformers":
            import sentence_transformers
        elif pkg == "rapidfuzz":
            import rapidfuzz
        elif pkg == "pandas":
            import pandas
        elif pkg == "scikit-learn":
            import sklearn
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing required packages: {', '.join(missing)}")
    response = input("Would you like to install them now? [y/N]: ").strip().lower()
    if response == "y":
        import sys, subprocess
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        print("Dependencies installed. Please re-run the script.")
        sys.exit(0)
    else:
        print("Please install the missing packages and re-run the script.")
        sys.exit(1)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from matching import MatchingService, DEFAULT_WEIGHTS

class MatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MedConnector Local Matcher")
        self.matcher = MatchingService()
        self.premed_file = None
        self.medstudent_file = None
        self.results = None
        
        # Weight descriptions dictionary for use throughout the app
        self.weight_descriptions = {
            "year": "Year (M1-M4)",
            "gap": "Gap Year",
            "degree": "Undergrad Degree",
            "clinical": "Clinical Interests",
            "research": "Research Interests",
            "motivation": "Motivation Essay",
            "orgs": "Student Organizations"
        }
        
        self.create_widgets()

    def create_widgets(self):
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(padx=20, pady=10)

        tk.Label(file_frame, text="PreMed Excel File:").grid(row=0, column=0, sticky="e")
        self.premed_entry = tk.Entry(file_frame, width=40)
        self.premed_entry.grid(row=0, column=1)
        tk.Button(file_frame, text="Browse", command=self.browse_premed).grid(row=0, column=2)

        tk.Label(file_frame, text="MedStudent Excel File:").grid(row=1, column=0, sticky="e")
        self.medstudent_entry = tk.Entry(file_frame, width=40)
        self.medstudent_entry.grid(row=1, column=1)
        tk.Button(file_frame, text="Browse", command=self.browse_medstudent).grid(row=1, column=2)

        # Add weights configuration section
        weights_frame = tk.LabelFrame(self.root, text="Matching Weights")
        weights_frame.pack(padx=20, pady=10, fill="x")
        
        # Get default weights from the matcher
        self.weights = DEFAULT_WEIGHTS.copy()
        
        # Weight sliders with labels
        self.weight_vars = {}
        self.percentage_vars = {}  # Move this outside the loop
        
        # Create sliders for each weight
        for i, (key, value) in enumerate(self.weights.items()):
            # Row with label and percentage display
            tk.Label(weights_frame, text=self.weight_descriptions.get(key, key)).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            # Create StringVar to track percentage
            percentage_var = tk.StringVar(value=f"{int(value*100)}%")
            self.percentage_vars[key] = percentage_var  # Store in class dictionary
            
            # Create slider variable 
            slider_var = tk.DoubleVar(value=value)
            self.weight_vars[key] = slider_var
            
            # Create and place slider
            slider = tk.Scale(weights_frame, from_=0.0, to=1.0, resolution=0.05, 
                             orient=tk.HORIZONTAL, variable=slider_var, showvalue=False,
                             length=200)
            slider.grid(row=i, column=1, padx=5, pady=2)
            
            # Set up a callback for this specific slider
            def update_percentage(val, k=key):
                self.weights[k] = float(val)
                self.percentage_vars[k].set(f"{int(float(val)*100)}%")
                
            slider.config(command=update_percentage)
            
            # Display percentage
            tk.Label(weights_frame, textvariable=percentage_var, width=5).grid(row=i, column=2, padx=5, pady=2)
        
        # Buttons row for weights
        buttons_frame = tk.Frame(weights_frame)
        buttons_frame.grid(row=len(self.weights), column=0, columnspan=3, pady=10)
        
        # Reset weights button
        def reset_weights():
            for key, value in DEFAULT_WEIGHTS.items():
                self.weight_vars[key].set(value)
                self.weights[key] = value
                self.percentage_vars[key].set(f"{int(value*100)}%")
            
        tk.Button(buttons_frame, text="Reset to Defaults", command=reset_weights).pack(side=tk.LEFT, padx=5)
        
        # Normalize weights button
        def normalize_weights():
            # Get current sum of weights
            total = sum(self.weight_vars[key].get() for key in self.weights)
            if total == 0:
                messagebox.showwarning("Warning", "All weights are zero. Cannot normalize.")
                return
                
            # Normalize to sum to 1.0
            for key in self.weights:
                normalized = self.weight_vars[key].get() / total
                self.weight_vars[key].set(normalized)
                self.weights[key] = normalized
                self.percentage_vars[key].set(f"{int(normalized*100)}%")
                
        tk.Button(buttons_frame, text="Normalize Weights", command=normalize_weights).pack(side=tk.LEFT, padx=5)
        
        # Run matching button below the weights
        tk.Button(self.root, text="Run Matching", command=self.run_matching, 
                 bg="#4CAF50", fg="black", font=("Helvetica", 12, "bold"), 
                 padx=20, pady=10).pack(pady=10)

        # Results frame with scrollbar
        container = tk.Frame(self.root)
        container.pack(padx=20, pady=10, fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def resize_canvas(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", resize_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Use scrollable_frame instead of results_frame for all match sections
        # Year matches
        year_frame = tk.LabelFrame(scrollable_frame, text="Year Matches")
        year_frame.pack(fill="x", expand=True, pady=5)
        self.year_premed_label = tk.Label(year_frame, text="Premed answer: ")
        self.year_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.year_tree = ttk.Treeview(year_frame, columns=("med_student_id", "score", "med_student_year"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_year"):
            self.year_tree.heading(col, text=col)
            self.year_tree.column(col, width=120, stretch=True)
        self.year_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Gap year matches
        gap_frame = tk.LabelFrame(scrollable_frame, text="Gap Year Matches")
        gap_frame.pack(fill="x", expand=True, pady=5)
        self.gap_premed_label = tk.Label(gap_frame, text="Premed answer: ")
        self.gap_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.gap_tree = ttk.Treeview(gap_frame, columns=("med_student_id", "score", "med_student_gap"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_gap"):
            self.gap_tree.heading(col, text=col)
            self.gap_tree.column(col, width=120, stretch=True)
        self.gap_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Undergrad degree matches
        undergrad_frame = tk.LabelFrame(scrollable_frame, text="Undergrad Degree Matches")
        undergrad_frame.pack(fill="x", expand=True, pady=5)
        self.undergrad_premed_label = tk.Label(undergrad_frame, text="Premed answer: ")
        self.undergrad_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.undergrad_tree = ttk.Treeview(undergrad_frame, columns=("med_student_id", "score", "med_student_degree"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_degree"):
            self.undergrad_tree.heading(col, text=col)
            self.undergrad_tree.column(col, width=120, stretch=True)
        self.undergrad_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Clinical interests matches
        clinical_frame = tk.LabelFrame(scrollable_frame, text="Clinical Interests Matches")
        clinical_frame.pack(fill="x", expand=True, pady=5)
        self.clinical_premed_label = tk.Label(clinical_frame, text="Premed answer: ")
        self.clinical_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.clinical_tree = ttk.Treeview(clinical_frame, columns=("med_student_id", "score", "med_student_interests"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_interests"):
            self.clinical_tree.heading(col, text=col)
            self.clinical_tree.column(col, width=120, stretch=True)
        self.clinical_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Student organization matches
        org_frame = tk.LabelFrame(scrollable_frame, text="Student Organization Matches")
        org_frame.pack(fill="x", expand=True, pady=5)
        self.org_premed_label = tk.Label(org_frame, text="Premed answer: ")
        self.org_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.org_tree = ttk.Treeview(org_frame, columns=("med_student_id", "score", "med_student_orgs"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_orgs"):
            self.org_tree.heading(col, text=col)
            self.org_tree.column(col, width=120, stretch=True)
        self.org_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Research interests matches
        research_frame = tk.LabelFrame(scrollable_frame, text="Research Interests Matches")
        research_frame.pack(fill="x", expand=True, pady=5)
        self.research_premed_label = tk.Label(research_frame, text="Premed answer: ")
        self.research_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.research_tree = ttk.Treeview(research_frame, columns=("med_student_id", "score", "med_student_research"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_research"):
            self.research_tree.heading(col, text=col)
            self.research_tree.column(col, width=120, stretch=True)
        self.research_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Motivation essay matches
        motivation_frame = tk.LabelFrame(scrollable_frame, text="Motivation Essay Matches")
        motivation_frame.pack(fill="x", expand=True, pady=5)
        self.motivation_premed_label = tk.Label(motivation_frame, text="Premed answer: ")
        self.motivation_premed_label.pack(anchor="w", padx=5, pady=(2,0))
        self.motivation_tree = ttk.Treeview(motivation_frame, columns=("med_student_id", "score", "med_student_essay"), show="headings", height=5)
        for col in ("med_student_id", "score", "med_student_essay"):
            self.motivation_tree.heading(col, text=col)
            self.motivation_tree.column(col, width=120, stretch=True)
        self.motivation_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Overall Match Index
        global_frame = tk.LabelFrame(scrollable_frame, text="Overall Match Index")
        global_frame.pack(fill="x", expand=True, pady=5)
        
        # Add label to show weights used for matching
        self.global_weights_label = tk.Label(global_frame, text="Weights used: ", anchor="w", wraplength=800, justify="left")
        self.global_weights_label.pack(anchor="w", padx=5, pady=(2,0))
        
        self.global_tree = ttk.Treeview(
            global_frame,
            columns=("med_student_id", "index", "year", "gap", "degree", "clinical", "research", "motivation", "orgs"),
            show="headings",
            height=5
        )
        for col in ("med_student_id", "index", "year", "gap", "degree", "clinical", "research", "motivation", "orgs"):
            self.global_tree.heading(col, text=col)
            self.global_tree.column(col, width=90, stretch=True)
        self.global_tree.pack(padx=5, pady=5, fill="x", expand=True)

        # Export button
        tk.Button(self.root, text="Export Results to CSV", command=self.export_results).pack(pady=5)

    def browse_premed(self):
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file:
            self.premed_file = file
            self.premed_entry.delete(0, tk.END)
            self.premed_entry.insert(0, file)

    def browse_medstudent(self):
        file = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file:
            self.medstudent_file = file
            self.medstudent_entry.delete(0, tk.END)
            self.medstudent_entry.insert(0, file)

    def run_matching(self):
        if not self.premed_file or not self.medstudent_file:
            messagebox.showerror("Error", "Please select both Excel files.")
            return
        try:
            # Show a "matching in progress" message with selected weights
            weights_str = ", ".join([f"{k}: {int(v*100)}%" for k, v in self.weights.items()])
            self.root.title(f"MedConnector Local Matcher - Running... (Weights: {weights_str})")
            self.root.update()  # Force UI update to show new title
            
            # Load data and run basic matching
            self.matcher.load_data(self.medstudent_file)
            
            # Set custom weights (this will override any environment variables)
            import os
            import json
            os.environ["MATCH_WEIGHTS"] = json.dumps(self.weights)
            
            # Run matching with the custom weights
            self.results = self.matcher.match_single_premed(self.premed_file)
            
            # ——— Pull premed research interests and motivation essay safely ———
            premed_df = pd.read_excel(self.premed_file)
            premed_df.columns = premed_df.columns.str.strip()
            raw_row = premed_df.iloc[0]
            self.last_premed_row = raw_row  # Save for display
            
            # Safely extract research interest
            premed_research = raw_row.get('Q5', "")
            # Convert Series to scalar if needed
            if isinstance(premed_research, pd.Series):
                premed_research = premed_research.iloc[0] if not premed_research.empty else ""
            premed_research = str(premed_research) if not pd.isna(premed_research) else ""
            
            # Safely extract motivation essay
            premed_essay = raw_row.get('Q7', "")
            # Convert Series to scalar if needed
            if isinstance(premed_essay, pd.Series):
                premed_essay = premed_essay.iloc[0] if not premed_essay.empty else ""
            premed_essay = str(premed_essay) if not pd.isna(premed_essay) else ""
            
            # Only run these matches if we have valid data
            if premed_research:
                self.results['research_matches'] = self.matcher.match_by_research_interests(premed_research)
            else:
                self.results['research_matches'] = []
                
            if premed_essay:
                self.results['motivation_matches'] = self.matcher.match_by_motivation_essay(premed_essay)
            else:
                self.results['motivation_matches'] = []
                
            # Reset title and display results
            self.root.title("MedConnector Local Matcher")
            self.display_results()
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to run matching: {e}\n\n{traceback.format_exc()}"
            print(error_msg)  # Print to console for debugging
            messagebox.showerror("Error", error_msg)
            self.root.title("MedConnector Local Matcher")

    def display_results(self):
        # Clear existing results
        for tree in [self.global_tree, self.year_tree, self.gap_tree, self.undergrad_tree, self.clinical_tree, self.org_tree, self.research_tree, self.motivation_tree]:
            for row in tree.get_children():
                tree.delete(row)

        if not self.results:
            return

        # Show premed answers at the top of each section
        premed = getattr(self, 'last_premed_row', None)
        if premed is not None:
            # Safely display premed values - handle Series
            def safe_get(row, key, default=""):
                val = row.get(key, default)
                if isinstance(val, pd.Series):
                    val = val.iloc[0] if not val.empty else default
                return str(val) if not pd.isna(val) else default
                
            self.year_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q1')}")
            self.gap_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q2')}")
            self.undergrad_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q3')}")
            
            clin = safe_get(premed, 'Q4')
            clin_other = safe_get(premed, 'Q4_18_TEXT')
            clin_disp = clin
            if clin_other:
                clin_disp = f"{clin} (Other: {clin_other})"
            self.clinical_premed_label.config(text=f"Premed answer: {clin_disp}")
            
            self.org_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q6')}")
            self.research_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q5')}")
            self.motivation_premed_label.config(text=f"Premed answer: {safe_get(premed, 'Q7')}")

        # Show weights used for this match in the global section
        weights_str = ", ".join([f"{self.weight_descriptions.get(k, k)}: {int(v*100)}%" for k, v in self.weights.items()])
        self.global_weights_label.config(text=f"Weights used: {weights_str}")

        # Display global match index
        for match in self.results.get('global_matches', []):
            # Extract values safely
            def safe_value(match_dict, key, default=""):
                val = match_dict.get(key, default)
                if isinstance(val, pd.Series):
                    val = val.iloc[0] if not val.empty else default
                return val
            
            self.global_tree.insert('', 'end', values=(
                safe_value(match, 'med_student_id', ''),
                safe_value(match, 'index', ''),
                safe_value(match, 'year', ''),
                safe_value(match, 'gap', ''),
                safe_value(match, 'degree', ''),
                safe_value(match, 'clinical', ''),
                safe_value(match, 'research', ''),
                safe_value(match, 'motivation', ''),
                safe_value(match, 'orgs', '')
            ))

        # Display year matches
        for match in self.results['year_matches']:
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            year = match.get('med_student_year', '')
            if isinstance(year, pd.Series):
                year = year.iloc[0] if not year.empty else ''
                
            self.year_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                year
            ))

        # Display gap year matches
        for match in self.results['gap_year_matches']:
            # Extract values safely 
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            gap = match.get('med_student_gap', '')
            if isinstance(gap, pd.Series):
                gap = gap.iloc[0] if not gap.empty else ''
                
            if pd.isna(gap):
                gap = "None"
                
            self.gap_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                gap
            ))

        # Display undergrad degree matches
        for match in self.results['undergrad_matches']:
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            degree = match.get('med_student_degree', '')
            if isinstance(degree, pd.Series):
                degree = degree.iloc[0] if not degree.empty else ''
                
            self.undergrad_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                degree
            ))

        # Display clinical interests matches
        for match in self.results['clinical_matches']:
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            interests = match.get('med_student_interests', [])
            
            # Handle interests as safely as possible
            interests_str = ""
            if isinstance(interests, list):
                interests_str = ', '.join(interests)
            elif isinstance(interests, pd.Series):
                interests_str = ', '.join(interests.tolist()) if not interests.empty else ""
            elif pd.isna(interests):
                interests_str = ""
            else:
                interests_str = str(interests)
                
            self.clinical_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                interests_str
            ))

        # Display student organization matches
        for match in self.results.get('student_org_matches', []):
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            orgs = match.get('med_student_orgs', '')
            
            # Handle orgs as safely as possible
            orgs_str = ""
            if isinstance(orgs, list):
                orgs_str = ', '.join(orgs)
            elif isinstance(orgs, pd.Series):
                orgs_str = ', '.join(orgs.tolist()) if not orgs.empty else ""
            elif pd.isna(orgs):
                orgs_str = ""
            else:
                orgs_str = str(orgs)
                
            self.org_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                orgs_str
            ))

        # Display research interests matches
        for match in self.results.get('research_matches', []):
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            research = match.get('med_student_research', '')
            if isinstance(research, pd.Series):
                research = research.iloc[0] if not research.empty else ''
                
            self.research_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                research
            ))

        # Display motivation essay matches
        for match in self.results.get('motivation_matches', []):
            # Extract values safely
            med_id = match.get('med_student_id', '')
            if isinstance(med_id, pd.Series):
                med_id = med_id.iloc[0] if not med_id.empty else ''
                
            score = match.get('score', 0)
            if isinstance(score, pd.Series):
                score = score.iloc[0] if not score.empty else 0
                
            essay = match.get('med_student_essay', '')
            if isinstance(essay, pd.Series):
                essay = essay.iloc[0] if not essay.empty else ''
            
            if pd.isna(essay):
                essay = ""
                
            self.motivation_tree.insert('', 'end', values=(
                med_id,
                round(float(score), 3),
                essay
            ))

    def export_results(self):
        if not self.results:
            messagebox.showerror("Error", "No results to export.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file:
            # Create separate DataFrames for each match type
            global_df = pd.DataFrame(self.results.get('global_matches', []))
            year_df = pd.DataFrame(self.results['year_matches'])
            gap_df = pd.DataFrame(self.results['gap_year_matches'])
            undergrad_df = pd.DataFrame(self.results['undergrad_matches'])
            clinical_df = pd.DataFrame(self.results['clinical_matches'])
            org_df = pd.DataFrame(self.results.get('student_org_matches', []))
            research_df = pd.DataFrame(self.results.get('research_matches', []))
            motivation_df = pd.DataFrame(self.results.get('motivation_matches', []))
            # Add type column to distinguish between match types
            global_df['match_type'] = 'global_index'
            year_df['match_type'] = 'year'
            gap_df['match_type'] = 'gap_year'
            undergrad_df['match_type'] = 'undergrad_degree'
            clinical_df['match_type'] = 'clinical_interests'
            org_df['match_type'] = 'student_orgs'
            research_df['match_type'] = 'research_interests'
            motivation_df['match_type'] = 'motivation_essay'
            # Combine and export
            combined_df = pd.concat([
                global_df, year_df, gap_df, undergrad_df, clinical_df, org_df, research_df, motivation_df
            ], ignore_index=True)
            combined_df.to_csv(file, index=False)
            messagebox.showinfo("Exported", f"Results exported to {os.path.basename(file)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MatchingApp(root)
    root.mainloop() 
