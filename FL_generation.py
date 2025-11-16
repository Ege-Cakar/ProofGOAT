import os
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv

# --------------------------------------------
# Load API key
# --------------------------------------------
load_dotenv(".env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")

configure(api_key=API_KEY)

LLM_MODEL = "gemini-2.5-flash"

# --------------------------------------------
#  Linking to /lean_env
# --------------------------------------------
import os

# Directory of this Python file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# lean_env is guaranteed to be in the same directory
ROOT = os.path.join(FILE_DIR, "lean_env")

if not os.path.isdir(ROOT):
    raise RuntimeError(f"lean_env/ not found at: {ROOT}")

print(f"Using Lean project root: {ROOT}")

# fl_outputs folder inside lean_env
OUT_DIR = os.path.join(ROOT, "fl_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Lake must point explicitly to the Lean project
LEAN_CMD = ["lake", "-d", ROOT, "env", "lean"]

COMMON_LEAN = open(os.path.join(ROOT, "common.lean")).read()

PATH_SYNTAX = os.path.join(ROOT, "mathlib_syntax.lean")
with open(PATH_SYNTAX) as f:
    mathlib_syntax = f.read()

# --------------------------------------------
# Data structure
# --------------------------------------------
@dataclass
class NLExample:
    id: str
    nl_statement: str
    nl_proof: str
    formal_signature: str
    src_header: str               


# --------------------------------------------
# Core generation function
# --------------------------------------------
def generate_full_proof(example: NLExample, max_attempts: int = 4):

    os.makedirs(OUT_DIR, exist_ok=True)
    lean_path = os.path.join(OUT_DIR, f"{example.id}.lean")
    meta_path = os.path.join(OUT_DIR, f"{example.id}_meta.json")

    model = GenerativeModel(LLM_MODEL)

    base_prompt = f"""
        You are a Lean 4 theorem-proving engine.

    Your ONLY task is to produce Lean 4 code that **compiles** under the following header:

    {example.src_header}

    This header imports `.common`, which contains a large portion of mathlib.  
    You MUST assume ONLY mathlib definitions, lemmas, namespaces, and syntax available after this header loads.  
    NEVER invent names, NEVER guess lemma names, NEVER create namespaces, NEVER define auxiliary lemmas.

    ============================================================
    ABSOLUTE OUTPUT RULES
    ============================================================

    1. Output **Lean code only**.  
    • No English.  
    • No comments.  
    • No Markdown.  
    • No explanations.  

    2. The output MUST begin exactly with:

    {example.src_header}

    3. Then output EXACTLY this theorem declaration:

    theorem {example.id}
    {example.formal_signature} := by

    4. After `by`, you may ONLY use the following constructs:
    • intro …  
    • have …  
    • apply …  
    • exact …  
    • simp  
    • simp [lemmas]  
    • rw [lemmas]  
    • calc …  
    • classical  
    • fun x => …  

    5. You MUST restrict your syntax to real mathlib idioms.  
    The ONLY legal API patterns for rewriting, inequalities, sup/inf, sets, rationality, and geometry  
    are exactly those shown in the examples below.

    ============================================================
    VALID MATHLIB SYNTAX EXAMPLES (DO NOT OUTPUT THESE)
    ============================================================

    {mathlib_syntax}

    These patterns show:
    • how simp/rw/calc are written  
    • how IsRat, irrational, add/sub, sup/inf, inequalities, closure, continuity, inner product, and spans are used  
    • the real mathlib namespace shapes  
    • the grammar that Lean accepts  

    You MUST imitate these patterns and nothing else.

    ============================================================
    THEOREM INPUT
    ============================================================

    Natural-language statement:
    """
    {example.nl_statement}
    """

    Natural-language proof sketch:
    """
    {example.nl_proof}
    """

    Produce the Lean 4 proof using ONLY:
    - the syntax allowed above  
    - the imports specified in {example.src_header}  
    - real mathlib lemmas (nothing invented)

    OUTPUT ONLY THE FINAL LEAN CODE.

    """

    prompt = base_prompt

    for attempt in range(1, max_attempts + 1):
        print(f"\n=== Attempt {attempt}/{max_attempts} ===")

        response = model.generate_content(prompt)
        lean_code = response.text.strip()

        with open(lean_path, "w") as f:
            f.write(lean_code)

        print(f"Saved: {lean_path}")

        proc = subprocess.run(
            LEAN_CMD + [lean_path],
            cwd=ROOT,                            
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if proc.returncode == 0:
            print("Lean accepted the proof.")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "example": asdict(example),
                        "success": True,
                        "attempts": attempt,
                        "lean_file_path": lean_path
                    },
                    f, indent=2
                )
            return True, lean_code

        print("\nLean error:")
        print(proc.stderr)

        prompt = (
            base_prompt
            + f"\nThe last attempt failed with this Lean error:\n{proc.stderr}\n"
            + "Fix all issues and output corrected Lean code."
        )

    print("\nAll attempts failed.")
    return False, None


# --------------------------------------------
# Run toy example
# --------------------------------------------
if __name__ == "__main__":
    toy = NLExample(
        id="Rudin_ex_1_1a",
        nl_statement="If r is rational (r ≠ 0) and x is irrational, prove r + x is irrational.",
        nl_proof="If r and r + x were both rational, then x = (r + x) − r would also be rational.",
        formal_signature="(x : ℝ) (y : ℚ) : irrational x → irrational (x + y)",
        src_header="""
        import .common

        open real complex
        open topological_space
        open filter
        open_locale real
        open_locale topology
        open_locale big_operators
        open_locale complex_conjugate
        open_locale filter

        noncomputable theory
        """
    )

    success, code = generate_full_proof(toy)
    if success:
        print("\nFinal Lean code:\n")
        print(code)
