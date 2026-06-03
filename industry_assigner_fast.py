import pandas as pd
import os
import time
import json
import argparse
import re
import random
import threading
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration & Master Mapping
# ---------------------------------------------------------------------------
# Speed-oriented defaults. Override from the command line or environment:
#   OPENAI_MODEL=gpt-4o-mini python industry_assigner_fast.py in.csv out.csv
#   python industry_assigner_fast.py in.csv out.csv --workers 10 --no-web-search

client = OpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT", "25")))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "120"))

TARGET_INDUSTRIES = [
    "Materials & Resources", "Manufacturing", "Construction & Engineering",
    "Logistics", "Wholesale & Distribution", "Retail & E-Commerce",
    "Restaurants & Bars", "Hospitality, Recreation & Travel",
    "Health & Personal Care", "Pet Products & Services", "Software",
    "Professional Services", "Field Services", "Real Estate"
]
TARGET_INDUSTRIES_SET = set(TARGET_INDUSTRIES)

NAICS_MAP = {
    "11": "Materials & Resources",
    "21": "Materials & Resources",
    "22": "Materials & Resources",
    "23": "Construction & Engineering",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale & Distribution",
    "44": "Retail & E-Commerce", "45": "Retail & E-Commerce",
    "48": "Logistics", "49": "Logistics",
    "5132": "Software", "5112": "Software",
    "51": "Professional Services",
    "52": "Professional Services",
    "531": "Real Estate",
    "53212": "Logistics",
    "53211": "Hospitality, Recreation & Travel",
    "5321": "Hospitality, Recreation & Travel",
    "5322": "Retail & E-Commerce",
    "5323": "Hospitality, Recreation & Travel",
    "5324": "Field Services",
    "53": "Professional Services",
    "5413": "Construction & Engineering",
    "5414": "Professional Services",
    "5415": "Software",
    "54194": "Pet Products & Services",
    "54": "Professional Services",
    "55": "Professional Services",
    "5617": "Field Services",
    "5616": "Field Services",
    "56191": "Logistics",
    "56192": "Hospitality, Recreation & Travel",
    "561": "Professional Services",
    "562": "Field Services",
    "61162": "Health & Personal Care",
    "61161": "Hospitality, Recreation & Travel",
    "611699": "Professional Services",
    "61": "Professional Services",
    "6215": "Professional Services",
    "621": "Health & Personal Care", "622": "Health & Personal Care", "623": "Health & Personal Care",
    "6244": "Field Services",
    "624": "Professional Services",
    "712130": "Hospitality, Recreation & Travel",
    "71": "Hospitality, Recreation & Travel",
    "721": "Hospitality, Recreation & Travel",
    "722": "Restaurants & Bars",
    "811": "Field Services",
    "8121": "Health & Personal Care",
    "81291": "Pet Products & Services",
    "81293": "Hospitality, Recreation & Travel",
    "8123": "Field Services",
    "812": "Health & Personal Care",
    "813312": "Field Services",
    "813": "Professional Services",
    "814": "Professional Services",
    "92": "Professional Services",
    "115210": "Materials & Resources",
    "336999": "Retail & E-Commerce",
    "3361": "Retail & E-Commerce",
    "3362": "Retail & E-Commerce",
    "3363": "Wholesale & Distribution",
    "333111": "Retail & E-Commerce",
    "333112": "Retail & E-Commerce",
    "311811": "Retail & E-Commerce",
    "312120": "Restaurants & Bars"
}

US_STATES = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"}

# Small in-process caches help when the same company, website, or row appears more than once.
_llm_cache = {}
_search_cache = {}
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _clean_website(val) -> str:
    raw = _clean(val)
    if not raw:
        return ""
    stripped = re.sub(r'^https?://', '', raw, flags=re.IGNORECASE).strip().rstrip('/')
    if re.fullmatch(r'[\d\s().+\-]{7,20}', stripped):
        return ""
    if re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', stripped):
        return ""
    if '.' not in stripped:
        return ""
    return raw


def validate_state(state: str) -> str:
    if not state:
        return ""
    s = state.strip().upper()
    if s in US_STATES:
        return s
    if s in ("NON-US", "NON US", "INTERNATIONAL"):
        return "Non-US"
    return ""


def normalize_naics(naics: str) -> str:
    """Return NAICS as digits only, with all labels/descriptors removed."""
    if not naics:
        return ""
    return re.sub(r'\D', '', str(naics))


def get_industry_from_naics(naics: str) -> str:
    n = normalize_naics(naics)
    if not n:
        return None
    for length in (6, 5, 4, 3, 2):
        prefix = n[:length]
        if prefix in NAICS_MAP:
            return NAICS_MAP[prefix]
    return None

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict:
    with _cache_lock:
        cached = _llm_cache.get(prompt)
    if cached is not None:
        return cached

    last_error = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Return compact valid JSON only. No explanation."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            data = json.loads(response.choices[0].message.content)
            with _cache_lock:
                _llm_cache[prompt] = data
            return data
        except openai.RateLimitError as e:
            last_error = e
            wait_time = min(45, (2 ** attempt) * 5) + random.uniform(0, 1.5)
            print(f"  [Rate Limit] Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            last_error = e
            wait_time = min(20, (2 ** attempt) * 2) + random.uniform(0, 1)
            print(f"  [Error] {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    print(f"  [LLM Failed] {last_error}")
    return {}


def web_search(company_name: str, website: str) -> str:
    key = (company_name.lower(), website.lower())
    with _cache_lock:
        cached = _search_cache.get(key)
    if cached is not None:
        return cached

    try:
        from ddgs import DDGS
        query = f'"{company_name}" {website} company profile description'
        results = DDGS().text(query, max_results=2)
        snippets = "" if not results else "\n".join([f"{r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception:
        snippets = ""

    with _cache_lock:
        _search_cache[key] = snippets
    return snippets


def enrich_company(row, allow_web_search=True):
    name = _clean(row.get("Company Name"))
    website = _clean_website(row.get("Website"))
    naics = _clean(row.get("Primary NAICS Code"))
    mailer = _clean(row.get("Mailer Industry"))
    seo = _clean(row.get("SEO"))
    existing_state = _clean(row.get("State"))

    industry = get_industry_from_naics(naics)
    industry_notes = "Assigned via NAICS mapping table." if industry else ""
    inferred_naics = ""
    identified_state = ""

    state_needed = not validate_state(existing_state)

    # Fast path: no API call if both industry and state are already solved.
    if industry and not state_needed:
        return industry, industry_notes, identified_state, inferred_naics

    prompt = f"""Classify this company. Return JSON only with keys industry, naics_inference, state.
Use exactly one of these industries: {', '.join(TARGET_INDUSTRIES)}. If unknown, use Unknown.
Return state as two-letter US abbreviation, Non-US, or empty string.
Name: {name}
Website: {website}
Mailer Industry: {mailer}
SEO: {seo}
Existing State: {existing_state}"""

    res = call_llm(prompt)

    if not industry:
        inf_n = normalize_naics(res.get("naics_inference"))
        industry = get_industry_from_naics(inf_n) or _clean(res.get("industry"))
        inferred_naics = inf_n if not naics else ""
        industry_notes = "Inferred from row data."

    if state_needed:
        identified_state = validate_state(_clean(res.get("state")))

    needs_web_search = ((not industry or industry not in TARGET_INDUSTRIES_SET) or (state_needed and not identified_state))
    if allow_web_search and needs_web_search:
        snippets = web_search(name, website)
        if snippets:
            prompt_ws = f"""Use these search snippets to classify the company. Return JSON only with keys industry, naics, state.
Use exactly one target industry: {', '.join(TARGET_INDUSTRIES)}. If unknown, use Unknown.
Return state as two-letter US abbreviation, Non-US, or empty string.
Company: {name}
Website: {website}
Snippets: {snippets[:1200]}"""
            res_ws = call_llm(prompt_ws)
            if not industry or industry == "Unknown" or industry not in TARGET_INDUSTRIES_SET:
                industry = _clean(res_ws.get("industry"))
                inferred_naics = normalize_naics(res_ws.get("naics")) if not naics else ""
                industry_notes = "Verified via web search."
            if state_needed and not identified_state:
                identified_state = validate_state(_clean(res_ws.get("state")))

    if industry not in TARGET_INDUSTRIES_SET:
        industry = "Unknown"
    return industry, industry_notes, identified_state, inferred_naics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(input_file, output_file, workers=8, resume=False, allow_web_search=True):
    print(f"\nProcessing with OpenAI model={MODEL}, workers={workers}, web_search={allow_web_search}...")
    df = pd.read_csv(input_file, dtype=str, encoding="latin-1").fillna("")
    for col in ["Industry", "Industry_Notes", "State_Identified", "NAICS_Inferred"]:
        if col not in df.columns:
            df[col] = ""

    to_process = df[df["Industry"].fillna("").eq("")].index if resume else df.index
    total = len(to_process)
    if total == 0:
        df.to_csv(output_file, index=False)
        print(f"\nNothing to process. Saved to {output_file}")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_company, df.loc[i], allow_web_search): i for i in to_process}
        completed = 0
        for f in as_completed(futures):
            i = futures[f]
            try:
                ind, note, st, inf_n = f.result()
                df.at[i, "Industry"] = ind
                df.at[i, "Industry_Notes"] = note
                df.at[i, "State_Identified"] = st
                df.at[i, "NAICS_Inferred"] = inf_n
            except Exception as e:
                print(f"  [Fatal Error on Row {i}] {e}")

            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  Progress: {completed}/{total} ({(completed / total) * 100:.1f}%)")
            if completed % 250 == 0:
                df.to_csv(output_file, index=False)

    df.to_csv(output_file, index=False)
    print(f"\nDone! Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV")
    parser.add_argument("output", help="Output CSV")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "8")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-web-search", action="store_true", help="Skip DuckDuckGo fallback searches for much faster, lower-cost runs.")
    args = parser.parse_args()
    process_csv(args.input, args.output, args.workers, args.resume, allow_web_search=not args.no_web_search)