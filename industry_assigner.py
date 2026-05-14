"""
industry_assigner.py
--------------------
Enriches a company CSV with Industry and State.

LOGIC: Mapping-First Approach
1. Checks NAICS/SICS codes against a Master Mapping Table (Instant & Accurate).
2. If no code, LLM (OpenAI gpt-4o-mini) extracts NAICS from Mailer Industry/SEO.
3. If still unknown, Web Search fallback.

Usage:
  python3 industry_assigner.py input.csv output.csv [--workers N] [--resume]
"""

import pandas as pd
import os
import time
import json
import argparse
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration & Master Mapping
# ---------------------------------------------------------------------------

# Initialize OpenAI client with API key
client = OpenAI(api_key="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")
MODEL = "gpt-4o-mini" # Extremely fast and cheap

TARGET_INDUSTRIES = [
    "Materials & Resources", "Manufacturing", "Construction & Engineering",
    "Logistics", "Wholesale & Distribution", "Retail & E-Commerce",
    "Restaurants & Bars", "Hospitality Recreation & Travel",
    "Health & Personal Care", "Pet Products & Services", "Software",
    "Professional Services", "Field Services", "Real Estate"
]


def map_naics_to_industry(naics_code):
    """
    Determine the internal industry category for a given NAICS code string or number.
    Uses longest-prefix matching against the NAICS_TO_INDUSTRY mapping.
    Returns the industry name if matched, otherwise a default "Professional Services".
    """
    code_str = str(naics_code).strip()
    best_match = None
    for prefix in NAICS_TO_INDUSTRY:
        if code_str.startswith(prefix):
            # Update best_match if this prefix is longer (more specific) than current best
            if best_match is None or len(prefix) > len(best_match):
                best_match = prefix
    if best_match:
        return NAICS_TO_INDUSTRY[best_match]
    else:
        # If no mapping found (unlikely for valid NAICS), default to Professional Services
        return "Professional Services"

# NAICS-to-Industry mapping with explicit overrides and prefixes.
# The keys can be full 6-digit NAICS codes or leading prefixes (2-5 digits).
# Longer (more specific) prefixes take precedence over shorter ones.
NAICS_TO_INDUSTRY = {
    # --- Specific NAICS (4-6 digit) overrides to correct common misclassifications ---
    "115210": "Materials & Resources",        # Support Activities for Animal Production (very specific agriculture support)
    "2382": "Field Services",              # Building Equipment Contractors (Electrical, Plumbing, HVAC)
    "23821": "Field Services",             # Electrical Contractors & Wiring Installation
    "23822": "Field Services",             # Plumbing, Heating, HVAC Contractors
    "311811": "Retail & E-Commerce",         # Retail Bakeries (e.g., local bakery shops) map to retail, not manufacturing
    "312120": "Restaurants & Bars",          # Breweries with taprooms (Microbreweries/Brewpubs) act like bars/restaurants
    "333111": "Retail & E-Commerce",         # Farm Machinery Dealers often misclassified under manufacturing
    "333112": "Retail & E-Commerce",         # Lawn & Garden Equipment Dealers misclassified under manufacturing
    "336999": "Retail & E-Commerce",         # Misc. Vehicle Dealers (golf carts, ATVs) misclassified under manufacturing
    "3361": "Retail & E-Commerce",           # Automobile/Light Vehicle Dealers often use 3361** (motor vehicle mfg) by mistake
    "3362": "Retail & E-Commerce",           # Trailer & RV Dealers misusing manufacturing codes
    "3363": "Wholesale & Distribution",      # Auto Parts distributors misusing manufacturing codes
    "45391": "Pet Products & Services",      # Pet and Pet Supplies Stores (retailers) should be in Pet Products & Services
    "484210": "Field Services",              # Residential Moving Companies (local freight, but essentially on-site service)
    "53211": "Hospitality Recreation & Travel",  # Passenger Car Rental & Leasing (e.g., rental car agencies for travelers)
    "53212": "Retail & E-Commerce",          # Consumer Truck, Trailer, RV Rental (often akin to retail tool rental)
    "5323": "Hospitality Recreation & Travel",   # General Rental Centers (party/event rental, recreation-related)
    "5324": "Field Services",                # Heavy Machinery Rental & Leasing (construction/industrial on-site services)
    "5413": "Construction & Engineering",    # Architectural & Engineering Services (fits Construction/Engineering industry)
    "5415": "Software",                      # IT Consulting/Systems Design Services (tech-heavy, align with Software/Tech)
    "54194": "Pet Products & Services",      # Veterinary Services (pet-focused health services, not generic Professional)
    "5615": "Hospitality Recreation & Travel",   # Travel Agencies & Tour Operators (tourism industry services)
    "5616": "Field Services",                # Investigation & Security Services (guards, alarm install – field operations)
    "5617": "Field Services",                # Services to Buildings & Dwellings (landscaping, cleaning – field work)
    "611699": "Professional Services",       # All Other Misc. Schools (no specific category; default to Professional Services)
    "712130": "Hospitality Recreation & Travel", # Zoos & Botanical Gardens (recreation/entertainment venues)
    "812210": "Professional Services",     # Funeral Homes & Funeral Services
    "81291": "Pet Products & Services",      # Pet Care (except veterinary) Services (grooming, boarding)
    "81293": "Field Services",  # Parking Lots & Garages (travel/transport support services)
    "813312": "Field Services",              # Environment, Conservation & Wildlife Orgs (field-oriented operations)
    # --- Broad NAICS prefix mappings (2-3 digit sectors) for general cases ---
    "11": "Materials & Resources",           # Agriculture, Forestry, Fishing and Hunting
    "21": "Materials & Resources",           # Mining, Quarrying, and Oil & Gas Extraction
    "22": "Materials & Resources",           # Utilities (energy, resources)
    "23": "Construction & Engineering",      # Construction
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",  # Manufacturing (broadly, except overridden cases above)
    "42": "Wholesale & Distribution",        # Wholesale Trade
    "44": "Retail & E-Commerce", "45": "Retail & E-Commerce",  # Retail Trade
    "48": "Logistics", "49": "Logistics",    # Transportation and Warehousing (Logistics)
    "51": "Professional Services",           # Information (Software & IT are overridden above; remaining info sector -> Pro Serv)
    "52": "Professional Services",           # Finance and Insurance
    "53": "Professional Services",           # Real Estate & Rental (Real Estate and Rentals are overridden above)
    "531": "Real Estate",                    # Real Estate (separate from generic Professional Services)
    "5321": "Hospitality Recreation & Travel",  # Broad Rental & Leasing in travel domain (overridden by specific 53211/53212 above)
    "54": "Professional Services",           # Professional, Scientific, and Technical Services (some overridden above)
    "5414": "Professional Services",         # Specialized Design Services (professional office-based work)
    "55": "Professional Services",           # Management of Companies and Enterprises
    "56": "Professional Services",           # Admin/Support Services (some overridden for Field or Travel above)
    "561": "Professional Services",          # (Catch-all for 561* not overridden)
    "562": "Field Services",                 # Waste Management & Remediation (field operations)
    "61": "Professional Services",           # Educational Services (no dedicated category, treat as professional by default)
    "621": "Health & Personal Care", "622": "Health & Personal Care", "623": "Health & Personal Care",  # Hospitals & healthcare
    "624": "Professional Services",          # Social Assistance (occupies NAICS 624, default to Professional Services)
    "6244": "Field Services",                # Child Day Care Services (on-site personal/field service)
    "71": "Hospitality Recreation & Travel", # Arts, Entertainment, and Recreation
    "721": "Hospitality Recreation & Travel",# Accommodation (Hotels, etc.)
    "722": "Restaurants & Bars",             # Food Services (Restaurants, Bars)
    "81": "Professional Services",           # Other Services (except Public Admin; see below for specific 812 overrides)
    "812": "Health & Personal Care",         # Personal Care Services (salons, personal care)
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)): return ""
    return str(val).strip()

def _clean_website(val) -> str:
    raw = _clean(val)
    if not raw: return ""
    stripped = re.sub(r'^https?://', '', raw, flags=re.IGNORECASE).strip().rstrip('/')
    if re.fullmatch(r'[\d\s().+\-]{7,20}', stripped): return ""
    if re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', stripped): return ""
    if '.' not in stripped: return ""
    return raw

def validate_state(state: str) -> str:
    if not state: return ""
    s = state.strip().upper()
    us_states = {"AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"}
    if s in us_states: return s
    if s in ("NON-US", "NON US", "INTERNATIONAL"): return "Non-US"
    return ""

def get_industry_from_naics(naics: str) -> str:
    """Checks the NAICS code against the Master Mapping Table."""
    if not naics: return None
    n = re.sub(r'\D', '', naics) # digits only
    # Check longest prefixes first
    for length in [5, 4, 3, 2]:
        prefix = n[:length]
        if prefix in NAICS_TO_INDUSTRY:
            return NAICS_TO_INDUSTRY[prefix]
    return None

def extract_naics_code(value: str) -> str:
    """Extract only the numeric NAICS code from a string."""
    if not value: return ""
    # Find first sequence of 2-6 digits
    match = re.search(r'\b(\d{2,6})\b', value)
    return match.group(1) if match else ""

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": "You are a business data assistant. Respond in JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [LLM Error] {e}. Retrying...")
            time.sleep(2 * (attempt + 1))
    return {}

def web_search(company_name: str, website: str) -> str:
    try:
        from ddgs import DDGS
        # Focus on finding a clear business description or profile
        query = f'"{company_name}" {website} company profile description'
        results = DDGS().text(query, max_results=3)
        
        # If no results, we don't dig deeper. They are likely closed or invisible.
        if not results:
            return ""
            
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])
    except:
        return ""

def enrich_company(row):
    name = _clean(row.get("Company Name"))
    website = _clean_website(row.get("Website"))
    naics = _clean(row.get("Primary NAICS Code"))
    mailer = _clean(row.get("Mailer Industry"))
    seo = _clean(row.get("SEO"))
    existing_state = _clean(row.get("State"))

    industry = get_industry_from_naics(naics)
    inferred_naics = extract_naics_code(naics) if naics else ""  # Use existing NAICS if available
    
    # Check if we need to find a state (only if original is missing/invalid)
    existing_state_valid = validate_state(existing_state)
    state_needed = not existing_state_valid
    identified_state = ""  # Only fill if state was missing and we found it

    if not industry or state_needed:
        prompt = f"""Enrich this company. Return JSON with keys: industry, naics_inference, state.
        IMPORTANT: naics_inference must be the most specific numeric NAICS code possible (full 6-digit preferred, e.g., "541511", but partial like "5415" is acceptable if full is uncertain). NOT a description.
        IMPORTANT: state must be a 2-letter US state code (e.g., "TX", "CA") if you can determine the state.
        Name: {name}, Website: {website}, Mailer: {mailer}, SEO: {seo}, Existing State: {existing_state}
        Target Industries: {', '.join(TARGET_INDUSTRIES)}"""
        
        res = call_llm(prompt)
        
        if not industry:
            inf_n = _clean(res.get("naics_inference"))
            industry = get_industry_from_naics(inf_n) or res.get("industry")
            # If no existing NAICS, use the inferred one
            if not inferred_naics:
                inferred_naics = extract_naics_code(inf_n)
            
        if state_needed:
            # Only set identified_state if original State was empty and we found one
            found_state = validate_state(res.get("state"))
            if found_state and not existing_state:
                identified_state = found_state

    # Web Search Fallback
    if (not industry or industry not in TARGET_INDUSTRIES) or (state_needed and not identified_state):
        snippets = web_search(name, website)
        if snippets:
            prompt_ws = f"""Use these search results to find Industry and State for {name}. 
            IMPORTANT: naics must be the most specific numeric NAICS code possible (full 6-digit preferred, e.g., "541511", but partial like "5415" is acceptable if full is uncertain). NOT a description.
            IMPORTANT: state must be a 2-letter US state code (e.g., "TX", "CA") if you can determine the state.
            JSON keys: industry, naics, state.
            Results: {snippets}"""
            res_ws = call_llm(prompt_ws)
            if not industry or industry == "Unknown":
                industry = res_ws.get("industry")
                ws_naics = _clean(res_ws.get("naics"))
                # If no existing or previous inferred NAICS, use this one
                if not inferred_naics:
                    inferred_naics = extract_naics_code(ws_naics)
            if state_needed and not identified_state:
                # Only set identified_state if we actually found one and original was empty
                found_state = validate_state(res_ws.get("state"))
                if found_state and not existing_state:
                    identified_state = found_state

    if industry not in TARGET_INDUSTRIES: industry = "Unknown"
    
    # Populate State_Identified: Use existing if present, else identified
    final_state = existing_state if existing_state_valid else identified_state
    
    return industry, final_state, inferred_naics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_csv(input_file, output_file, workers=20, resume=False):
    print(f"\nProcessing with OpenAI ({MODEL})...")
    df = pd.read_csv(input_file, dtype=str, encoding="latin-1")
    for col in ["Industry", "State_Identified", "NAICS_Inferred"]:
        if col not in df.columns: df[col] = ""

    to_process = df[df["Industry"] == ""].index if resume else df.index
    total = len(to_process)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_company, df.loc[i]): i for i in to_process}
        completed = 0
        for f in as_completed(futures):
            i = futures[f]
            ind, st, inf_n = f.result()
            df.at[i, "Industry"] = ind
            df.at[i, "State_Identified"] = st
            df.at[i, "NAICS_Inferred"] = inf_n

    df.to_csv(output_file, index=False)
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input CSV")
    parser.add_argument("output", help="Output CSV")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    process_csv(args.input, args.output, args.workers, args.resume)
