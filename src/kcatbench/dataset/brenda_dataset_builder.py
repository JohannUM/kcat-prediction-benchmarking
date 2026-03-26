import logging
from pathlib import Path
from collections import defaultdict
from typing import Generator, Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)

def parse_brenda_flatfile(filepath: str | Path) -> Generator[Dict[str, List[str]], None, None]:
    """
    A memory-efficient generator that parses the BRENDA flat text file.
    
    Yields:
        A dictionary representing a single EC number record. Keys are the 2/3 letter 
        acronyms (e.g., 'ID', 'TN', 'SP') and values are lists of string entries.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Could not find the BRENDA file at: {filepath}")

    # We use defaultdict(list) because a single record can have multiple 
    # entries for the same key (e.g., dozens of 'TN' turnover numbers)
    current_record = defaultdict(list)
    current_key: Optional[str] = None
    current_buffer: List[str] = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # Remove trailing newlines but preserve leading whitespace
                line_stripped = line.rstrip('\n')

                # 1. Check for end of EC record
                if line_stripped == '///':
                    # Save the very last buffered entry before yielding
                    if current_key and current_buffer:
                        current_record[current_key].append(" ".join(current_buffer))
                    
                    # Yield the completed record (if not empty) and reset
                    if current_record:
                        yield dict(current_record)
                    
                    current_record = defaultdict(list)
                    current_key = None
                    current_buffer = []
                    continue

                # Ignore entirely empty lines just in case
                if not line_stripped:
                    continue

                # 2. Check for continuation lines
                # The README states: "Empty spaces at the beginning of a line indicate a continuation line."
                if line.startswith(' ') or line.startswith('\t'):
                    if current_key:
                        # Append the stripped text to our current string buffer
                        current_buffer.append(line_stripped.strip())
                    else:
                        logger.debug(f"Line {line_num}: Orphaned continuation line found. Skipping.")
                
                # 3. Handle new data keys
                else:
                    # If we were tracking a previous key, save its buffer to the record
                    if current_key and current_buffer:
                        current_record[current_key].append(" ".join(current_buffer))
                        current_buffer = []

                    # The README states contents start after a TAB
                    parts = line_stripped.split('\t', 1)
                    
                    if len(parts) == 2:
                        current_key = parts[0]
                        current_buffer = [parts[1].strip()]
                    else:
                        # Fallback for malformed lines
                        logger.debug(f"Line {line_num}: Malformed key line (no TAB). Skipping.")
                        current_key = None
                        current_buffer = []
                        
    except Exception as e:
        logger.error(f"Failed parsing BRENDA file at line {line_num}: {str(e)}")
        raise e



def extract_benchmark_data_from_record(ec_record: dict[str, list[str]]) -> list[dict[str, Any]]:
    """
    Takes a raw EC record dictionary and extracts a clean list of kcat data points.
    
    Cross-references Protein IDs (#1#) to link turnover numbers with their 
    specific organisms, UniProt IDs, and full reaction equations. Automatically 
    filters out engineered mutants.
    """
    ec_number = ec_record.get('ID', [''])[0].strip()
    if not ec_number:
        return []

    # -------------------------------------------------------------------------
    # STEP 1: Parse Proteins (PR) to build a mapping dictionary
    # Example line: "#1# Saccharomyces cerevisiae P00924 (mutant W34A) <1>"
    # -------------------------------------------------------------------------
    protein_map = {}
    mutant_keywords = {'mutant', 'mutated', 'variant', 'engineered'}
    
    for pr_line in ec_record.get('PR', []):
        # Extract the protein ID
        pr_match = re.search(r'^#(\d+)#\s+(.+)', pr_line)
        if not pr_match:
            continue
            
        prot_id = pr_match.group(1)
        remainder = pr_match.group(2)
        
        # Check for mutant status in the PR line
        is_mutant = any(keyword in remainder.lower() for keyword in mutant_keywords)
        
        # Extract UniProt ID (Usually 6 to 10 alphanumeric uppercase characters)
        # We look for a standalone token that fits the standard UniProt pattern
        uniprot_match = re.search(r'\b([O,P,Q][0-9][A-Z0-9]{3}[0-9]|[A-N,R-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})\b', remainder)
        uniprot_id = uniprot_match.group(1) if uniprot_match else None
        
        # Clean up organism name (strip out refs, comments, and the uniprot ID)
        organism = re.sub(r'<[^>]+>', '', remainder)  # Remove refs <1>
        organism = re.sub(r'\([^)]+\)', '', organism)  # Remove comments (mutant)
        if uniprot_id:
            organism = organism.replace(uniprot_id, '')
        organism = organism.strip()
        
        protein_map[prot_id] = {
            'organism': organism,
            'uniprot': uniprot_id,
            'is_mutant': is_mutant
        }

    # -------------------------------------------------------------------------
    # STEP 2: Parse Substrates & Products (SP) to get full equations
    # Example line: "ATP + D-glucose = ADP + D-glucose 6-phosphate (#1#) <1>"
    # -------------------------------------------------------------------------
    reaction_map = {} # Maps prot_id -> list of reactions
    
    for sp_line in ec_record.get('SP', []):
        # Find which proteins this equation belongs to
        prot_match = re.search(r'\(#([\d,]+)#\)', sp_line)
        if not prot_match:
            continue
            
        prot_ids = prot_match.group(1).split(',')
        
        # Isolate the chemical equation
        eq_text = re.sub(r'\(#[\d,]+#\)', '', sp_line) # Remove protein tags
        eq_text = re.sub(r'<[^>]+>', '', eq_text)      # Remove reference tags
        eq_text = eq_text.strip()
        
        # Split into substrates and products (handles =, <=>, ->)
        sides = re.split(r'\s+(?:=|<=>|->|<-)\s+', eq_text)
        if len(sides) == 2:
            substrates = [s.strip() for s in sides[0].split(' + ')]
            products = [p.strip() for p in sides[1].split(' + ')]
            
            for pid in prot_ids:
                if pid not in reaction_map:
                    reaction_map[pid] = []
                reaction_map[pid].append({'substrates': substrates, 'products': products})

    # -------------------------------------------------------------------------
    # STEP 3: Parse Turnover Numbers (TN) and stitch it all together
    # Example line: "2.5 {D-glucose} (#1#) (pH 7.5, 25 °C) <1,2>"
    # -------------------------------------------------------------------------
    extracted_data = []
    
    for tn_line in ec_record.get('TN', []):
        # 1. Extract kcat value
        val_match = re.search(r'^([0-9\.]+)', tn_line)
        if not val_match:
            continue
        kcat_val = float(val_match.group(1))
        if kcat_val <= 0:
            continue
            
        # 2. Extract specific substrate
        sub_match = re.search(r'\{([^}]+)\}', tn_line)
        kcat_substrate = sub_match.group(1).strip() if sub_match else None
        
        # 3. Extract proteins
        prot_match = re.search(r'\(#([\d,]+)#\)', tn_line)
        tn_prot_ids = prot_match.group(1).split(',') if prot_match else []
        
        # 4. Extract comments and references
        # Match parentheses that do NOT contain a protein hash (e.g. not (#1#))
        comment_match = re.search(r'\((?!#[\d,]+#)([^)]+)\)', tn_line)
        comment = comment_match.group(1) if comment_match else ""
        
        ref_match = re.search(r'<([^>]+)>', tn_line)
        references = ref_match.group(1) if ref_match else ""

        # Filter out mutants identified in the TN comment
        if any(keyword in comment.lower() for keyword in mutant_keywords):
            continue

        # Extract Temperature and pH from comment
        ph_match = re.search(r'pH\s*([0-9\.]+)', comment, re.IGNORECASE)
        pH_val = float(ph_match.group(1)) if ph_match else None
        
        temp_match = re.search(r'([0-9\.]+)\s*°C', comment, re.IGNORECASE)
        temp_val = float(temp_match.group(1)) if temp_match else None

        # Build the final record for each protein associated with this kcat
        for pid in tn_prot_ids:
            protein_info = protein_map.get(pid)
            
            # Skip if we couldn't find the protein or if it's a known mutant
            if not protein_info or protein_info['is_mutant']:
                continue
                
            # Attempt to find the full reaction in the SP map
            # We look for a reaction where our kcat_substrate is in the substrates list
            full_substrates = [kcat_substrate] if kcat_substrate else []
            full_products = []
            
            if pid in reaction_map and kcat_substrate:
                for rxn in reaction_map[pid]:
                    # Lowercase matching to bypass minor capitalization differences
                    if any(kcat_substrate.lower() == s.lower() for s in rxn['substrates']):
                        full_substrates = rxn['substrates']
                        full_products = rxn['products']
                        break # Take the first matching reaction equation
                        
            # Assemble the exact datapoint
            data_point = {
                'EC_Number': ec_number,
                'Organism': protein_info['organism'],
                'UniProt_ID': protein_info['uniprot'],
                'kcat_substrate': kcat_substrate,
                'all_substrates': full_substrates,
                'all_products': full_products,
                'kcat_value': kcat_val,
                'temperature': temp_val,
                'pH': pH_val,
                'references': references, # Vital for SABIO-RK deduplication!
            }
            extracted_data.append(data_point)
            
    return extracted_data