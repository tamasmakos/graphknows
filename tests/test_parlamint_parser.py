import sys
import os
from datetime import date

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parlamint.parser import parse_parlamint_session

def verify_parser():
    print("Verifying ParlaMint Parser...")
    
    input_dir = "/workspaces/kg/input/ParlaMint-HU-en.ana/ParlaMint-HU-en.txt/2023"
    filename = "ParlaMint-HU-en_2023-02-20.txt"
    txt_path = os.path.join(input_dir, filename)
    meta_path = os.path.join(input_dir, "ParlaMint-HU-en_2023-02-20-meta.tsv")
    ana_meta_path = os.path.join(input_dir, "ParlaMint-HU-en_2023-02-20-ana-meta.tsv")
    
    if not os.path.exists(txt_path):
        print(f"Error: File not found {txt_path}")
        return

    print(f"Parsing {filename}...")
    speeches = parse_parlamint_session(txt_path, meta_path, ana_meta_path)
    
    print(f"Parsed {len(speeches)} speeches.")
    
    if not speeches:
        print("No speeches parsed.")
        return
        
    # Check first speech
    s = speeches[0]
    print(f"Speech 0 ID: {s.speech_id}")
    print(f"Speaker: {s.speaker_name} ({s.party})")
    print(f"Sentiment: {s.sentiment}")
    
    # Check a speech with known sentiment
    # From debug output: u2023-02-20-0 has segments with sentiment
    target_id = "u2023-02-20-0"
    found = False
    for s in speeches:
        if s.speech_id == target_id:
            print(f"\nTarget Speech {target_id}:")
            print(f"Sentiment: {s.sentiment}")
            found = True
            break
            
    if not found:
        print(f"Target speech {target_id} not found.")

if __name__ == "__main__":
    verify_parser()
