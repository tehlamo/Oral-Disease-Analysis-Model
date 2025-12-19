import os
import zipfile

# --- CONFIGURATION ---
# Master Schema: Maps class names to integer IDs
KEEP_CLASSES = {
    "caries": 0,
    "calculus": 1,
    "gingivitis": 2,
    "ulcer": 3,
    "tooth discoloration": 4,
    "discoloration": 4,
    "plaque": 5
}
# ---------------------

def get_roboflow_map(zip_path):
    """Extracts class names from the data.yaml file within a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        yaml_file = next((f for f in files if f.endswith('data.yaml')), None)
        if yaml_file:
            content = z.read(yaml_file).decode('utf-8')
            import re
            match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
            if match:
                names = match.group(1).replace("'", "").replace('"', "").split(',')
                return [n.strip().lower() for n in names]
    return []

def align_ids():
    folder = "raw_data"
    multi_zip = os.path.join("downloads", "multi.zip")
    
    # Retrieve source class names
    source_names = get_roboflow_map(multi_zip)
    print(f"Source classes detected: {source_names}")
    
    files_fixed = 0
    lines_removed = 0

    for filename in os.listdir(folder):
        if not filename.endswith(".txt"): continue
        
        # Skip Zenodo files as they are already correctly indexed
        if not filename.startswith("multi_"): continue 

        path = os.path.join(folder, filename)
        new_lines = []
        modified = False
        
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            try:
                old_id = int(parts[0])
                if old_id < len(source_names):
                    original_name = source_names[old_id]
                    found = False
                    # Map source name to target ID
                    for keep_name, keep_id in KEEP_CLASSES.items():
                        if keep_name in original_name:
                            parts[0] = str(keep_id)
                            new_lines.append(" ".join(parts) + "\n")
                            found = True
                            break
                    if not found:
                        lines_removed += 1
                        modified = True
                else:
                    lines_removed += 1
                    modified = True
            except ValueError:
                continue

        if modified or len(new_lines) != len(lines):
            files_fixed += 1
            with open(path, 'w') as f:
                f.writelines(new_lines)

    print(f"Class alignment complete. Files updated: {files_fixed}")
    print(f"Irrelevant labels removed: {lines_removed}")

if __name__ == "__main__":
    align_ids()