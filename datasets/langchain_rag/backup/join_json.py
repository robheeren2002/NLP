import json
import glob

# 1) Find all your files (adjust the pattern/path as needed)
files = glob.glob("docs_web_map_ev_database*with_images.json")   # e.g., put your files in a folder named "data"

all_items = []

# 2) Load each file and extend the big list
for path in files:
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)   # each file is a list of dicts like your example
        all_items.extend(data)

print("Total items:", len(all_items))

# --- Optional: de-duplicate (keeps first occurrence) ---
# Pick a key that is stable in your data, e.g. metadata.loc or metadata.source
seen = set()
deduped = []
for item in all_items:
    key = (item.get("metadata") or {}).get("loc") or (item.get("metadata") or {}).get("source")
    if key and key in seen:
        continue
    if key:
        seen.add(key)
    deduped.append(item)

print("After dedupe:", len(deduped))

# 3) Save the combined output
with open("combined.json", "w", encoding="utf-8") as f:
    json.dump(deduped, f, ensure_ascii=False, indent=2)
