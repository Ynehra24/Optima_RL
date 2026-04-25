import re
import json

FILE_PATH = "/Users/yatharthnehva/Downloads/ripe_traceroute_3_5_days.json"   # change path if needed

with open(FILE_PATH, "r", errors="ignore") as f:
    txt = f.read()

# Extract first complete RIPE object using regex
match = re.search(r'\{.*?"stored_timestamp":\d+\}', txt)

if match:
    obj = json.loads(match.group())
    
    print("=" * 70)
    print("RIPE SAMPLE ENTRY")
    print("=" * 70)

    for k, v in obj.items():
        print(f"{k:<20}: {v}")

    print("=" * 70)

else:
    print("No valid entry found.")