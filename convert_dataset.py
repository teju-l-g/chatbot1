import json

# Load your Kaggle-style JSON
with open("data/intents.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

converted = {"intents": []}

for intent_obj in raw_data["intents"]:
    tag = intent_obj.get("intent", "unknown")
    patterns = intent_obj.get("text", [])
    responses = intent_obj.get("responses", [])

    converted["intents"].append({
        "tag": tag,
        "patterns": patterns,
        "responses": responses
    })

# Save to cleaned format
with open("data/intents.json", "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2)

print("✅ Converted dataset to expected format.")
