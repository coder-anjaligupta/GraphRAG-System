import json
import time
import ollama
from neo4j import GraphDatabase

# Neo4j Database Connection Configuration
URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "password123")  

driver = None
for attempt in range(1, 6):
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Neo4j Database Connected Successfully!")
        break
    except Exception:
        print(f"Waiting for Neo4j readiness... attempt {attempt}/5")
        time.sleep(3)

if not driver:
    print("Failed to connect to Neo4j. Verify container status.")
    exit(1)

# Sample input text (Replace or extend with any text)
sample_text = (
    "Microsoft was founded by Bill Gates and Paul Allen. "
    "Microsoft acquired GitHub in 2018. "
    "GitHub is headquartered in San Francisco."
)

# Fully Dynamic System Prompt (No Hardcoded Entity Examples)
prompt = f"""
You are an AI data ingestion engine that extracts Knowledge Graph triples.
Extract all relationships from the input text below.

STRICT INSTRUCTIONS:
1. Output MUST be a valid JSON object with a single top-level key "relationships".
2. "relationships" must contain a list of objects.
3. Each object MUST have "subject", "predicate", and "object" string keys.
4. Normalize predicates into uppercase words with underscores (e.g., FOUNDED_BY, ACQUIRED).

Text to process:
{sample_text}
"""

print("Extracting Triples using Ollama Llama3.2 (JSON Mode)...")

try:
    client = ollama.Client(host='http://127.0.0.1:11434', timeout=None)

    response = client.generate(
        model='llama3.2',
        prompt=prompt,
        format='json',
        options={
            "num_predict": 256,
            "temperature": 0.1
        
        }
    )

    raw_response = response['response']
    print("\n--- Raw LLM Response ---")
    print(raw_response)

    parsed_json = json.loads(raw_response)

    # Dynamic extraction of relationships list
    triples = []
    if isinstance(parsed_json, dict):
        triples = parsed_json.get("relationships", parsed_json.get("triples", []))
        if not triples and len(parsed_json) > 0:
            first_value = list(parsed_json.values())[0]
            if isinstance(first_value, list):
                triples = first_value
    elif isinstance(parsed_json, list):
        triples = parsed_json

    print("\n--- Extracted Triples Saved To Neo4j ---")

    # DB Persistence
    with driver.session() as session:
        for item in triples:
            if isinstance(item, dict):
                sub = str(item.get('subject', item.get('head', ''))).strip()
                pred = str(item.get('predicate', item.get('relation', ''))).strip().replace(' ', '_').upper()
                obj = str(item.get('object', item.get('tail', ''))).strip()

                if sub and pred and obj:
                    query = (
                        "MERGE (a:Entity {name: $sub}) "
                        "MERGE (b:Entity {name: $obj}) "
                        "MERGE (a)-[r:" + pred + "]->(b)"
                    )
                    session.run(query, sub=sub, obj=obj)
                    print(f"Saved to Neo4j: ({sub}) -[{pred}]-> ({obj})")

    print("\nData Ingestion Completed Successfully!")

except Exception as err:
    print("\n[Error Details]:", err)

finally:
    if driver:
        driver.close()

print("\nIngestion Process Finished!")