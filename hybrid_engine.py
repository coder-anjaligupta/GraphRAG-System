import ollama
import time
import numpy as np
from neo4j import GraphDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Configuration
URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "password123")

client = ollama.Client(host='http://127.0.0.1:11434', timeout=None)

# Initialize HuggingFace Embeddings Model for Raw Text Vector Search
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Text Corpus Chunks
corpus_chunks = [
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Microsoft acquired GitHub in 2018 for 7.5 billion dollars.",
    "GitHub is headquartered in San Francisco and provides Git repository hosting."
]

# Pre-compute Vector Embeddings for Text Chunks
corpus_embeddings = embeddings_model.embed_documents(corpus_chunks)

def text_retrieval(query, top_k=3):
    # Generate vector embedding for user query
    query_vector = embeddings_model.embed_query(query)
    
    # Calculate Cosine Similarity for Vector Search
    similarities = []
    for chunk, emb in zip(corpus_chunks, corpus_embeddings):
        similarity = np.dot(query_vector, emb) / (np.linalg.norm(query_vector) * np.linalg.norm(emb))
        similarities.append((similarity, chunk))
        
    # Sort chunks by highest vector similarity score
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in similarities[:top_k]]

def safe_graph_traversal(entity_name):
    query = """
    MATCH (a:Entity {name: $entity})-[r1]->(b:Entity)
    OPTIONAL MATCH (b)-[r2]->(c:Entity)
    RETURN a.name AS src1, type(r1) AS rel1, b.name AS tgt1,
           type(r2) AS rel2, c.name AS tgt2
    """
    raw_relations = set()
    
    # Retry mechanism for Neo4j connection
    for attempt in range(1, 4):
        try:
            driver = GraphDatabase.driver(URI, auth=AUTH)
            with driver.session() as session:
                result = session.run(query, entity=entity_name)
                for record in result:
                    if record['src1'] and record['rel1'] and record['tgt1']:
                        raw_relations.add(f"{record['src1'].strip()} {record['rel1'].strip()} {record['tgt1'].strip()}")
                    if record['rel2'] and record['tgt2']:
                        raw_relations.add(f"{record['tgt1'].strip()} {record['rel2'].strip()} {record['tgt2'].strip()}")
            driver.close()
            break
        except Exception as e:
            time.sleep(1)
            if attempt == 3:
                print(f"[Warning] Graph fallback triggered: {e}")
                
    return sorted(list(raw_relations))

def hybrid_query_engine(user_query, key_entity):
    print("Performing Text Vector Retrieval...")
    text_context = text_retrieval(user_query)
    
    print("Performing Graph Traversal...")
    graph_context = safe_graph_traversal(key_entity)
    
    combined_context = f"""
    Text Context (Unstructured):
    {' '.join(text_context)}
    
    Graph Knowledge (Structured):
    {', '.join(graph_context) if graph_context else 'No graph triples retrieved.'}
    """
    
    prompt = f"""
    You are an expert Knowledge Graph Assistant. Answer the question precisely using only the provided Context.
    
    Context:
    {combined_context}
    
    Question: {user_query}
    Answer:
    """
    
    print("Generating LLM Final Answer...")
    response = client.generate(
        model='llama3.2',
        prompt=prompt,
        options={"num_predict": 200, "temperature": 0.1}
    )
    return response['response'], combined_context

if __name__ == "__main__":
    query = "Where is the company headquartered that was acquired by Microsoft?"
    entity = "Microsoft"
    
    print("Running Hybrid Retrieval Engine...")
    final_answer, retrieved_context = hybrid_query_engine(query, entity)
    
    print("\n--- Retrieved Context ---")
    print(retrieved_context)
    
    print("\n--- Final Answer ---")
    print(final_answer)