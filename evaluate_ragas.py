import os
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# 1. Initialize Local LLM for evaluation and HuggingFace Embeddings 
eval_llm = Ollama(model="llama3.2", base_url="http://127.0.0.1:11434", timeout=None)
eval_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Vanilla Vector Search Dataset (Optimized with 1 Question and 3 Metrics)
vanilla_eval_data = {
    "question": [
        "Where is the company headquartered that was acquired by Microsoft?"
    ],
    "contexts": [
        [
            "Microsoft acquired GitHub in 2018.",
            "GitHub is headquartered in San Francisco."
        ]
    ],
    "answer": [
        "The company acquired by Microsoft is GitHub, which is headquartered in San Francisco."
    ],
    "ground_truth": [
        "San Francisco"
    ]
}

# 3. GraphRAG Dataset (Optimized with 1 Question and 3 Metrics)
graphrag_eval_data = {
    "question": [
        "Where is the company headquartered that was acquired by Microsoft?"
    ],
    "contexts": [
        [
            "Microsoft acquired GitHub in 2018.",
            "GitHub is headquartered in San Francisco.",
            "Microsoft ACQUIRED GitHub, GitHub HEADQUARTERED_IN San Francisco"
        ]
    ],
    "answer": [
        "The company acquired by Microsoft is GitHub, which is headquartered in San Francisco."
    ],
    "ground_truth": [
        "San Francisco"
    ]
}

def run_evaluation():
    print("Preparing Datasets for Vanilla RAG vs GraphRAG Evaluation...")
    vanilla_dataset = Dataset.from_dict(vanilla_eval_data)
    graphrag_dataset = Dataset.from_dict(graphrag_eval_data)

    config = RunConfig(timeout=None, max_retries=2, max_workers=1)

    # 4. Run Evaluation for Vanilla RAG with Latency Tracking
    print("\n--- Running Vanilla Vector Search Evaluation ---")
    start_time_vanilla = time.time()
    vanilla_results = evaluate(
        dataset=vanilla_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=config
    )
    vanilla_latency = time.time() - start_time_vanilla
    print(f"Vanilla RAG Latency: {vanilla_latency:.2f} seconds")

    # 5. Run Evaluation for GraphRAG with Latency Tracking
    print("\n--- Running GraphRAG Evaluation ---")
    start_time_graphrag = time.time()
    graphrag_results = evaluate(
        dataset=graphrag_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=config
    )
    graphrag_latency = time.time() - start_time_graphrag
    print(f"GraphRAG Latency: {graphrag_latency:.2f} seconds")
    
    # 6. Combine and Compare Results in a Pandas DataFrame
    print("\n--- Comparison Results DataFrame ---")
    vanilla_df = vanilla_results.to_pandas()
    graphrag_df = graphrag_results.to_pandas()

    vanilla_df['Approach'] = 'Vanilla RAG'
    graphrag_df['Approach'] = 'GraphRAG'

    comparison_df = pd.concat([vanilla_df, graphrag_df], ignore_index=True)

    # Force pandas to display all columns without truncating with...
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df[['Approach', 'question', 'faithfulness', 'answer_relevancy', 'context_precision']].to_string())

    # 7. Print Latency Summary Report
    print("\n--- Latency Performance Report ---")
    print(f"Vanilla Vector Search Execution Time: {vanilla_latency:.2f}s")
    print(f"GraphRAG Execution Time: {graphrag_latency:.2f}s")

if __name__ == "__main__":
    run_evaluation()