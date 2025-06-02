from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# ——— Setup ———
ES = Elasticsearch(
    "https://…your-cluster…:443",
    api_key="YOUR_API_KEY"
)
INDEX = "version-1"
MODEL = SentenceTransformer("TechWolf/JobBERT-v2")

def make_embeddings(record):
    """
    Modified to work with the new JSON format:
    {
        "user_id": "some_id",
        "acad_string": "Computer Science Masters Machine Learning",
        "prof_string": "Python JavaScript React Node.js Senior Developer Tech",
        "topic_string": "Career guidance in software development"
    }
    """
    # Extract the pre-built strings from the new format
    academic_text = record['acad_string']
    professional_text = record['prof_string'] 
    topic_text = record['topic_string']

    # Generate embeddings for each text
    academic_vec = MODEL.encode(academic_text, convert_to_numpy=True)
    professional_vec = MODEL.encode(professional_text, convert_to_numpy=True)
    topic_vec = MODEL.encode(topic_text, convert_to_numpy=True)

    return academic_vec, professional_vec, topic_vec

# Build a script_score query for one vector field
def vector_query(field, vec, k):
    return {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.vec, '{field}') + 1.0",
                    "params": {"vec": vec}
                }
            }
        }
    }

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match():
    """
    API endpoint that accepts JSON in format:
    {
        "user_id": "123",
        "acad_string": "Computer Science Masters AI",
        "prof_string": "Python TensorFlow Senior ML Engineer",
        "topic_string": "Machine Learning career advice",
        "top_k": 5  // optional, defaults to 5
    }
    """
    data = request.json
    
    # Validate required fields
    required_fields = ['user_id', 'acad_string', 'prof_string', 'topic_string']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    top_k = data.get("top_k", 5)
    user_id = data.get("user_id")  # Now we can access the user_id if needed

    try:
        # 1. Generate the three embeddings from the string inputs
        acad_vec, prof_vec, topic_vec = make_embeddings(data)
        
        # Convert numpy arrays to Python lists for JSON serialization
        acad_vec = acad_vec.tolist()
        prof_vec = prof_vec.tolist() 
        topic_vec = topic_vec.tolist()

        # 2. Run three separate Elasticsearch queries
        res_acad = ES.search(index=INDEX, body=vector_query("academic_vec", acad_vec, top_k))
        res_prof = ES.search(index=INDEX, body=vector_query("professional_vec", prof_vec, top_k))
        res_topic = ES.search(index=INDEX, body=vector_query("mentorship_topic_vec", topic_vec, top_k))

        # 3. Aggregate scores per document ID with weighted combination
        scores = {}
        
        def collect(search_results, weight):
            """Helper function to collect and weight scores from search results"""
            for hit in search_results["hits"]["hits"]:
                doc_id = hit["_id"]
                score = hit["_score"] * weight
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Apply different weights to each type of match
        # You can adjust these weights based on importance
        collect(res_acad, 0.2)   # Academic background: 20%
        collect(res_prof, 0.3)   # Professional skills: 30% 
        collect(res_topic, 0.5)  # Mentorship topic: 50%

        # 4. Sort by combined score (highest first) and limit to top_k
        ranked_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 5. Fetch full documents for the top matches
        results = []
        for doc_id, combined_score in ranked_matches:
            try:
                doc = ES.get(index=INDEX, id=doc_id)["_source"]
                doc["match_score"] = combined_score
                doc["matched_mentor_id"] = doc_id
                results.append(doc)
            except Exception as e:
                print(f"Error fetching document {doc_id}: {e}")
                continue

        return jsonify({
            "user_id": user_id,
            "matches": results,
            "total_matches": len(results)
        })

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)