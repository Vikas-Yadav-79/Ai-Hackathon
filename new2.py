from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer



ES_HOST = "https://a7b16fc079fb47698fe7b89b50ff406b.us-east-1.aws.found.io:443"
API_KEY = "OTNhN0lKY0IxaWdKNjRyOXg4UUY6WVZKTW5aX3VuQVRtUEhrVDVNWWxMQQ=="

# ——— Setup ———
ES = Elasticsearch(
    ES_HOST,
    api_key=API_KEY
)



INDEX = "version-1"
MODEL = SentenceTransformer("TechWolf/JobBERT-v2")

def make_embeddings(record):
    """
    Generate embeddings from the new JSON format:
    {
        "user_id": "some_id",
        "acad_string": "Computer Science Masters Machine Learning",
        "prof_string": "Python JavaScript React Node.js Senior Developer Tech",
        "topic_string": "Career guidance in software development"
    }
    """
    academic_text = record['acad_string']
    professional_text = record['prof_string'] 
    topic_text = record['topic_string']

    academic_vec = MODEL.encode(academic_text, convert_to_numpy=True)
    professional_vec = MODEL.encode(professional_text, convert_to_numpy=True)
    topic_vec = MODEL.encode(topic_text, convert_to_numpy=True)

    return academic_vec, professional_vec, topic_vec

def vector_query_by_user_types(field, vec, k, user_types):
    """
    Build a script_score query that searches specific user types
    user_types: list of user types to search (e.g., ["mentor"] or ["mentee", "normal_user"])
    """
    return {
        "size": k,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"user_type": user_types}}
                        ]
                    }
                },
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
    user_id = data.get("user_id")

    try:
        # 1. Generate the three embeddings from the string inputs
        acad_vec, prof_vec, topic_vec = make_embeddings(data)
        
        # Convert numpy arrays to Python lists for JSON serialization
        acad_vec = acad_vec.tolist()
        prof_vec = prof_vec.tolist() 
        topic_vec = topic_vec.tolist()

        # 2. First try to find mentors
        def search_and_collect(user_types, search_type):
            """Helper function to search and collect results for given user types"""
            res_acad = ES.search(
                index=INDEX, 
                body=vector_query_by_user_types("academic_vec", acad_vec, top_k, user_types)
            )
            res_prof = ES.search(
                index=INDEX, 
                body=vector_query_by_user_types("professional_vec", prof_vec, top_k, user_types)
            )
            res_topic = ES.search(
                index=INDEX, 
                body=vector_query_by_user_types("mentorship_topic_vec", topic_vec, top_k, user_types)
            )

            # Aggregate scores per document ID with weighted combination
            scores = {}
            
            def collect(search_results, weight):
                """Helper function to collect and weight scores from search results"""
                for hit in search_results["hits"]["hits"]:
                    doc_id = hit["_id"]
                    score = hit["_score"] * weight
                    scores[doc_id] = scores.get(doc_id, 0.0) + score

            # Apply different weights to each type of match
            collect(res_acad, 0.2)   # Academic background: 20%
            collect(res_prof, 0.3)   # Professional skills: 30% 
            collect(res_topic, 0.5)  # Mentorship topic: 50%
            
            return scores, search_type

        # Try mentors first
        mentor_scores, search_type = search_and_collect(["mentor"], "mentors")
        
        # If no mentors found, try mentees and normal users
        if not mentor_scores:
            other_scores, search_type = search_and_collect(["mentee", "normal_user"], "peers")
            final_scores = other_scores
        else:
            final_scores = mentor_scores

        # 4. Sort by combined score (highest first) and limit to top_k
        ranked_matches = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 5. Fetch full documents for the top matches
        results = []
        for doc_id, combined_score in ranked_matches:
            try:
                doc = ES.get(index=INDEX, id=doc_id)["_source"]
                doc["match_score"] = combined_score
                doc["matched_user_id"] = doc_id
                results.append(doc)
            except Exception as e:
                print(f"Error fetching document {doc_id}: {e}")
                continue

        return jsonify({
            "user_id": user_id,
            "matches": results,
            "total_matches": len(results),
            "search_type": search_type,
            "message": "Found mentors for you!" if search_type == "mentors" else "No mentors available, showing similar peers instead"
        })

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)