from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────
ES_HOST = "https://a7b16fc079fb47698fe7b89b50ff406b.us-east-1.aws.found.io:443"
API_KEY = "a0NBYkpaY0JpaWhlRDhLb1ZtX186YnJjU3lyNF9ZNm02ME83TS05RUNUQQ=="
INDEX   = "version-1"

# Connect to Elasticsearch
ES = Elasticsearch(ES_HOST, api_key=API_KEY)

# Load the JobBERT-v2 model for generating embeddings
MODEL = SentenceTransformer("TechWolf/JobBERT-v2")

app = Flask(__name__)

# ─── Utility Functions ─────────────────────────────────────────────────────────

def make_embeddings(record):
    """
    Input JSON must contain:
      - acad_string    (string for degree+department+specialization)
      - prof_string    (string for skills+designation+industry)
      - topic_string   (string for mentee_problem or mentor_topic)

    Returns three numpy arrays (768 dims each).
    """
    academic_text     = record["acad_string"]
    professional_text = record["prof_string"]
    topic_text        = record["topic_string"]

    academic_vec      = MODEL.encode(academic_text,    convert_to_numpy=True)
    professional_vec  = MODEL.encode(professional_text, convert_to_numpy=True)
    topic_vec         = MODEL.encode(topic_text,       convert_to_numpy=True)

    return academic_vec, professional_vec, topic_vec


def vector_query_for(field_name, vector_list, k, user_types):
    """
    Builds a script_score query that:
      • filters on user_type in the provided list (e.g. ["mentor"], ["normal_user"], ["mentee"])
      • uses cosineSimilarity(params.vec, '<field_name>') + 1.0 as the scoring function.

    `field_name` must exactly match one of:
      - "academic_vec"
      - "professional_vec"
      - "mentorship_topic_vec"
    """
    return {
      "size": k,
      "query": {
        "script_score": {
          "query": {
            "bool": {
              "must": [
                { "terms": { "user_type": user_types } }
              ]
            }
          },
          "script": {
            "source": f"cosineSimilarity(params.vec, '{field_name}') + 1.0",
            "params": { "vec": vector_list }
          }
        }
      }
    }


def pop_vectors_from_doc(doc_source):
    """
    Remove vector fields from an Elasticsearch _source dict:
      • "academic_vec"
      • "professional_vec"
      • "mentorship_topic_vec"
    """
    doc_source.pop("academic_vec",      None)
    doc_source.pop("professional_vec",  None)
    doc_source.pop("mentorship_topic_vec", None)
    return doc_source


# ─── /match ENDPOINT (MENTOR‐ONLY, no fallback) ─────────────────────────────────

@app.route("/match", methods=["POST"])
def match():
    data = request.json

    # 1) Validate fields
    for fld in ["user_id", "acad_string", "prof_string", "topic_string"]:
        if fld not in data:
            return jsonify({"error": f"Missing required field: {fld}"}), 400

    top_k   = data.get("top_k", 5)
    user_id = data["user_id"]

    try:
        # 2) Generate embeddings
        acad_vec, prof_vec, topic_vec = make_embeddings(data)
        acad_list  = acad_vec.tolist()
        prof_list  = prof_vec.tolist()
        topic_list = topic_vec.tolist()

        # 3) Only search mentors (three-vector scoring):
        scores = {}

        # Academic match (20% weight)
        res_acad = ES.search(
            index=INDEX,
            body=vector_query_for("academic_vec", acad_list, top_k, ["mentor"])
        )
        # Professional match (30% weight)
        res_prof = ES.search(
            index=INDEX,
            body=vector_query_for("professional_vec", prof_list, top_k, ["mentor"])
        )
        # Topic match (50% weight)
        res_topic= ES.search(
            index=INDEX,
            body=vector_query_for("mentorship_topic_vec", topic_list, top_k, ["mentor"])
        )

        def collect(res, weight):
            for hit in res["hits"]["hits"]:
                doc_id = hit["_id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + hit["_score"] * weight

        collect(res_acad,  0.2)
        collect(res_prof,  0.3)
        collect(res_topic, 0.5)

        # 4) If no mentors found, return empty results immediately
        if not scores:
            return jsonify({
                "user_id": user_id,
                "matches": [],
                "total_matches": 0,
                "search_type": "mentors",
                "message": "No mentors found."
            })

        # 5) Otherwise, sort by score descending and take top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 6) Fetch each doc, pop vector fields, attach match metadata
        results = []
        for doc_id, combined_score in ranked:
            try:
                src = ES.get(index=INDEX, id=doc_id)["_source"]
                pop_vectors_from_doc(src)
                src["match_score"]     = combined_score
                src["matched_user_id"] = doc_id
                results.append(src)
            except Exception as e:
                print(f"Warning: failed to retrieve {doc_id}: {e}")
                continue

        return jsonify({
            "user_id":       user_id,
            "matches":       results,
            "total_matches": len(results),
            "search_type":   "mentors",
            "message":       "Found mentors for you!"
        })

    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


# ─── /recommend ENDPOINT (NORMAL_USER FIRST, THEN MENTEE) ───────────────────────

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    # 1) Validate required fields
    for fld in ["user_id", "acad_string", "prof_string", "topic_string"]:
        if fld not in data:
            return jsonify({"error": f"Missing required field: {fld}"}), 400

    top_k   = data.get("top_k", 5)
    user_id = data["user_id"]

    try:
        # 2) Generate academic embedding
        acad_vec = MODEL.encode(data["acad_string"], convert_to_numpy=True)
        acad_list= acad_vec.tolist()

        # 3) Merge prof + topic into a single text, then embed
        merged_prof_topic = data["prof_string"] + " " + data["topic_string"]
        prof_topic_vec    = MODEL.encode(merged_prof_topic, convert_to_numpy=True)
        prof_topic_list   = prof_topic_vec.tolist()

        # Prepare container for combined scores
        scores_normal = {}
        scores_mentee = {}

        # ─ Normal_User search ───────────────────────────────────────────────────
        # 4a) Academic match for normal_user (weight 0.3)
        res_a = ES.search(
            index=INDEX,
            body=vector_query_for("academic_vec", acad_list, top_k, ["normal_user"])
        )
        for hit in res_a["hits"]["hits"]:
            doc_id = hit["_id"]
            scores_normal[doc_id] = scores_normal.get(doc_id, 0.0) + hit["_score"] * 0.3

        # 4b) “Professional+Topic” match for normal_user (weight 0.7, using same professional_vec)
        res_pt = ES.search(
            index=INDEX,
            body=vector_query_for("professional_vec", prof_topic_list, top_k, ["normal_user"])
        )
        for hit in res_pt["hits"]["hits"]:
            doc_id = hit["_id"]
            scores_normal[doc_id] = scores_normal.get(doc_id, 0.0) + hit["_score"] * 0.7

        # Sort the normal_user results by score descending
        ranked_normal = sorted(scores_normal.items(), key=lambda x: x[1], reverse=True)[:top_k]
        num_normal    = len(ranked_normal)

        # If we already have top_k normal_users, we don’t need to look at mentees
        if num_normal >= top_k:
            final_ranked = ranked_normal[:top_k]
        else:
            # We need additional (top_k - num_normal) mentees
            remaining_k = top_k - num_normal

            # ─ Mentee search ───────────────────────────────────────────────────
            scores_temp = {}

            # 5a) Academic match for mentee (weight 0.3)
            res_ma = ES.search(
                index=INDEX,
                body=vector_query_for("academic_vec", acad_list, remaining_k, ["mentee"])
            )
            for hit in res_ma["hits"]["hits"]:
                doc_id = hit["_id"]
                scores_temp[doc_id] = scores_temp.get(doc_id, 0.0) + hit["_score"] * 0.3

            # 5b) “Professional+Topic” match for mentee (weight 0.7)
            res_mt = ES.search(
                index=INDEX,
                body=vector_query_for("professional_vec", prof_topic_list, remaining_k, ["mentee"])
            )
            for hit in res_mt["hits"]["hits"]:
                doc_id = hit["_id"]
                scores_temp[doc_id] = scores_temp.get(doc_id, 0.0) + hit["_score"] * 0.7

            # Sort mentee results
            ranked_mentee = sorted(scores_temp.items(), key=lambda x: x[1], reverse=True)[:remaining_k]

            # Combine normal_user + mentee scores, preserving the order
            final_ranked = ranked_normal + ranked_mentee

        # 6) Fetch full docs and remove vector fields
        recommendations = []
        for doc_id, combined_score in final_ranked:
            try:
                src = ES.get(index=INDEX, id=doc_id)["_source"]
                pop_vectors_from_doc(src)
                src["match_score"]     = combined_score
                src["matched_user_id"] = doc_id
                recommendations.append(src)
            except Exception as e:
                print(f"Warning: could not load {doc_id}: {e}")
                continue

        return jsonify({
            "user_id":         user_id,
            "recommendations": recommendations,
            "total_recs":      len(recommendations),
            "message":         "Recommended normal users first, then mentees"
        })

    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
