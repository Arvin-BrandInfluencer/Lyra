from flask import Flask, jsonify, request
from flask_cors import CORS
import query_engine

app = Flask(__name__)
CORS(app) 

@app.route('/')
def health_check():
    return jsonify({"status": "healthy", "message": "Brand Influence Query API is running."})

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid request: No JSON payload received."}), 400
        
        result = query_engine.execute_query(payload)
        
        if "error" in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
