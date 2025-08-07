from flask import Flask, jsonify, request
from flask_cors import CORS
import query_engine
import time

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

# Health endpoints that support both GET and POST methods
@app.route('/health', methods=['GET', 'POST'])
def health_status():
    return jsonify({
        "status": "healthy", 
        "message": "Brand Influence Query API health check",
        "timestamp": time.time(),
        "service": "query-api",
        "method": request.method
    })

@app.route('/ready', methods=['GET', 'POST'])
def readiness_check():
    try:
        # Add more sophisticated readiness checks here if needed
        # For example, checking database connectivity with Supabase
        return jsonify({
            "status": "ready",
            "message": "Service is ready to accept requests",
            "timestamp": time.time(),
            "method": request.method
        })
    except Exception as e:
        return jsonify({
            "status": "not ready",
            "message": f"Service readiness check failed: {str(e)}",
            "timestamp": time.time(),
            "method": request.method
        }), 503

@app.route('/healthz', methods=['GET', 'POST'])
def kubernetes_health():
    """Kubernetes-style health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time()
    })

if __name__ == '__main__':
    print("Starting API server on port 5001...")
    print("Health endpoints available:")
    print("  - http://localhost:5001/health")
    print("  - http://localhost:5001/ready") 
    print("  - http://localhost:5001/healthz")
    
    app.run(debug=True, port=5001)
