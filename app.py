from flask import Flask, jsonify, request
from flask_cors import CORS
import query_engine
import threading
import time

# Main application
app = Flask(__name__)
CORS(app) 

# Health check application
health_app = Flask(__name__ + '_health')
CORS(health_app)

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

# Health check endpoints on separate app
@health_app.route('/health')
def health_status():
    return jsonify({
        "status": "healthy", 
        "message": "Brand Influence Query API health check",
        "timestamp": time.time(),
        "service": "query-api"
    })

@health_app.route('/ready')
def readiness_check():
    try:
        # You can add more sophisticated readiness checks here
        # For example, checking database connectivity
        return jsonify({
            "status": "ready",
            "message": "Service is ready to accept requests",
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "status": "not ready",
            "message": f"Service readiness check failed: {str(e)}",
            "timestamp": time.time()
        }), 503

def run_health_server():
    """Run the health check server on a separate port"""
    health_app.run(debug=False, port=8080, host='0.0.0.0')

if __name__ == '__main__':
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    # Start main application server
    print("Starting main API server on port 5001...")
    print("Health check server running on port 8080...")
    print("Health endpoints: http://localhost:8080/health and http://localhost:8080/ready")
    
    app.run(debug=True, port=5001)
