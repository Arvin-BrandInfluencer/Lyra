from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import query_engine

# Main application
app = Flask(__name__)
CORS(app)

# Health check application on separate port
health_app = Flask(__name__)
CORS(health_app)

# Health check endpoint on dedicated port
@health_app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Brand Influence Query API is running.",
        "service": "arvin-brandinfluencer-api"
    })

@health_app.route('/')
def health_root():
    return jsonify({
        "status": "healthy", 
        "message": "Health check service is running.",
        "endpoints": ["/health"]
    })

# Main application endpoints
@app.route('/')
def main_root():
    return jsonify({
        "service": "Brand Influence Query API",
        "version": "1.0",
        "endpoints": ["/query"],
        "health_port": 8080
    })

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

def run_health_server():
    """Run health check server on port 8080"""
    health_app.run(debug=False, port=8080, host='0.0.0.0')

def run_main_server():
    """Run main application server on port 5001"""
    app.run(debug=True, port=5001, host='0.0.0.0')

if __name__ == '__main__':
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    print("Health check server started on port 8080")
    print("Starting main application server on port 5001")
    
    # Start main application server
    run_main_server()
