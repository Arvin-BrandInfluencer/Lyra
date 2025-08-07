from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import query_engine

# Main application
app = Flask(__name__)
CORS(app)

# Health check application
health_app = Flask(__name__ + "_health")
CORS(health_app)

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

# Dedicated health check endpoint
@health_app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Brand Influence Query API is running.",
        "port": 5001
    })

@health_app.route('/')
def health_root():
    return jsonify({
        "status": "healthy", 
        "message": "Health check service is running.",
        "main_service_port": 5001
    })

def run_health_server():
    health_app.run(debug=False, port=5002, host='0.0.0.0')

if __name__ == '__main__':
    # Start health check server in a separate thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    
    # Start main application
    app.run(debug=True, port=5001, host='0.0.0.0')
