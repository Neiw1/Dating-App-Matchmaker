from flask import Flask, request, jsonify, render_template_string
from cupid.model.main import CupidMatchmaker
import os
import json

app = Flask(__name__)

# Global matchmaker instance
matchmaker = None

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cupid Matchmaker</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }
            button { background: #ff6b9d; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #ff4d8d; }
            input { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 3px; }
            .result { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>💘 Cupid Matchmaker</h1>
        <p>The best matchmaker for Chula students!</p>
        
        <div class="container">
            <h2>🚀 Train Models</h2>
            <p>First, train the models with the speed dating dataset:</p>
            <button onclick="trainModels()">Train Models</button>
            <div id="training-status"></div>
        </div>
        
        <div class="container">
            <h2>💕 Find Matches</h2>
            <p>Enter a user ID to find their top matches:</p>
            <input type="number" id="user-id" placeholder="Enter user ID">
            <input type="number" id="top-k" placeholder="Number of matches (default: 5)" value="5">
            <button onclick="findMatches()">Find Matches</button>
            <button onclick="getAvailableUsers()">Show Available Users</button>
            <div id="matches-result"></div>
        </div>
        
        <div class="container">
            <h2>👥 Available Users</h2>
            <button onclick="getAvailableUsers()">Load Available Users</button>
            <div id="users-result"></div>
        </div>
        
        <div class="container">
            <h2>📊 Model Status</h2>
            <button onclick="checkStatus()">Check Status</button>
            <div id="status-result"></div>
        </div>
        
        <script>
            async function trainModels() {
                const statusDiv = document.getElementById('training-status');
                statusDiv.innerHTML = 'Training models... This may take a few minutes.';
                
                try {
                    const response = await fetch('/train', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.success) {
                        statusDiv.innerHTML = '<div class="result">' + result.message + '</div>';
                    } else {
                        statusDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + result.error + '</div>';
                    }
                } catch (error) {
                    statusDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + error.message + '</div>';
                }
            }
            
            async function findMatches() {
                const userId = document.getElementById('user-id').value;
                const topK = document.getElementById('top-k').value || 5;
                const resultDiv = document.getElementById('matches-result');
                
                if (!userId) {
                    resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Please enter a user ID</div>';
                    return;
                }
                
                try {
                    const response = await fetch('/matches', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ user_id: parseInt(userId), top_k: parseInt(topK) })
                    });
                    const result = await response.json();
                    
                    if (result.success) {
                        let html = '<div class="result"><h3>Top ' + topK + ' Matches:</h3>';
                        result.matches.forEach((match, index) => {
                            html += '<p><strong>Match ' + (index + 1) + ':</strong> User ' + match[0] + ' (Compatibility: ' + (match[1] * 100).toFixed(1) + '%)</p>';
                        });
                        html += '</div>';
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + result.error + '</div>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + error.message + '</div>';
                }
            }
            
            async function checkStatus() {
                const resultDiv = document.getElementById('status-result');
                
                try {
                    const response = await fetch('/status');
                    const result = await response.json();
                    resultDiv.innerHTML = '<div class="result">' + result.message + '</div>';
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + error.message + '</div>';
                }
            }
            
            async function getAvailableUsers() {
                const resultDiv = document.getElementById('users-result');
                resultDiv.innerHTML = 'Loading available users...';
                
                try {
                    const response = await fetch('/users');
                    const result = await response.json();
                    
                    if (result.success) {
                        let html = '<div class="result"><h3>Available Users (' + result.total_users + ' total):</h3>';
                        html += '<p><strong>First 20 users:</strong></p><ul>';
                        result.users.slice(0, 20).forEach(user => {
                            html += '<li>User ID: ' + user.user_id + ' (Index: ' + user.index + ')</li>';
                        });
                        if (result.users.length > 20) {
                            html += '<li>... and ' + (result.users.length - 20) + ' more users</li>';
                        }
                        html += '</ul></div>';
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + result.error + '</div>';
                    }
                } catch (error) {
                    resultDiv.innerHTML = '<div class="result" style="background: #ffe6e6;">Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/train', methods=['POST'])
def train():
    global matchmaker
    
    try:
        # Initialize matchmaker
        matchmaker = CupidMatchmaker()
        
        # Load and process data
        processed_data = matchmaker.load_and_process_data()
        
        # Prepare training data
        data_dict = matchmaker.prepare_training_data(processed_data)
        
        # Train models (with more epochs for better performance)
        train_losses, val_losses = matchmaker.train_models(
            data_dict, autoencoder_epochs=50, matching_epochs=50
        )
        
        # Save models
        matchmaker.save_models()
        
        return jsonify({
            'success': True,
            'message': f'Models trained successfully! Autoencoder epochs: {len(train_losses)}, Matching epochs: {len(val_losses)}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/matches', methods=['POST'])
def get_matches():
    """
    Get top-k matches for a user based on compatibility scores.
    
    Request Schema:
    {
        "user_id": int,     // Required: User ID from speed dating dataset
        "top_k": int        // Optional: Number of matches (default: 5, max: 20)
    }
    
    Response Schema:
    {
        "success": bool,
        "matches": [
            [partner_id, compatibility_score],  // partner_id, score (0-1)
            ...
        ],
        "user_info": {      // Basic info about the user
            "user_id": int,
            "age": int,
            "gender": str
        }
    }
    """
    global matchmaker
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
            
        user_id = data.get('user_id')
        if user_id is None:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
            
        top_k = data.get('top_k', 10)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            return jsonify({
                'success': False,
                'error': 'top_k must be an integer between 1 and 20'
            }), 400
        
        if matchmaker is None:
            # Try to load existing models
            try:
                matchmaker = CupidMatchmaker()
                matchmaker.load_models()
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': 'Models not trained. Please train models first.'
                }), 400
        
        # Validate user exists
        if user_id not in matchmaker.user_id_to_index:
            available_users = list(matchmaker.user_id_to_index.keys())[:10]  # Show first 10
            return jsonify({
                'success': False,
                'error': f'User {user_id} not found in training data. Available users (first 10): {available_users}'
            }), 400
        
        matches = matchmaker.get_matches(user_id, top_k=top_k)
        
        # Get basic user info (you'd need to extend this with actual user data)
        user_info = {
            "user_id": user_id,
            "message": "User details not implemented yet"
        }
        
        return jsonify({
            'success': True,
            'matches': matches,
            'user_info': user_info,
            'total_available_partners': len(matchmaker.partner_id_to_index)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/users', methods=['GET'])
def get_available_users():
    """
    Get list of available user IDs that can be used for matching.
    
    Response Schema:
    {
        "success": bool,
        "users": [
            {
                "user_id": int,
                "index": int
            },
            ...
        ],
        "total_users": int
    }
    """
    global matchmaker
    
    try:
        if matchmaker is None:
            # Try to load existing models
            try:
                matchmaker = CupidMatchmaker()
                matchmaker.load_models()
            except:
                return jsonify({
                    'success': False,
                    'error': 'Models not trained. Please train models first.'
                }), 400
        
        users = [
            {"user_id": user_id, "index": index} 
            for user_id, index in matchmaker.user_id_to_index.items()
        ]
        
        return jsonify({
            'success': True,
            'users': users,
            'total_users': len(users)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/status')
def status():
    global matchmaker
    
    try:
        if matchmaker is None:
            # Check if models exist
            if os.path.exists('models/autoencoder.pth') and os.path.exists('models/matching_model.pth'):
                matchmaker = CupidMatchmaker()
                matchmaker.load_models()
                return jsonify({
                    'success': True,
                    'message': f'Models loaded successfully! Users: {len(matchmaker.user_id_to_index)}, Partners: {len(matchmaker.partner_id_to_index)}'
                })
            else:
                return jsonify({
                    'success': True,
                    'message': 'Models not trained yet. Please train models first.'
                })
        else:
            return jsonify({
                'success': True,
                'message': f'Models ready! Users: {len(matchmaker.user_id_to_index)}, Partners: {len(matchmaker.partner_id_to_index)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)