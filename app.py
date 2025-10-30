"""
MikeGPT Flask Backend
"""

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from model import Model
import os
import argparse
import json

app = Flask(__name__, static_folder='static')

# Initialize model
model = None

# Store conversation history per session (in production, use a proper session store)
conversations = {}

@app.route('/api/generate', methods=['POST'])
def generate():
    """
    Generate and stream responses one by one.

    Streams Server-Sent Events (SSE) with each response as it's generated.
    """
    data = request.json
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default')
    history = data.get('history', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    def generate_stream():
        try:
            # Get conversation history for this session
            nonlocal history
            if not history and session_id in conversations:
                history = conversations[session_id]

            # Start building new history
            new_history = history + f"<|Them|>{user_message}"

            # Stream each response as it's generated
            for response in model.generate_response_stream(history, user_message):
                # Update history for this response
                if response.startswith('<|') and response.endswith('|>'):
                    new_history += f"{response}"
                else:
                    new_history += f"<|Me|>{response}"

                # Send this response immediately
                yield f"data: {json.dumps({'response': response})}\n\n"

            # Save final history
            conversations[session_id] = new_history

            # Send final message with updated history
            yield f"data: {json.dumps({'done': True, 'history': new_history})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation history for a session."""
    data = request.json
    session_id = data.get('session_id', 'default')

    if session_id in conversations:
        del conversations[session_id]

    return jsonify({'success': True})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':

    arguments = argparse.ArgumentParser(description="Run MikeGPT")

    arguments.add_argument("--checkpoint", type=str, required=True, help="File from which to load model checkpoint")
    args = arguments.parse_args()

    model = Model(checkpoint_path = args.checkpoint)

    # Create static folder if it doesn't exist
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)
