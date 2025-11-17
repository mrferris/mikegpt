"""
MikeGPT Flask Backend
"""

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from model import Model
import os
import argparse
import json
import torch

app = Flask(__name__, static_folder="static")

# Initialize model
model = None

# Store conversation history per session (in production, use a proper session store)
conversations = {}


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate and stream responses one by one.

    Streams Server-Sent Events (SSE) with each response as it's generated.
    """
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    history = data.get("history", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

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
                if response.startswith("<|") and response.endswith("|>"):
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

    return Response(
        stream_with_context(generate_stream()), mimetype="text/event-stream"
    )


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset conversation history for a session."""
    data = request.json
    session_id = data.get("session_id", "default")

    if session_id in conversations:
        del conversations[session_id]

    return jsonify({"success": True})


@app.route("/api/beam-tree", methods=["POST"])
def beam_tree():
    """
    Generate a beam search tree for token exploration.

    Expects JSON: {"prompt": str, "k": int, "n": int}
    Returns: Tree structure with top K tokens at each of N levels
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    k = data.get("k", 5)
    n = data.get("n", 5)  # Default to 5 levels for performance

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Sanity check to prevent exponential explosion
    if k > 50:
        return jsonify({"error": "k must be <= 50"}), 400
    if n > 10:
        return (
            jsonify(
                {
                    "error": "n must be <= 10 for full tree generation. Use lazy loading for deeper trees."
                }
            ),
            400,
        )

    try:
        tree = model.build_beam_tree(prompt, k=k, n=n)
        return jsonify(tree)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/expand-node", methods=["POST"])
def expand_node():
    """
    Get additional top-k tokens for a specific position in the tree.

    Expects JSON: {
        "prompt": str,
        "path": [token_id, token_id, ...],  # path to current position
        "current_k": int,  # how many we already have
        "additional_k": int,  # how many more to fetch
        "n": int  # depth to expand
    }
    Returns: Additional tokens with their children
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    path = data.get("path", [])
    current_k = data.get("current_k", 0)
    additional_k = data.get("additional_k", 5)
    n = data.get("n", 5)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Build token sequence by following the path (prompt already has special tokens)
        tokens = model.tokenizer.encode(prompt)
        if len(tokens) > model.context_length:
            tokens = tokens[-model.context_length :]

        # Add path tokens
        for token_id in path:
            tokens.append(token_id)
            if len(tokens) > model.context_length:
                tokens = tokens[-model.context_length :]

        tokens_tensor = torch.tensor([tokens], device=model.device, dtype=torch.long)

        # Get top (current_k + additional_k) tokens
        total_k = current_k + additional_k
        top_tokens = model.get_top_k_tokens(tokens_tensor, k=total_k)

        # Return only the new tokens (skip the first current_k)
        new_tokens = top_tokens[current_k:]

        # Build tree for these new tokens
        result = []
        for token_id, token_str, probability in new_tokens:
            new_token = torch.tensor([[token_id]], device=model.device)
            if tokens_tensor.size(1) >= model.context_length:
                new_tokens_tensor = torch.cat([tokens_tensor[:, 1:], new_token], dim=1)
            else:
                new_tokens_tensor = torch.cat([tokens_tensor, new_token], dim=1)

            # Build children recursively
            def build_children(tensor, depth, cumulative_prob):
                if depth >= n:
                    return None
                top_k = model.get_top_k_tokens(tensor, k=current_k + additional_k)
                children = []
                for tid, tstr, prob in top_k:
                    nt = torch.tensor([[tid]], device=model.device)
                    if tensor.size(1) >= model.context_length:
                        next_tensor = torch.cat([tensor[:, 1:], nt], dim=1)
                    else:
                        next_tensor = torch.cat([tensor, nt], dim=1)

                    children.append(
                        {
                            "token_id": tid,
                            "token_str": tstr,
                            "probability": prob,
                            "cumulative_prob": cumulative_prob * prob,
                            "depth": depth,
                            "children": build_children(
                                next_tensor, depth + 1, cumulative_prob * prob
                            ),
                        }
                    )
                return children if children else None

            result.append(
                {
                    "token_id": token_id,
                    "token_str": token_str,
                    "probability": probability,
                    "cumulative_prob": probability,
                    "depth": 0,
                    "children": build_children(new_tokens_tensor, 1, probability),
                }
            )

        return jsonify({"new_tokens": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/drive")
def drive():
    return send_from_directory("static", "drive.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description="Run MikeGPT")

    arguments.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="File from which to load model checkpoint",
    )
    args = arguments.parse_args()

    model = Model(checkpoint_path=args.checkpoint)

    # Create static folder if it doesn't exist
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5002, host="127.0.0.1")
