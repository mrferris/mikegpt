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
            # Add <|ConversationStart|> only at the beginning of a new conversation
            if not history:
                new_history = f"<|ConversationStart|><|Them|>{user_message}"
            else:
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
    if k > 500:
        return jsonify({"error": "k must be <= 500"}), 400
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


@app.route("/api/expand-depth", methods=["POST"])
def expand_depth():
    """
    Extend the tree depth by one layer for specific nodes.

    Expects JSON: {
        "prompt": str,
        "nodes": [
            {
                "path": [token_id, ...],  # path to this node
                "token_id": int  # the node itself
            },
            ...
        ],
        "k": int  # how many children to generate per node
    }
    Returns: Map of node paths to their children
    """
    data = request.json
    prompt = data.get("prompt", "").strip()
    nodes_to_expand = data.get("nodes", [])
    k = data.get("k", 5)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = {}

        for node_info in nodes_to_expand:
            path = node_info["path"]
            node_token_id = node_info["token_id"]

            # Build token sequence by following the path
            tokens = model.tokenizer.encode(prompt)
            if len(tokens) > model.context_length:
                tokens = tokens[-model.context_length :]

            # Add path tokens + this node's token
            for token_id in path:
                tokens.append(token_id)
                if len(tokens) > model.context_length:
                    tokens = tokens[-model.context_length :]

            tokens.append(node_token_id)
            if len(tokens) > model.context_length:
                tokens = tokens[-model.context_length :]

            tokens_tensor = torch.tensor(
                [tokens], device=model.device, dtype=torch.long
            )

            # Get top k tokens for this node
            top_tokens = model.get_top_k_tokens(tokens_tensor, k=k)

            children = []
            for token_id, token_str, probability in top_tokens:
                children.append(
                    {
                        "token_id": token_id,
                        "token_str": token_str,
                        "probability": probability,
                        "cumulative_prob": probability,  # Will be updated by frontend
                        "depth": 0,  # Will be updated by frontend
                        "children": None,  # Will be loaded lazily
                    }
                )

            # Use path + node_token_id as key
            path_key = ",".join(map(str, path + [node_token_id]))
            result[path_key] = children

        return jsonify({"children_map": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def train():
    """
    Train the model using DPO with positive and negative token ID sequences.

    Expects JSON:
    {
        "prompt": str,
        "positive_token_ids": [[int, ...], ...],  # Token ID sequences (continuations only)
        "negative_token_ids": [[int, ...], ...]   # Token ID sequences (continuations only)
    }
    """
    data = request.json
    prompt_text = data.get("prompt", "")
    positive_token_ids = data.get("positive_token_ids", [])
    negative_token_ids = data.get("negative_token_ids", [])

    if not prompt_text or not positive_token_ids or not negative_token_ids:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Tokenize the prompt (this is the shared context)
        prompt_tokens = model.tokenizer.encode(prompt_text)

        # Use the first positive and first negative token sequence for this training step
        # The token IDs from the frontend are already the continuation only (not including prompt)
        positive_continuation = positive_token_ids[0]
        negative_continuation = negative_token_ids[0]

        # Call the DPO training step
        probability_changes = model.do_dpo_step(
            prompt=prompt_tokens,
            positive=positive_continuation,
            negative=negative_continuation,
        )

        return jsonify(
            {
                "success": True,
                "prompt_length": len(prompt_tokens),
                "positive_length": len(positive_continuation),
                "negative_length": len(negative_continuation),
                "num_positive_paths": len(positive_token_ids),
                "num_negative_paths": len(negative_token_ids),
                "positive_change_percent": probability_changes[0],
                "negative_change_percent": probability_changes[1],
            }
        )

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
