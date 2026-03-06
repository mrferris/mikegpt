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
    session,
)
from functools import wraps
from model import Model
import os
import argparse
import json
import secrets
import yaml
from pathlib import Path
from datetime import datetime
import resource

app = Flask(__name__, static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(16))

ADMIN_PASSWORD = None  # Set via --admin-password flag or ADMIN_PASSWORD env var
TRAINING_HISTORY_PATH = Path(__file__).parent / "data" / "training_history.yml"
CONVERSATIONS_DIR = Path(__file__).parent / "data" / "conversations"


def load_training_history():
    """Load training history from YAML file."""
    if not TRAINING_HISTORY_PATH.exists():
        return {"steps": []}
    with open(TRAINING_HISTORY_PATH) as f:
        return yaml.safe_load(f) or {"steps": []}


def save_training_step(step_data: dict):
    """Append a training step to the YAML file."""
    history = load_training_history()
    step_data["id"] = len(history["steps"]) + 1
    history["steps"].append(step_data)
    TRAINING_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_HISTORY_PATH, "w") as f:
        yaml.dump(history, f, default_flow_style=False)


def safe_conversation_path(session_id):
    """Return a safe file path for a session, or None if the ID is invalid."""
    # Sanitize: only allow alphanumeric, underscore, hyphen
    safe_id = "".join(c for c in session_id if c.isalnum() or c in ("_", "-"))
    if not safe_id:
        return None
    filepath = CONVERSATIONS_DIR / f"{safe_id}.json"
    # Verify resolved path is within CONVERSATIONS_DIR
    if not filepath.resolve().is_relative_to(CONVERSATIONS_DIR.resolve()):
        return None
    return filepath


def save_conversation(session_id, history, user_agent=None):
    """Persist a conversation to disk."""
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = safe_conversation_path(session_id)
    if not filepath:
        return
    now = datetime.now().isoformat()
    # Preserve created_at and user_agent from first save
    created_at = now
    existing_ua = None
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text())
            created_at = existing.get("created_at", now)
            existing_ua = existing.get("user_agent")
        except (json.JSONDecodeError, KeyError):
            pass
    filepath.write_text(json.dumps({
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": now,
        "history": history,
        "user_agent": existing_ua or user_agent,
    }, indent=2))


def admin_required(f):
    """Decorator to require admin authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# Initialize model
model = None

# Store conversation history per session (in production, use a proper session store)
conversations = {}


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate and stream responses one by one.

    Streams Server-Sent Events (SSE) with each response as it's generated.

    If auto_start=True, MikeGPT sends the first message (no user message required).
    """
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    history = data.get("history", "")
    auto_start = data.get("auto_start", False)

    # If not auto_start, require a user message
    if not auto_start and not user_message:
        return jsonify({"error": "No message provided"}), 400

    def generate_stream():
        try:
            # Get conversation history for this session
            nonlocal history
            if not history and session_id in conversations:
                history = conversations[session_id]

            # Start building new history
            if auto_start:
                # MikeGPT starts first: use <|ConversationStart|><|Me|> as the prompt
                new_history = "<|ConversationStart|><|Me|>"
                prompt_for_model = new_history
            else:
                # Normal mode: user sends first message
                if not history:
                    new_history = f"<|ConversationStart|><|Them|>{user_message}"
                else:
                    new_history = history + f"<|Them|>{user_message}"
                prompt_for_model = new_history

            # Stream each response as it's generated
            for response, token_ids in model.generate_response_stream(
                history if not auto_start else "",
                user_message,
                auto_start=auto_start,
                auto_start_prompt=prompt_for_model if auto_start else None,
            ):
                # Update history for this response
                if response.startswith("<|") and response.endswith("|>"):
                    new_history += f"{response}"
                else:
                    if auto_start and new_history == "<|ConversationStart|><|Me|>":
                        # First response in auto_start mode, don't add another <|Me|>
                        new_history += f"{response}"
                    else:
                        new_history += f"<|Me|>{response}"

                # Send this response immediately with token IDs
                print(
                    f"[generate] Sending response='{response}' with token_ids={token_ids}"
                )
                yield f"data: {json.dumps({'response': response, 'token_ids': token_ids})}\n\n"

            # Save final history
            conversations[session_id] = new_history
            save_conversation(session_id, new_history, request.headers.get("User-Agent"))

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


@app.route("/api/training-history", methods=["GET"])
def get_training_history():
    """Return all training history."""
    return jsonify(load_training_history())


@app.route("/api/checkpoints", methods=["GET"])
def list_checkpoints():
    """List available model checkpoints."""
    from model import Model

    checkpoints = Model.list_checkpoints()
    # Add current checkpoint info
    current = model.current_checkpoint if model else None
    return jsonify({"checkpoints": checkpoints, "current": current})


@app.route("/api/switch-model", methods=["POST"])
def switch_model():
    """Hot-swap to a different checkpoint."""
    from pathlib import Path

    data = request.json
    checkpoint_path = data.get("checkpoint_path")

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return jsonify({"error": "Invalid checkpoint path"}), 400

    model.reload_checkpoint(checkpoint_path)
    return jsonify({"success": True, "loaded": checkpoint_path})


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
        raw = data.get("raw", False)
        tree = model.build_beam_tree(prompt, k=k, n=n, raw=raw)
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
        # Encode prompt once, not per-node
        prompt_tokens = model.tokenizer.encode(prompt)
        if len(prompt_tokens) > model.context_length:
            prompt_tokens = prompt_tokens[-model.context_length :]

        # Build all token sequences and path keys upfront
        sequences = []
        path_keys = []
        for node_info in nodes_to_expand:
            path = node_info["path"]
            node_token_id = node_info["token_id"]

            tokens = list(prompt_tokens)
            tokens.extend(path)
            tokens.append(node_token_id)

            # Truncate to context length once
            if len(tokens) > model.context_length:
                tokens = tokens[-model.context_length :]

            sequences.append(tokens)
            path_keys.append(",".join(map(str, path + [node_token_id])))

        # Batched forward pass — cached nodes are served from cache, only
        # uncached nodes hit the GPU.  Full distributions are stored so
        # subsequent requests with larger k need zero GPU work.
        batch_results = model.get_top_k_cached_batch(sequences, path_keys, prompt, k=k)

        result = {}
        for path_key, top_tokens in zip(path_keys, batch_results):
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
            result[path_key] = children

        # Peak RSS in MB (macOS ru_maxrss is bytes, Linux is KB)
        import sys
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = peak_kb / 1024 if sys.platform == 'linux' else peak_kb / (1024 * 1024)
        cache_entries = len(model._probs_cache)
        cache_mb = sum(t.nelement() * t.element_size() for pair in model._probs_cache.values() for t in pair) / (1024 * 1024)
        print(f"[memory] Peak RSS: {peak_mb:.1f} MB | probs_cache: {cache_entries} entries, {cache_mb:.1f} MB")

        return jsonify({"children_map": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/grpo-generate", methods=["POST"])
def grpo_generate():
    """
    Generate 8 responses for GRPO ranking via streaming SSE.

    Expects JSON:
    {
        "prompt": str,
        "temperature": float (optional, default 1.0),
        "top_k": int (optional),
        "top_p": float (optional),
        "use_top_k": bool (optional, default False)
    }

    Streams:
    - { index: 0-7, token: "...", done: false } per token
    - { index: 0-7, done: true, full_response: "...", tokens: [...] } when response complete
    - { all_done: true, responses: [...] } when all 8 complete
    """
    data = request.json
    prompt_text = data.get("prompt", "").strip()
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 5)
    top_p = data.get("top_p", 0.9)
    use_top_k = data.get("use_top_k", False)

    if not prompt_text:
        return jsonify({"error": "No prompt provided"}), 400

    def generate_stream():
        try:
            responses = []
            seen_texts = set()
            full_prompt = f"<|ConversationStart|><|Them|>{prompt_text}<|Me|>"
            max_attempts = 24  # Prevent infinite loops if model is too deterministic

            attempts = 0
            while len(responses) < 8 and attempts < max_attempts:
                attempts += 1
                model.prime(full_prompt)
                current_response = ""
                response_tokens = []
                max_tokens = 100

                for _ in range(max_tokens):
                    token = model.next_token(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        use_top_k=use_top_k,
                    )

                    # Get the token ID from the last position in current_tokens
                    token_id = int(model.current_tokens[0, -1].item())
                    response_tokens.append(token_id)

                    # Check for stop tokens
                    if token in [
                        "<|Me|>",
                        "<|Them|>",
                        "<|endoftext|>",
                        "<|ConversationStart|>",
                    ]:
                        # Remove the stop token from the list
                        response_tokens.pop()
                        break

                    current_response += token

                # Check for duplicates before adding
                response_text = current_response.strip()
                if response_text in seen_texts:
                    # Duplicate, skip and try again
                    continue

                # Unique response - add it and stream to frontend
                seen_texts.add(response_text)
                i = len(responses)

                # Stream tokens for this response (decode each token individually)
                for tid in response_tokens:
                    token_str = model.tokenizer.decode([tid])
                    yield f"data: {json.dumps({'index': i, 'token': token_str, 'done': False})}\n\n"

                # Send completion for this response
                responses.append({"text": response_text, "tokens": response_tokens})
                yield f"data: {json.dumps({'index': i, 'done': True, 'full_response': response_text, 'tokens': response_tokens})}\n\n"

            # Send final message with all responses
            yield f"data: {json.dumps({'all_done': True, 'responses': responses})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate_stream()), mimetype="text/event-stream"
    )


@app.route("/api/train", methods=["POST"])
def train():
    """
    Unified training endpoint for both pair and group modes.

    Expects JSON:
    {
        "prompt": str,
        "responses": [[int, ...], ...],  # Token ID sequences for each response
        "rewards": [float, ...]          # Reward for each response
    }

    Returns probability changes for each response.
    """
    data = request.json
    prompt_text = data.get("prompt", "")
    responses = data.get("responses", [])
    rewards = data.get("rewards", [])

    if not prompt_text or not responses or not rewards:
        return jsonify({"error": "Missing required fields"}), 400

    if len(responses) != len(rewards):
        return jsonify({"error": "responses and rewards must have same length"}), 400

    try:
        # Tokenize the prompt
        prompt_tokens = model.tokenizer.encode(prompt_text)

        # Call unified training step (loops until target KL reached)
        training_result = model.do_training_step(
            prompt=prompt_tokens, responses=responses, rewards=rewards
        )

        probability_changes = training_result["probability_changes"]
        l2_diff = training_result["l2_diff"]
        kl_divergence = training_result["kl_divergence"]
        steps_taken = training_result["steps_taken"]

        # Determine type based on group size
        train_type = "pair" if len(responses) == 2 else "group"

        # Decode responses to text for display
        response_texts = [model.tokenizer.decode(r) for r in responses]

        # Save to persistent training history
        history = load_training_history()
        step_num = len(history["steps"]) + 1

        checkpoint_name = f"step_{step_num}"

        save_training_step(
            {
                "timestamp": datetime.now().isoformat(),
                "type": train_type,
                "prompt": prompt_text,
                "responses": responses,
                "response_texts": response_texts,
                "rewards": rewards,
                "probability_changes": probability_changes,
                "l2_diff": l2_diff,
                "kl_divergence": kl_divergence,
                "steps_taken": steps_taken,
                "checkpoint": checkpoint_name,
            }
        )

        # Save checkpoint for this training step
        checkpoint_path = model.save_checkpoint(checkpoint_name)

        return jsonify(
            {
                "success": True,
                "prompt_length": len(prompt_tokens),
                "probability_changes": probability_changes,
                "l2_diff": l2_diff,
                "kl_divergence": kl_divergence,
                "steps_taken": steps_taken,
                "checkpoint_saved": checkpoint_name,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    """Authenticate for admin dashboard."""
    if not ADMIN_PASSWORD:
        return jsonify({"error": "Admin not configured"}), 403
    if request.json.get("password") == ADMIN_PASSWORD:
        session["admin"] = True
        return jsonify({"success": True})
    return jsonify({"error": "Invalid password"}), 401


@app.route("/api/admin/training-steps", methods=["GET"])
@admin_required
def admin_training_steps():
    """Return training steps with checkpoint status."""
    history = load_training_history()
    checkpoints_dir = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
    for step in history["steps"]:
        cp_name = step.get("checkpoint", f"step_{step['id']}")
        cp_path = checkpoints_dir / f"{cp_name}.pt"
        step["checkpoint_exists"] = cp_path.exists()
        step["checkpoint_path"] = str(cp_path)
    return jsonify(history)


@app.route("/api/admin/conversations", methods=["GET"])
@admin_required
def admin_conversations():
    """List all saved conversations."""
    convos = []
    if CONVERSATIONS_DIR.exists():
        for f in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
                # Parse message count from history
                hist = data.get("history", "")
                msg_count = hist.count("<|Them|>") + hist.count("<|Me|>")
                # Extract first user message as preview
                preview = ""
                if "<|Them|>" in hist:
                    after_them = hist.split("<|Them|>")[1]
                    preview = after_them.split("<|")[0][:80]
                elif "<|Me|>" in hist:
                    after_me = hist.split("<|Me|>")[1]
                    preview = after_me.split("<|")[0][:80]
                # Parse user agent into short device label
                ua = data.get("user_agent", "")
                device = ""
                if ua:
                    if "iPhone" in ua:
                        device = "iPhone"
                    elif "iPad" in ua:
                        device = "iPad"
                    elif "Android" in ua:
                        device = "Android"
                    elif "Macintosh" in ua:
                        device = "Mac"
                    elif "Windows" in ua:
                        device = "Windows"
                    elif "Linux" in ua:
                        device = "Linux"
                convos.append({
                    "session_id": data.get("session_id", f.stem),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": msg_count,
                    "preview": preview,
                    "device": device,
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return jsonify({"conversations": convos})


@app.route("/api/admin/conversations/<session_id>", methods=["GET"])
@admin_required
def admin_conversation_detail(session_id):
    """Return a single conversation's full history."""
    filepath = safe_conversation_path(session_id)
    if not filepath or not filepath.exists():
        return jsonify({"error": "Not found"}), 404
    data = json.loads(filepath.read_text())
    return jsonify(data)


@app.route("/api/conversation/<session_id>", methods=["GET"])
def get_conversation(session_id):
    """Return a conversation's history (public, for replay)."""
    filepath = safe_conversation_path(session_id)
    if not filepath or not filepath.exists():
        return jsonify({"error": "Not found"}), 404
    data = json.loads(filepath.read_text())
    return jsonify({"history": data.get("history", "")})


@app.route("/api/admin/delete-checkpoint", methods=["POST"])
@admin_required
def admin_delete_checkpoint():
    """Soft-delete a checkpoint (remove file, keep training step record)."""
    step_id = request.json.get("step_id")
    if not step_id:
        return jsonify({"error": "step_id required"}), 400

    history = load_training_history()
    step = next((s for s in history["steps"] if s["id"] == step_id), None)
    if not step:
        return jsonify({"error": "Step not found"}), 404

    # Delete the checkpoint file
    cp_name = step.get("checkpoint", f"step_{step_id}")
    checkpoints_dir = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
    cp_path = checkpoints_dir / f"{cp_name}.pt"
    if cp_path.exists():
        cp_path.unlink()

    # Mark as deleted in training history
    step["deleted"] = True
    with open(TRAINING_HISTORY_PATH, "w") as f:
        yaml.dump(history, f, default_flow_style=False)

    return jsonify({"success": True})


@app.route("/api/admin/rollback", methods=["POST"])
@admin_required
def admin_rollback():
    """Rollback model to a specific training step's checkpoint."""
    step_id = request.json.get("step_id")
    if not step_id:
        return jsonify({"error": "step_id required"}), 400

    history = load_training_history()
    step = next((s for s in history["steps"] if s["id"] == step_id), None)
    if not step:
        return jsonify({"error": "Step not found"}), 404

    cp_name = step.get("checkpoint", f"step_{step_id}")
    checkpoints_dir = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
    cp_path = checkpoints_dir / f"{cp_name}.pt"

    if not cp_path.exists():
        return jsonify({"error": "Checkpoint file not found (deleted?)"}), 404

    model.reload_checkpoint(str(cp_path))
    return jsonify({"success": True, "loaded": str(cp_path)})


@app.route("/api/admin/rollback-pretrained", methods=["POST"])
@admin_required
def admin_rollback_pretrained():
    """Rollback model to the base pretrained checkpoint."""
    checkpoints_dir = Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints"))
    cp_path = checkpoints_dir / "pretrained.pt"
    if not cp_path.exists():
        return jsonify({"error": "pretrained.pt not found"}), 404
    model.reload_checkpoint(str(cp_path))
    return jsonify({"success": True, "loaded": str(cp_path)})


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/mike-rl")
def drive():
    return send_from_directory("static", "drive.html")


@app.route("/admin")
def admin():
    return send_from_directory("static", "admin.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description="Run MikeGPT")

    default_checkpoint = os.path.join(
        os.environ.get("CHECKPOINTS_DIR", "checkpoints"),
        "pretrained.pt",
    )
    arguments.add_argument(
        "--checkpoint",
        type=str,
        default=default_checkpoint,
        help="File from which to load model checkpoint (default: $CHECKPOINTS_DIR/pretrained.pt)",
    )
    arguments.add_argument(
        "--admin-password",
        type=str,
        default=os.environ.get("ADMIN_PASSWORD"),
        help="Password for /admin dashboard (or set ADMIN_PASSWORD env var)",
    )
    args = arguments.parse_args()

    global ADMIN_PASSWORD
    ADMIN_PASSWORD = args.admin_password
    model = Model(checkpoint_path=args.checkpoint)

    # Create static folder if it doesn't exist
    os.makedirs("static", exist_ok=True)
    app.run(debug=False, port=5002, host="0.0.0.0")
