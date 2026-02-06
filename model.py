import sys
from pathlib import Path

# Add sibling repo to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "artisinal-lm"))

from lm.model.model import TransformerLM, TrainableModel
from lm.training.utils.checkpointing import load_checkpoint
from lm.tokenization.bpe import Tokenizer
import torch
import torch.nn.functional as F


class Model:
    def __init__(self, checkpoint_path):
        self.device = "mps"
        self.context_length = 256
        d_model = 256
        vocab_size = 8192
        num_heads = 4
        num_layers = 4
        d_ff = 1344
        rope_theta = 10000

        self.model = (
            TransformerLM(
                d_model=d_model,
                vocab_size=vocab_size,
                context_length=self.context_length,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                device=self.device,
            )
            .to(self.device)
            .eval()
        )

        load_checkpoint(checkpoint_path, self.model, None)

        self.trainable_model = TrainableModel(model=self.model)

        # Load vocab to extract emojis
        import json

        with open(f"vocab/mikegpt_vocab_{vocab_size}.json") as f:
            vocab_json = json.load(f)

        # Extract emojis from vocab (tokens 10-282)
        emojis = []
        for token_id in range(10, 283):
            if str(token_id) in vocab_json:
                hex_bytes = vocab_json[str(token_id)]
                emoji = bytes.fromhex(hex_bytes).decode("utf-8", errors="replace")
                emojis.append(emoji)

        # Define special tokens (core + reactions + emojis)
        special_tokens = [
            "<|endoftext|>",
            "<|Me|>",
            "<|Them|>",
            "<|ConversationStart|>",
            "<|Loved|>",
            "<|Liked|>",
            "<|Laughed at|>",
            "<|Disliked|>",
            "<|Questioned|>",
            "<|Emphasized|>",
        ] + emojis

        self.tokenizer = Tokenizer.from_files(
            f"vocab/mikegpt_vocab_{vocab_size}.json",
            f"vocab/mikegpt_merges_{vocab_size}.pkl",
            special_tokens=special_tokens,
        )

        self.current_tokens = None  # running token buffer on device

        # Cache full sorted probability distributions so repeated/expanding
        # top-k queries for the same node don't need another forward pass.
        # Key: path_key string, Value: (sorted_probs, sorted_indices) CPU tensors
        self._probs_cache = {}
        self._cache_prompt = None
        self.current_checkpoint = checkpoint_path

    def save_checkpoint(self, name: str = None) -> str:
        """Save current model state and optimizer state. Returns the checkpoint path."""
        from datetime import datetime

        checkpoints_dir = Path(__file__).parent / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        checkpoint_path = checkpoints_dir / f"{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.trainable_model.grpo_optimizer.state_dict(),
            },
            checkpoint_path,
        )
        self.current_checkpoint = str(checkpoint_path)
        return str(checkpoint_path)

    def reload_checkpoint(self, checkpoint_path: str):
        """Hot-swap model weights and optimizer state from a different checkpoint."""
        state = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(state["model"])
        # Reinitialize trainable model with new weights
        self.trainable_model = TrainableModel(model=self.model)
        # Restore optimizer state if available
        if "optimizer" in state:
            self.trainable_model.grpo_optimizer.load_state_dict(state["optimizer"])
        # Clear caches
        self._probs_cache = {}
        self._cache_prompt = None
        self.current_checkpoint = checkpoint_path

    @staticmethod
    def list_checkpoints() -> list[dict]:
        """List available checkpoints with metadata."""
        checkpoints_dir = Path(__file__).parent / "checkpoints"
        if not checkpoints_dir.exists():
            return []

        checkpoints = []
        for f in checkpoints_dir.glob("*.pt"):
            checkpoints.append(
                {
                    "name": f.stem,
                    "path": str(f),
                    "modified": f.stat().st_mtime,
                }
            )
        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)

    def prime(self, prompt: str):
        """Tokenize prompt once and store on device."""
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.context_length:
            tokens = tokens[-self.context_length :]
        self.current_tokens = torch.tensor(
            [tokens], device=self.device, dtype=torch.long
        )

    def next_token(
        self,
        temperature: float = 1.0,
        top_p: float = 0.45,
        top_k: int = 5,
        use_top_k: bool = False,
    ) -> str:
        """
        Fast next-token generation with temperature and top-p or top-k sampling.

        Args:
            temperature: Temperature for scaling logits (higher = more random)
            top_p: Nucleus sampling threshold (used when use_top_k=False)
            top_k: Number of top tokens to sample from (used when use_top_k=True)
            use_top_k: If True, use top-k sampling; if False, use top-p sampling (default)
        """
        if self.current_tokens is None:
            raise RuntimeError("Must call .prime(prompt) before .next_token()")

        with torch.no_grad():
            logits = self.model(self.current_tokens)
            last_logits = logits[0, -1] / temperature  # apply temperature
            last_logits[0] = float("-inf")  # suppress <|endoftext|>
            probs = F.softmax(last_logits, dim=-1)

            if use_top_k:
                # Top-k sampling
                top_probs, top_idx = torch.topk(probs, k=top_k)
                top_probs = top_probs / top_probs.sum()  # renormalize to sum to 1

                # Sample one token from the top-k distribution
                choice = torch.multinomial(top_probs, 1)
                chosen_id = int(top_idx[choice].item())
            else:
                # Top-p (nucleus) sampling
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Find the cutoff index where cumulative probability exceeds top_p
                # Keep at least one token
                cutoff_index = torch.searchsorted(cumulative_probs, top_p) + 1

                # Select tokens up to cutoff
                nucleus_probs = sorted_probs[:cutoff_index]
                nucleus_indices = sorted_indices[:cutoff_index]

                # Renormalize to sum to 1
                nucleus_probs = nucleus_probs / nucleus_probs.sum()

                # Sample one token from the nucleus distribution
                choice = torch.multinomial(nucleus_probs, 1)
                chosen_id = int(nucleus_indices[choice].item())

        # Append new token on GPU, cropping if needed
        new_token = torch.tensor([[chosen_id]], device=self.device)
        if self.current_tokens.size(1) >= self.context_length:
            self.current_tokens = torch.cat(
                [self.current_tokens[:, 1:], new_token], dim=1
            )
        else:
            self.current_tokens = torch.cat([self.current_tokens, new_token], dim=1)

        return self.tokenizer.decode([chosen_id])

    def get_top_k_tokens(self, tokens_tensor, k: int = 20, temperature: float = 1.0):
        """
        Get top K tokens and their probabilities for a given token sequence.
        Does not modify model state.

        Args:
            tokens_tensor: Token tensor (shape: [1, seq_len])
            k: Number of top tokens to return

        Returns:
            List of tuples: [(token_id, token_str, probability), ...]
        """
        with torch.no_grad():
            logits = self.model(tokens_tensor)
            last_logits = logits[0, -1] / temperature
            last_logits[0] = float("-inf")  # suppress <|endoftext|>
            probs = F.softmax(last_logits, dim=-1)

            top_probs, top_idx = torch.topk(probs, k=k)

            results = []
            for prob, idx in zip(top_probs, top_idx):
                token_id = int(idx.item())
                token_str = self.tokenizer.decode([token_id])
                probability = float(prob.item())
                results.append((token_id, token_str, probability))

            return results

    def get_top_k_tokens_batch(
        self, sequences: list, k: int = 20, temperature: float = 1.0
    ):
        """
        Get top K tokens for multiple sequences in a single batched forward pass.

        Args:
            sequences: List of token lists (potentially different lengths)
            k: Number of top tokens to return per sequence
            temperature: Temperature for scaling logits

        Returns:
            List of lists of tuples: [[(token_id, token_str, probability), ...], ...]
        """
        if not sequences:
            return []

        if len(sequences) == 1:
            tokens_tensor = torch.tensor(
                [sequences[0]], device=self.device, dtype=torch.long
            )
            return [self.get_top_k_tokens(tokens_tensor, k=k, temperature=temperature)]

        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)

        # Right-pad to equal length. Causal attention means padding after the
        # last real token never affects earlier positions' outputs.
        padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
        batch_tensor = torch.tensor(padded, device=self.device, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(batch_tensor)  # [N, max_len, vocab_size]

            # Gather the logit vector at each sequence's last real token position
            last_indices = torch.tensor(
                [l - 1 for l in lengths], device=self.device, dtype=torch.long
            )
            # last_indices shape [N] -> [N, 1, 1] expanded to [N, 1, vocab_size]
            gather_idx = last_indices.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            last_logits = (
                logits.gather(1, gather_idx).squeeze(1) / temperature
            )  # [N, vocab_size]
            last_logits[:, 0] = float("-inf")  # suppress <|endoftext|>

            probs = F.softmax(last_logits, dim=-1)  # [N, vocab_size]
            top_probs, top_idx = torch.topk(probs, k=k, dim=-1)  # [N, k] each

            # Move to CPU once for all decoding
            top_probs_cpu = top_probs.cpu()
            top_idx_cpu = top_idx.cpu()

            batch_results = []
            for i in range(len(sequences)):
                results = []
                for j in range(k):
                    token_id = int(top_idx_cpu[i, j].item())
                    token_str = self.tokenizer.decode([token_id])
                    probability = float(top_probs_cpu[i, j].item())
                    results.append((token_id, token_str, probability))
                batch_results.append(results)

            return batch_results

    def _extract_top_k(self, cached, k):
        """Extract top-k results from a cached (sorted_probs, sorted_indices) pair."""
        sorted_probs, sorted_indices = cached
        results = []
        for j in range(min(k, len(sorted_probs))):
            token_id = int(sorted_indices[j].item())
            token_str = self.tokenizer.decode([token_id])
            probability = float(sorted_probs[j].item())
            results.append((token_id, token_str, probability))
        return results

    def get_top_k_cached_batch(
        self, sequences, path_keys, prompt, k=20, temperature=1.0
    ):
        """
        Like get_top_k_tokens_batch but caches the full sorted probability
        distribution for each node. Subsequent requests for the same node
        (even with a larger k) are served from cache with zero GPU work.

        Args:
            sequences: List of token lists (potentially different lengths)
            path_keys: List of cache key strings, one per sequence
            prompt: The prompt string (cache is invalidated if this changes)
            k: Number of top tokens to return per sequence
            temperature: Temperature for scaling logits

        Returns:
            List of lists of tuples: [[(token_id, token_str, probability), ...], ...]
        """
        if not sequences:
            return []

        # Invalidate cache if prompt changed
        if self._cache_prompt != prompt:
            self._probs_cache.clear()
            self._cache_prompt = prompt

        # Separate cached vs uncached
        all_results = [None] * len(sequences)
        uncached = []  # (original_index, sequence)

        for i, pk in enumerate(path_keys):
            if pk in self._probs_cache:
                all_results[i] = self._extract_top_k(self._probs_cache[pk], k)
            else:
                uncached.append((i, sequences[i]))

        if not uncached:
            return all_results

        # Batched forward pass for uncached sequences only
        uncached_indices, uncached_seqs = zip(*uncached)
        uncached_seqs = list(uncached_seqs)

        lengths = [len(seq) for seq in uncached_seqs]
        max_len = max(lengths)

        padded = [seq + [0] * (max_len - len(seq)) for seq in uncached_seqs]
        batch_tensor = torch.tensor(padded, device=self.device, dtype=torch.long)

        with torch.no_grad():
            logits = self.model(batch_tensor)  # [N, max_len, vocab_size]

            # Gather logits at each sequence's last real token position
            last_indices = torch.tensor(
                [l - 1 for l in lengths], device=self.device, dtype=torch.long
            )
            gather_idx = last_indices.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
            last_logits = (
                logits.gather(1, gather_idx).squeeze(1) / temperature
            )  # [N, vocab_size]
            last_logits[:, 0] = float("-inf")  # suppress <|endoftext|>

            probs = F.softmax(last_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

            sorted_probs_cpu = sorted_probs.cpu()
            sorted_indices_cpu = sorted_indices.cpu()

        # Cache full distributions and extract top-k
        for batch_i, orig_i in enumerate(uncached_indices):
            cached = (sorted_probs_cpu[batch_i], sorted_indices_cpu[batch_i])
            self._probs_cache[path_keys[orig_i]] = cached
            all_results[orig_i] = self._extract_top_k(cached, k)

        return all_results

    def build_beam_tree(
        self, prompt: str, k: int = 20, n: int = 100, raw: bool = False
    ):
        """
        Build a beam search tree showing top K tokens at each level for N levels.

        Args:
            prompt: Initial prompt to start from
            k: Number of top tokens to explore at each level
            n: Number of levels deep to explore
            raw: If True, use prompt as-is without wrapping in conversation tokens

        Returns:
            Tree structure with nodes containing token info and children
        """
        # Tokenize the initial prompt
        if not raw:
            prompt = "<|ConversationStart|><|Them|>" + prompt + "<|Me|>"
        tokens = self.tokenizer.encode(prompt)
        print(f"[build_beam_tree] Input prompt tokens: {tokens}")
        if len(tokens) > self.context_length:
            tokens = tokens[-self.context_length :]

        initial_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)

        # Build the tree recursively
        def build_node(tokens_tensor, depth, cumulative_prob):
            print(f"Depth: {depth}")
            if depth >= n:
                return None

            # Get top K tokens for current state
            top_k = self.get_top_k_tokens(tokens_tensor, k=k)

            children = []
            for token_id, token_str, probability in top_k:
                # Create new token sequence with this token appended
                new_token = torch.tensor([[token_id]], device=self.device)

                # Handle context length limit
                if tokens_tensor.size(1) >= self.context_length:
                    new_tokens = torch.cat([tokens_tensor[:, 1:], new_token], dim=1)
                else:
                    new_tokens = torch.cat([tokens_tensor, new_token], dim=1)

                # Recursively build children for this token
                child_node = {
                    "token_id": token_id,
                    "token_str": token_str,
                    "probability": probability,
                    "cumulative_prob": cumulative_prob * probability,
                    "depth": depth,
                    "children": build_node(
                        new_tokens, depth + 1, cumulative_prob * probability
                    ),
                }
                children.append(child_node)

            return children if children else None

        # Build tree starting from initial prompt
        tree = {
            "prompt": prompt,
            "children": build_node(initial_tokens, 0, 1.0),
        }

        return tree

    def generate_response_stream(
        self,
        conversation_history: str,
        user_message: str,
        auto_start: bool = False,
        auto_start_prompt: str = None,
    ):
        """
        Generate responses one at a time, yielding each as it's complete.

        Args:
            conversation_history: Previous conversation context
            user_message: The new message from the user
            auto_start: If True, MikeGPT starts the conversation (no user message)
            auto_start_prompt: The prompt to use for auto_start mode

        Yields:
            Tuples of (response_text, token_ids) where token_ids includes
            the leading <|Me|> token for tree navigation.
        """

        # 1. Build initial context and prime the model
        if auto_start and auto_start_prompt:
            # MikeGPT starts first - use the provided prompt directly
            context = auto_start_prompt
        else:
            # Normal mode - user sent a message
            context = conversation_history + f"<|Them|>{user_message}<|Me|>"

        self.prime(context)

        # Get <|Me|> token ID for the leading tag (part of context, not generated)
        me_token_id = self.tokenizer.encode("<|Me|>")[0]

        current_response = ""
        # First response starts with the context's <|Me|> for tree navigation
        response_token_ids = [me_token_id]
        max_tokens = 200  # safety cap
        generated_any = False

        for _ in range(max_tokens):
            token = self.next_token(top_p=0.5)
            token_id = int(self.current_tokens[0, -1].item())

            # 2. Handle special tokens
            if token in [
                "<|Me|>",
                "<|Them|>",
                "<|endoftext|>",
                "<|ConversationStart|>",
            ]:
                if current_response.strip():
                    yield (current_response.strip(), response_token_ids)
                    generated_any = True
                    current_response = ""
                    response_token_ids = []

                # stop generating once next speaker starts
                if token in ["<|Them|>", "<|endoftext|>", "<|ConversationStart|>"]:
                    break

                # <|Me|> separator — include in next response's token IDs
                response_token_ids = [token_id]

            elif token in [
                "<|Liked|>",
                "<|Laughed at|>",
                "<|Loved|>",
                "<|Disliked|>",
                "<|Questioned|>",
                "<|Emphasized|>",
            ]:
                # Reaction: include any accumulated token IDs (e.g. preceding <|Me|>)
                yield (token, response_token_ids + [token_id])
                generated_any = True
                response_token_ids = []

            else:
                current_response += token
                response_token_ids.append(token_id)

        if current_response.strip():
            yield (current_response.strip(), response_token_ids)
            generated_any = True

        # If no responses, retry (but not for auto_start to avoid infinite loops)
        if not generated_any:
            if auto_start:
                # For auto_start, yield a default greeting if nothing generated
                yield ("Hey", [me_token_id] + self.tokenizer.encode("Hey"))
            else:
                yield from self.generate_response_stream(
                    conversation_history, user_message
                )

    def do_training_step(
        self,
        prompt: list[int],
        responses: list[list[int]],
        rewards: list[float],
        num_steps: int = 1,
    ) -> dict:
        """
        Unified GRPO-based training for any group size.

        For pair mode: responses=[positive, negative], rewards=[1.0, -1.0]
        For group mode: responses=[resp1..resp8], rewards=[8,7,6,5,4,3,2,1] (from ranks)

        Returns dict with:
        - probability_changes: list[float] - percentage change per response
        - l2_diff: float - L2 norm of parameter changes
        - kl_divergence: float - mean KL(before || after) across response positions
        """
        from torch.nn.utils.rnn import pad_sequence
        from lm.training.reinforcement.log_probs import calculate_model_log_probs

        group_size = len(responses)

        # Prepare tensors for log prob calculation
        prompt_tensor = torch.tensor(
            prompt, dtype=torch.long, device=self.device
        ).expand(group_size, -1)
        prompt_lengths = torch.tensor([len(prompt)] * group_size, device=self.device)
        response_lengths = torch.tensor([len(r) for r in responses], device=self.device)
        response_tensor = pad_sequence(
            [torch.tensor(r, dtype=torch.long, device=self.device) for r in responses],
            batch_first=True,
            padding_value=0,
        )

        # Build full sequences (prompt + response) for KL divergence calculation
        full_sequences = [prompt + r for r in responses]
        max_len = max(len(s) for s in full_sequences)
        padded = [s + [0] * (max_len - len(s)) for s in full_sequences]
        full_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)

        # Snapshot parameters before training (for L2 diff)
        param_snapshot = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        # Calculate log probs before training
        # Note: calculate_model_log_probs returns (per_token_log_probs, mask) tuple
        with torch.no_grad():
            before_per_token_log_probs, _ = calculate_model_log_probs(
                self.model,
                prompt_tensor,
                prompt_lengths,
                response_tensor,
                response_lengths,
            )
            # Sum per-token log probs to get sequence-level
            before_log_probs = before_per_token_log_probs.sum(dim=-1)

            # Get full logits for KL divergence (before training)
            before_logits = self.model(full_tensor)  # [batch, seq_len, vocab_size]
            before_log_probs_full = F.log_softmax(before_logits, dim=-1)
            before_probs_full = torch.exp(before_log_probs_full)

        # Execute the GRPO training step
        # Note: model parameter removed, uses self.model internally
        self.trainable_model.do_grpo_step(
            prompt=prompt, responses=responses, rewards=rewards, num_steps=num_steps
        )

        # Calculate L2 diff of parameter changes
        l2_diff = 0.0
        for name, param in self.model.named_parameters():
            diff = param - param_snapshot[name]
            l2_diff += (diff**2).sum().item()
        l2_diff = l2_diff**0.5

        # Calculate log probs after training
        with torch.no_grad():
            after_per_token_log_probs, _ = calculate_model_log_probs(
                self.model,
                prompt_tensor,
                prompt_lengths,
                response_tensor,
                response_lengths,
            )
            # Sum per-token log probs to get sequence-level
            after_log_probs = after_per_token_log_probs.sum(dim=-1)

            # Get full logits for KL divergence (after training)
            after_logits = self.model(full_tensor)
            after_log_probs_full = F.log_softmax(after_logits, dim=-1)

            # KL(before || after) = sum_x P_before(x) * (log P_before(x) - log P_after(x))
            # Compute per position, then average over response positions only
            kl_per_position = (
                before_probs_full * (before_log_probs_full - after_log_probs_full)
            ).sum(dim=-1)

            # Only include response positions (after prompt), average across all
            prompt_len = len(prompt)
            kl_divergence = kl_per_position[:, prompt_len:].mean().item()

        # Convert to probability changes (as percentages)
        before_probs = torch.exp(before_log_probs) * 100
        after_probs = torch.exp(after_log_probs) * 100
        prob_changes = (after_probs - before_probs).tolist()

        return {
            "probability_changes": prob_changes,
            "l2_diff": l2_diff,
            "kl_divergence": kl_divergence,
        }
