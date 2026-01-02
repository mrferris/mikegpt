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

        with open("vocab/mikegpt_vocab_8192.json") as f:
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
            "vocab/mikegpt_vocab_8192.json",
            "vocab/mikegpt_merges_8192.pkl",
            special_tokens=special_tokens,
        )

        self.current_tokens = None  # running token buffer on device

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
            probs = F.softmax(last_logits, dim=-1)

            top_probs, top_idx = torch.topk(probs, k=k)

            results = []
            for prob, idx in zip(top_probs, top_idx):
                token_id = int(idx.item())
                token_str = self.tokenizer.decode([token_id])
                probability = float(prob.item())
                results.append((token_id, token_str, probability))

            return results

    def build_beam_tree(self, prompt: str, k: int = 20, n: int = 100):
        """
        Build a beam search tree showing top K tokens at each level for N levels.

        Args:
            prompt: Initial prompt to start from
            k: Number of top tokens to explore at each level
            n: Number of levels deep to explore

        Returns:
            Tree structure with nodes containing token info and children
        """
        # Tokenize the initial prompt
        prompt = "<|ConversationStart|><|Them|>" + prompt + "<|Me|>"
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.context_length:
            tokens = tokens[-self.context_length :]

        initial_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)

        # Build the tree recursively
        def build_node(tokens_tensor, depth, cumulative_prob):
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
            Individual response messages as they're generated
        """

        # 1. Build initial context and prime the model
        if auto_start and auto_start_prompt:
            # MikeGPT starts first - use the provided prompt directly
            context = auto_start_prompt
        else:
            # Normal mode - user sent a message
            context = conversation_history + f"<|Them|>{user_message}<|Me|>"

        self.prime(context)
        current_response = ""
        max_tokens = 200  # safety cap
        generated_any = False

        for _ in range(max_tokens):
            token = self.next_token(use_top_k=True, top_k=3)

            # 2. Handle special tokens
            if token in [
                "<|Me|>",
                "<|Them|>",
                "<|endoftext|>",
                "<|ConversationStart|>",
            ]:
                if current_response.strip():
                    yield current_response.strip()
                    generated_any = True
                    current_response = ""

                # stop generating once next speaker starts
                if token in ["<|Them|>", "<|endoftext|>", "<|ConversationStart|>"]:
                    break

            elif token in [
                "<|Liked|>",
                "<|Laughed at|>",
                "<|Loved|>",
                "<|Disliked|>",
                "<|Questioned|>",
                "<|Emphasized|>",
            ]:
                # single reaction token as message
                yield token
                generated_any = True

            else:
                current_response += token

        if current_response.strip():
            yield current_response.strip()
            generated_any = True

        # If no responses, retry (but not for auto_start to avoid infinite loops)
        if not generated_any:
            if auto_start:
                # For auto_start, yield a default greeting if nothing generated
                yield "Hey"
            else:
                yield from self.generate_response_stream(
                    conversation_history, user_message
                )

    def do_dpo_step(
        self, prompt: list[int], positive: list[int], negative: list[int]
    ) -> tuple[float]:
        """
        Returns a tuple containing the change in probablities for the
        positive and negative responses, respectively.
        """
        return self.trainable_model.do_simpo_step(prompt, positive, negative)
