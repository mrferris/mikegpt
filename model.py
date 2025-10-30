import sys
from pathlib import Path

# Add sibling repo to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "cs336/assignment1-basics/"))

from cs336_basics.model.transformer import TransformerLM
from cs336_basics.training.checkpointing import load_checkpoint
from cs336_basics.tokenization.bpe import Tokenizer
import torch
import torch.nn.functional as F
import random

class Model:
    def __init__(self, checkpoint_path):
        self.device = "cpu"
        self.context_length = 256
        d_model = 512
        vocab_size = 8192
        num_heads = 4
        num_layers = 4
        d_ff = 1344
        rope_theta = 10000

        self.model = TransformerLM(
            d_model=d_model,
            vocab_size=vocab_size,
            context_length=self.context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=self.device,
        ).to(self.device).eval()

        load_checkpoint(checkpoint_path, self.model, None)

        self.tokenizer = Tokenizer.from_files(
            "../cs336/assignment1-basics/cs336_basics/output/openwebtext_vocab.json",
            "../cs336/assignment1-basics/cs336_basics/output/openwebtext_merges.pkl"
        )

        self.current_tokens = None  # running token buffer on device

    def prime(self, prompt: str):
        """Tokenize prompt once and store on device."""
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.context_length:
            tokens = tokens[-self.context_length:]
        self.current_tokens = torch.tensor([tokens], device=self.device, dtype=torch.long)

    def next_token(self) -> str:
        """Fast next-token generation (no re-tokenization)."""
        if self.current_tokens is None:
            raise RuntimeError("Must call .prime(prompt) before .next_token()")

        with torch.no_grad():
            logits = self.model(self.current_tokens)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits, dim=-1)

        top_k = 3
        top_probs, top_idx = torch.topk(probs, k=top_k)
        chosen_id = int(top_idx[random.randint(0, top_k - 1)].item())

        # Append new token on GPU, cropping if needed
        new_token = torch.tensor([[chosen_id]], device=self.device)
        if self.current_tokens.size(1) >= self.context_length:
            self.current_tokens = torch.cat([self.current_tokens[:, 1:], new_token], dim=1)
        else:
            self.current_tokens = torch.cat([self.current_tokens, new_token], dim=1)

        return self.tokenizer.decode([chosen_id])
        

    def generate_response_stream(self, conversation_history: str, user_message: str):
        """
        Generate responses one at a time, yielding each as it's complete.

        Args:
            conversation_history: Previous conversation context
            user_message: The new message from the user

        Yields:
            Individual response messages as they're generated
        """

        # 1. Build initial context and prime the model
        context = conversation_history + f"<|Them|>{user_message}<|Me|>"
        self.prime(context)

        current_response = ""
        max_tokens = 200  # safety cap
        generated_any = False

        for _ in range(max_tokens):
            token = self.next_token()
            print(token)

            # 2. Handle special tokens
            if token in ["<|Me|>", "<|Them|>", "<|endoftext|>"]:
                if current_response.strip():
                    yield current_response.strip()
                    generated_any = True
                    current_response = ""

                # stop generating once next speaker starts
                if token in ["<|Them|>", "<|endoftext|>"]:
                    break

            elif token in ["<|Liked|>", "<|Laughed|>", "<|Loved|>", "<|Disliked|>", "<|Questioned|>", "<|Emphasized|>"]:
                # single reaction token as message
                yield token
                generated_any = True
                break

            else:
                current_response += token

        if current_response.strip():
            yield current_response.strip()
            generated_any = True

        # If no responses, retry
        if not generated_any:
            yield from self.generate_response_stream(conversation_history, user_message)
