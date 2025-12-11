#!/usr/bin/env python3
"""
Tokenization utilities for MikeGPT.

Trains a BPE tokenizer with special tokens including all emojis from the dataset.
"""

import sys
from pathlib import Path
import re
from collections import Counter
import os

# Add artisinal-lm to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "artisinal-lm"))

from lm.tokenization.bpe import train_bpe, Tokenizer


def extract_emojis_from_file(input_path: str) -> list[str]:
    """
    Extract all unique emojis from a text file.

    Returns a sorted list of unique emoji strings, including both full sequences
    and their base components (matching the format in emojis.py).
    """
    # Read the file
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pattern that matches proper emoji sequences (base + modifiers, but NOT complex ZWJ sequences)
    emoji_pattern = re.compile(
        "(?:"
        # Flags (pairs of regional indicator letters)
        "[\U0001f1e0-\U0001f1ff]{2}|"
        # Emoji with optional skin tone and variation selector
        # But NOT followed by ZWJ (to avoid complex sequences)
        "[\U0001f600-\U0001f64f]"  # Emoticons
        "(?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?(?!\u200d)|"  # Optional skin tone + variation selector, not followed by ZWJ
        "[\U0001f300-\U0001f5ff]"  # Symbols & pictographs
        "(?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?(?!\u200d)|"
        "[\U0001f680-\U0001f6ff]"  # Transport & map symbols
        "(?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?(?!\u200d)|"
        "[\U0001f900-\U0001f9ff]"  # Supplemental symbols
        "(?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?(?!\u200d)|"
        "[\U0001fa00-\U0001faff]"  # Extended pictographs
        "(?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?(?!\u200d)|"
        # Miscellaneous symbols with optional skin tone and variation selector
        "[\U00002600-\U000027bf](?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?|"
        # Other special symbols with optional skin tone and variation selector
        "[\u2300-\u23ff](?:[\U0001f3fb-\U0001f3ff])?[\ufe0f]?|"
        "[\u2640-\u2642][\ufe0f]?"
        ")",
        flags=re.UNICODE,
    )

    # Find all emoji sequences
    emoji_sequences = emoji_pattern.findall(text)

    # Count emoji occurrences
    emoji_counts = Counter(emoji_sequences)

    # Sort by frequency
    unique_emojis = sorted(
        emoji_counts.keys(), key=lambda x: emoji_counts[x], reverse=True
    )

    return unique_emojis


def generate_vocab(
    input_path: str,
    vocab_size: int,
    output_vocab_path: str,
    output_merges_path: str,
    include_emojis: bool = True,
) -> Tokenizer:
    """
    Generate a BPE vocabulary from a text file with special tokens.

    Args:
        input_path: Path to the training text file
        vocab_size: Total vocabulary size (including special tokens)
        output_vocab_path: Path to save the vocabulary JSON file
        output_merges_path: Path to save the merges pickle file
        include_emojis: Whether to automatically extract and include emojis as special tokens

    Returns:
        The trained tokenizer

    The special tokens are added in this order:
    1. Core special tokens: <|endoftext|>, <|Me|>, <|Them|>, <|ConversationStart|>
    2. Reaction tokens: <|Loved|>, <|Liked|>, <|Laughed at|>, <|Disliked|>, <|Questioned|>, <|Emphasized|>
    3. Emojis: All unique emojis found in the input file (if include_emojis=True)
    4. Byte tokens: All 256 byte values
    5. BPE merges: Learned byte-pair merges up to vocab_size
    """
    # Define core special tokens
    core_special_tokens = [
        "<|endoftext|>",
        "<|Me|>",
        "<|Them|>",
        "<|ConversationStart|>",
    ]

    # Define reaction special tokens
    reaction_tokens = [
        "<|Loved|>",
        "<|Liked|>",
        "<|Laughed at|>",
        "<|Disliked|>",
        "<|Questioned|>",
        "<|Emphasized|>",
    ]

    # Combine special tokens
    special_tokens = core_special_tokens + reaction_tokens

    # Extract and add emojis if requested
    if include_emojis:
        print(f"Extracting emojis from {input_path}...")
        emojis = extract_emojis_from_file(input_path)
        print(f"Found {len(emojis)} unique emojis")
        print(emojis)
        special_tokens.extend(emojis)

    print(f"Total special tokens: {len(special_tokens)}")
    print(f"Training BPE with vocab_size={vocab_size}...")

    # Train BPE
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    print(f"Trained vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    # Create tokenizer and save
    print(f"Saving vocabulary to {output_vocab_path}")
    print(f"Saving merges to {output_merges_path}")

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(output_vocab_path, output_merges_path)

    print("Done!")
    return tokenizer


def encode_datasets(
    tokenizer: Tokenizer,
    train_path: str,
    validation_path: str | None = None,
    output_dir: str = "data/encoded",
) -> None:
    """
    Encode training and validation datasets to numpy arrays.

    Args:
        tokenizer: The trained tokenizer
        train_path: Path to training text file
        validation_path: Path to validation text file (optional)
        output_dir: Directory to save encoded datasets
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Encode training dataset
    train_output = os.path.join(output_dir, "train.npy")
    print(f"\nEncoding training dataset: {train_path} -> {train_output}")
    tokenizer.encode_dataset_to_numpy(train_path, train_output)

    # Encode validation dataset if provided
    if validation_path and os.path.exists(validation_path):
        val_output = os.path.join(output_dir, "val.npy")
        print(f"\nEncoding validation dataset: {validation_path} -> {val_output}")
        tokenizer.encode_dataset_to_numpy(validation_path, val_output)
    elif validation_path:
        print(f"Warning: Validation file not found: {validation_path}")

    print("\nEncoding complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate BPE vocabulary for MikeGPT and encode datasets"
    )
    parser.add_argument("train_path", type=str, help="Path to the training text file")
    parser.add_argument(
        "--validation-path",
        type=str,
        help="Path to the validation text file (optional)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Total vocabulary size (including special tokens)",
    )
    parser.add_argument(
        "--output-vocab",
        type=str,
        required=True,
        help="Path to save vocabulary JSON file",
    )
    parser.add_argument(
        "--output-merges",
        type=str,
        required=True,
        help="Path to save merges pickle file",
    )
    parser.add_argument(
        "--no-emojis",
        action="store_true",
        help="Don't include emojis as special tokens",
    )
    parser.add_argument(
        "--no-encode",
        action="store_true",
        help="Don't encode datasets to numpy arrays (only train vocab)",
    )
    parser.add_argument(
        "--encoded-output-dir",
        type=str,
        default="data/encoded",
        help="Directory to save encoded numpy arrays (default: data/encoded)",
    )

    args = parser.parse_args()

    tokenizer = generate_vocab(
        input_path=args.train_path,
        vocab_size=args.vocab_size,
        output_vocab_path=args.output_vocab,
        output_merges_path=args.output_merges,
        include_emojis=not args.no_emojis,
    )

    if not args.no_encode:
        encode_datasets(
            tokenizer=tokenizer,
            train_path=args.train_path,
            validation_path=args.validation_path,
            output_dir=args.encoded_output_dir,
        )
