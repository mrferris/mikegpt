#!/usr/bin/env python3
import json
import argparse

def hex_to_text(h):
    # h is something like "204c696e7573"
    try:
        return bytes.fromhex(h).decode("utf-8", errors="replace")
    except Exception:
        # If it's not valid hex, just return as-is
        return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab_file")
    args = ap.parse_args()

    with open(args.vocab_file, "r") as f:
        vocab = json.load(f)

    # If you want to keep it looking like JSON, wrap in braces optionally:
    # print("{")
    first = True
    for idx_str, hex_str in vocab.items():
        decoded = hex_to_text(hex_str)
        # Use json.dumps to correctly escape quotes, backslashes, etc.
        value_str = json.dumps(decoded)
        comma = ","  # if you want to paste back into a JSON object
        print(f'  "{idx_str}": {value_str}{comma}')
    # print("}")

if __name__ == "__main__":
    main()