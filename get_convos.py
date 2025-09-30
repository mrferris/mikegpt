#!/usr/bin/env python3
import sqlite3
import os
import re

DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")
OUTPUT_FILE = "imessages_dataset.txt"

# Patterns
REACTION_PATTERNS = re.compile(r'^(Loved|Liked|Laughed at|Disliked) ".+"$', re.DOTALL)
URL_PATTERN = re.compile(r'https?://\S+')

# Your Apple ID (to filter out self-convos)
SELF_ADDRESS = "+17163594066"


def fetch_conversations():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Only 1-on-1 chats (not group chats)
    cursor.execute("""
        SELECT chat.ROWID
        FROM chat
        JOIN chat_handle_join chj ON chat.ROWID = chj.chat_id
        GROUP BY chat.ROWID
        HAVING COUNT(chj.handle_id) = 1
    """)
    chat_ids = [row[0] for row in cursor.fetchall()]

    all_convos = []
    for chat_id in chat_ids:
        cursor.execute("""
            SELECT
                message.ROWID,
                datetime(message.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch') as message_date,
                handle.id as sender,
                message.text,
                message.is_from_me
            FROM chat_message_join cmj
            JOIN message ON cmj.message_id = message.ROWID
            LEFT JOIN handle ON message.handle_id = handle.ROWID
            WHERE cmj.chat_id = ?
            ORDER BY message_date ASC
        """, (chat_id,))
        messages = cursor.fetchall()

        # Skip convos with yourself
        if any(m[2] == SELF_ADDRESS for m in messages if m[2]):
            continue

        # Apply filters
        filtered = []
        for rowid, date, sender, text, is_from_me in messages:
            if not text or not re.search(r"\S", text):
                continue
            if URL_PATTERN.search(text):
                continue
            if not is_from_me and REACTION_PATTERNS.match(text.strip()):
                continue

            # Remove object replacement characters and other special chars
            text = text.replace('\ufffc', '').strip()
            if not text or not re.search(r"\S", text):
                continue

            role = "Me:" if is_from_me else "Them:"
            filtered.append((role, text))

        if not filtered:
            continue

        # Keep only convos where you appear at least once
        roles = [r for r, _ in filtered]
        if "Me:" not in roles:
            continue

        all_convos.append(filtered)

    conn.close()
    return all_convos


def write_dataset(conversations):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        convo_index = 0
        for convo in conversations:
            cleaned = [(role, text) for role, text in convo if text and re.search(r"\S", text)]
            if not cleaned:
                continue

            convo_index += 1
            f.write(f"====== Conversation {convo_index} ======\n")

            # Filter out whitespace-only messages before merging
            non_whitespace = [(role, text) for role, text in cleaned if re.search(r"\S", text)]

            # Concatenate successive "Them:" messages only
            merged = []
            for role, text in non_whitespace:
                if merged and merged[-1][0] == role and role == "Them:":
                    merged[-1] = (role, merged[-1][1] + " " + text)
                else:
                    merged.append((role, text))

            for role, text in merged:
                f.write(f"{role} {text}<|endoftext|>\n")
            f.write("\n")  # blank line between conversations


if __name__ == "__main__":
    convos = fetch_conversations()
    print(f"✅ Found {len(convos)} conversations")
    write_dataset(convos)
    print(f"💾 Saved dataset to {OUTPUT_FILE}")
