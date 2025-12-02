#!/usr/bin/env python3
import sqlite3
import os
import re
import string
import argparse

OUTPUT_FILE = "imessages_dataset.txt"

# Global variable to store database paths
DB_PATHS = []

# Patterns
REACTION_PATTERNS = re.compile(r'^(Loved|Liked|Laughed at|Disliked|Emphasized) ', re.DOTALL)
URL_PATTERN = re.compile(r'https?://\S+')

SELF_ADDRESS = "REPLACE"

ME_PATTERN = "<|Me|>"
THEM_PATTERN = "<|Them|>"

PRINTABLE = set(string.printable)  # ascii letters, digits, punctuation, etc.

def extract_text(row_text, row_attributed):
    """Get clean message text from iMessage DB row."""
    # First try the plain text field
    if row_text and row_text.strip():
        return row_text.strip()
    
    if not row_attributed:
        return None
    
    try:
        # Use PyObjC to properly deserialize the NSAttributedString
        from Foundation import NSData, NSKeyedUnarchiver
        
        # Convert bytes to NSData
        ns_data = NSData.dataWithBytes_length_(row_attributed, len(row_attributed))
        
        # Unarchive the attributed string
        unarchiver = NSKeyedUnarchiver.alloc().initForReadingWithData_(ns_data)
        unarchiver.setRequiresSecureCoding_(False)
        attributed_string = unarchiver.decodeObjectForKey_("root")
        unarchiver.finishDecoding()
        
        if attributed_string is None:
            return None
        
        # Get the plain string from the attributed string
        text = attributed_string.string()
        
        if text:
            return text.strip()
        
        return None
        
    except Exception as e:
        # If PyObjC fails, print debug info and return None
        print(f"⚠️  Failed to extract attributed text: {e}")
        print(f"   Raw bytes preview: {row_attributed[:100] if row_attributed else 'None'}")
        return None

def fetch_conversations():
    """Fetch conversations from all databases and combine them."""
    all_convos = []

    for db_path in DB_PATHS:
        if not os.path.exists(db_path):
            print(f"Warning: Database not found: {db_path}")
            continue

        try:
            conn = sqlite3.connect(db_path)
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

            for chat_id in chat_ids:
                cursor.execute("""
                    SELECT
                        message.ROWID,
                        datetime(message.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch') as message_date,
                        handle.id as sender,
                        message.text,
                        message.attributedBody,
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
                for rowid, date, sender, text, attrib, is_from_me in messages:

                    text = extract_text(text, attrib)

                    if not text or not re.search(r"\S", text):
                        continue
                    if URL_PATTERN.search(text):
                        continue

                    # Handle reaction messages
                    reaction_match = REACTION_PATTERNS.match(text.strip())
                    if reaction_match:
                        if is_from_me:
                            # Convert my reactions to tokens
                            reaction_type = reaction_match.group(1)
                            text = f"<|{reaction_type}|>"
                        else:
                            # Skip their reactions
                            continue

                    # Remove object replacement characters and other special chars
                    text = text.replace('\ufffc', '').strip()
                    if not text or not re.search(r"\S", text):
                        continue

                    role = ME_PATTERN if is_from_me else THEM_PATTERN
                    filtered.append((role, text))

                if not filtered:
                    continue

                # Keep only convos where you appear at least once
                roles = [r for r, _ in filtered]
                if ME_PATTERN not in roles:
                    continue

                all_convos.append(filtered)

            conn.close()

        except Exception as e:
            print(f"Error reading database {db_path}: {e}")
            continue

    return all_convos


def write_dataset(conversations):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        convo_index = 0
        for convo in conversations:
            cleaned = [(role, text) for role, text in convo if text and re.search(r"\S", text)]
            if not cleaned:
                continue

            convo_index += 1
            # Filter out whitespace-only messages before merging
            non_whitespace = [(role, text) for role, text in cleaned if re.search(r"\S", text)]

            # Concatenate successive "Them:" messages only
            merged = []
            for role, text in non_whitespace:
                if merged and merged[-1][0] == role and role == THEM_PATTERN:
                    merged[-1] = (role, merged[-1][1] + " " + text)
                else:
                    merged.append((role, text))

            for role, text in merged:
                f.write(f"{role}{text}")
            f.write("<|endoftext|>")  # blank line between conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract conversations from one or more iMessage databases')
    parser.add_argument('--db', action='append', dest='databases',
                        help='Path to chat.db file (can be specified multiple times)',
                        default=[])
    args = parser.parse_args()

    # If no databases specified, use default
    if not args.databases:
        default_db = os.path.expanduser("~/Library/Messages/chat.db")
        DB_PATHS = [default_db]
        print(f"📂 Using default database: {default_db}")
    else:
        DB_PATHS = [os.path.expanduser(db) for db in args.databases]
        print(f"📂 Loading {len(DB_PATHS)} database(s):")
        for db in DB_PATHS:
            print(f"   - {db}")

    convos = fetch_conversations()
    print(f"✅ Found {len(convos)} conversations")
    write_dataset(convos)
    print(f"💾 Saved dataset to {OUTPUT_FILE}")
