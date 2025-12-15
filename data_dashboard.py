#!/usr/bin/env python3
import sqlite3
import os
import re
import string
import argparse
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Global variable to store database paths
DB_PATHS = []

# Self address to filter out conversations with one's self
SELF_ADDRESS = "PUT_YOUR_PHONE_NUMBER_HERE"

REACTION_PATTERNS = re.compile(
    r"^(Loved|Liked|Laughed at|Disliked|Emphasized) ", re.DOTALL
)
URL_PATTERN = re.compile(r"https?://\S+")

ME_PATTERN = "<|Me|>"
THEM_PATTERN = "<|Them|>"

PRINTABLE = set(string.printable)


def extract_text(row_text, row_attributed, rowid=None):
    """Get clean message text from iMessage DB row."""
    # First try the plain text field
    if row_text and row_text.strip():
        return row_text.strip()

    if not row_attributed:
        return None

    try:
        # Method 1: Try PyObjC approach
        from Foundation import NSData, NSKeyedUnarchiver

        # Convert bytes to NSData
        ns_data = NSData.dataWithBytes_length_(row_attributed, len(row_attributed))

        # Try the newer unarchiver API first
        try:
            attributed_string = (
                NSKeyedUnarchiver.unarchivedObjectOfClass_fromData_error_(
                    None, ns_data, None
                )[0]
            )
        except:
            # Fall back to older API
            unarchiver = NSKeyedUnarchiver.alloc().initForReadingWithData_(ns_data)
            if unarchiver is None:
                raise Exception(
                    "NSKeyedUnarchiver.alloc().initForReadingWithData_() returned None"
                )
            unarchiver.setRequiresSecureCoding_(False)
            attributed_string = unarchiver.decodeObjectForKey_("root")
            unarchiver.finishDecoding()

        if attributed_string is None:
            raise Exception("attributedBody extraction returned None")

        # Get the plain string from the attributed string
        text = attributed_string.string()

        if text:
            return text.strip()
        else:
            raise Exception("attributedBody.string() was empty")

    except Exception:
        # Method 2: Fallback to binary parsing with proper understanding of NSKeyedArchiver format
        try:
            text_bytes = row_attributed
            decoded = text_bytes.decode("utf-8", errors="ignore")

            import re

            # In NSKeyedArchiver binary plist format:
            # Strings have a length prefix like: +<char> where the char encodes length
            # The actual message text comes AFTER this prefix
            #
            # Pattern: NSString<binary markers>+<length_char><optional_junk>ACTUAL_TEXT
            #
            # Strategy: Find NSString markers, skip past any +<char> prefix, extract text

            # Find all potential text chunks after NSString markers
            parts = decoded.split("NSString")

            candidates = []
            for part in parts[1:]:  # Skip first part (before any NSString)
                # Skip short parts (likely just metadata)
                if len(part) < 10:
                    continue

                # Remove binary junk from the beginning (control chars)
                cleaned = re.sub(r"^[\x00-\x1f\x7f-\x9f]+", "", part)

                # Remove the length prefix pattern: +<single char><optional replacement char>
                # This handles: +: +g +# etc.
                cleaned = re.sub(r"^\+[\x00-\x7f][\ufffd\uFFFD]*", "", cleaned)

                # Also remove any other common binary prefix patterns
                cleaned = re.sub(r"^[\x80-\xff]+", "", cleaned)

                # Now extract the actual message text (stop at next control char or metadata marker)
                match = re.match(
                    r"^([^\x00-\x08\x0b-\x0c\x0e-\x1f]+?)(?:[\x00-\x1f]|__kIM|NSDictionary|NSNumber|NSMutable|bplist|\$)",
                    cleaned,
                    re.UNICODE,
                )

                if match:
                    text = match.group(1).strip()

                    # Validate it looks like real message text
                    if len(text) < 5:
                        continue

                    # Must have letters/numbers/emojis
                    if not re.search(r"[\w😀-🙏🌀-🗿]", text, re.UNICODE):
                        continue

                    # Reject if it's mostly metadata
                    if any(
                        marker in text
                        for marker in ["NSString", "NSData", "streamtyped", "__kIM"]
                    ):
                        continue

                    candidates.append(text)

            # Return the longest valid candidate
            if candidates:
                # Sort by length, prefer longer messages
                candidates.sort(key=lambda x: len(x), reverse=True)
                return candidates[0]

        except Exception:
            pass

        # If fallback parsing failed or returned nothing, skip this message
        print(
            f"WARNING: Could not extract attributedBody for rowid={rowid}, skipping message"
        )
        return None


def get_formatted_conversation(contact_id):
    """
    Fetch a single conversation with full message details and formatting.
    Returns the conversation exactly as it would appear in training data,
    including auto-added <|ConversationStart|> tokens.
    """
    conversation_messages = []
    seen_messages = (
        set()
    )  # Track (db_path, rowid, date, text) tuples to prevent duplicates
    debug_contact = (
        contact_id == "+17132618883"
    )  # Enable debug logging for this contact

    if debug_contact:
        print(f"\n=== DEBUG: Fetching conversation for {contact_id} ===")
        print(f"Number of databases: {len(DB_PATHS)}")

    for db_path in DB_PATHS:
        if debug_contact:
            print(f"\n--- Processing database: {db_path} ---")
        if not os.path.exists(db_path):
            continue

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all 1-on-1 chat IDs
            cursor.execute("""
                SELECT chat.ROWID
                FROM chat
                JOIN chat_handle_join chj ON chat.ROWID = chj.chat_id
                GROUP BY chat.ROWID
                HAVING COUNT(chj.handle_id) = 1
            """)
            chat_ids = [row[0] for row in cursor.fetchall()]

            for chat_id in chat_ids:
                cursor.execute(
                    """
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
                """,
                    (chat_id,),
                )
                messages = cursor.fetchall()

                # Check if this conversation is for the requested contact
                contact_match = None
                for _, _, sender, _, _, _ in messages:
                    if sender and sender != SELF_ADDRESS:
                        contact_match = sender
                        break

                if contact_match != contact_id:
                    continue

                if debug_contact:
                    print(
                        f"  Found matching chat_id: {chat_id} with {len(messages)} messages"
                    )

                # Skip convos with yourself
                if any(m[2] == SELF_ADDRESS for m in messages if m[2]):
                    continue

                # Apply filters and format messages
                for rowid, date, sender, text, attrib, is_from_me in messages:
                    text = extract_text(text, attrib, rowid)

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
                    text = text.replace("\ufffc", "").strip()
                    if not text or not re.search(r"\S", text):
                        continue

                    # Filter out messages with weird replacement characters (�)
                    if "\ufffd" in text or "�" in text:
                        continue

                    # Create unique key: (date, is_from_me, text)
                    # Same timestamp + sender + text = same message, regardless of DB/rowid
                    message_key = (date, is_from_me, text)
                    if message_key in seen_messages:
                        if debug_contact and "instrumentals" in text.lower():
                            print(
                                f"  SKIPPING DUPLICATE: db={os.path.basename(db_path)}, rowid={rowid}, date={date}, text={text[:50]}"
                            )
                        continue
                    seen_messages.add(message_key)

                    role = ME_PATTERN if is_from_me else THEM_PATTERN

                    if debug_contact and "instrumentals" in text.lower():
                        print(
                            f"  ADDING: rowid={rowid}, date={date}, from_me={is_from_me}, text={text[:50]}"
                        )

                    conversation_messages.append(
                        {"role": role, "text": text, "date": date}
                    )

            conn.close()

        except Exception as e:
            print(f"Error reading database {db_path}: {e}")
            continue

    # Sort all messages by date to ensure correct chronological order
    conversation_messages.sort(key=lambda m: m["date"])

    if debug_contact:
        print(f"\n=== BEFORE MERGING: {len(conversation_messages)} messages ===")
        for i, msg in enumerate(conversation_messages):
            if "instrumentals" in msg["text"].lower():
                print(f"  [{i}] {msg['role']}{msg['text'][:50]} (date: {msg['date']})")

    # Merge successive "Them:" messages only if within 1 hour
    from datetime import datetime

    merged = []
    for msg in conversation_messages:
        should_merge = False

        if merged and merged[-1]["role"] == msg["role"] and msg["role"] == THEM_PATTERN:
            # Check if messages are within 1 hour of each other
            try:
                prev_time = datetime.fromisoformat(merged[-1]["date"].replace(" ", "T"))
                curr_time = datetime.fromisoformat(msg["date"].replace(" ", "T"))
                time_diff = (curr_time - prev_time).total_seconds()

                # Merge if within 1 hour (3600 seconds)
                if time_diff <= 3600:
                    should_merge = True
            except:
                # If date parsing fails, don't merge
                pass

        if should_merge:
            merged[-1]["text"] += " " + msg["text"]
            # Update date to the later message
            merged[-1]["date"] = msg["date"]
        else:
            merged.append(msg)

    if debug_contact:
        print(f"\n=== AFTER MERGING: {len(merged)} messages ===")
        for i, msg in enumerate(merged):
            if "instrumentals" in msg["text"].lower():
                print(f"  [{i}] {msg['role']}{msg['text'][:50]} (date: {msg['date']})")

    # Remove 'rowid' field before returning (only keep role, text, date)
    for msg in merged:
        msg.pop("rowid", None)

    # Add auto-generated <|ConversationStart|> indices
    # This detects where ConversationStart tokens WOULD be auto-added during generation
    auto_start_indices = []
    auto_start_indices.append(0)  # Always at beginning

    # Detect 72+ hour breaks
    for idx in range(1, len(merged)):
        try:
            prev_date = datetime.fromisoformat(
                merged[idx - 1]["date"].replace(" ", "T")
            )
            curr_date = datetime.fromisoformat(merged[idx]["date"].replace(" ", "T"))
            time_diff = (curr_date - prev_date).total_seconds()

            # 72 hours = 259200 seconds
            if time_diff >= 259200:
                auto_start_indices.append(idx)
        except:
            pass

    # Add metadata about auto-start indices
    # Frontend can use this to render them differently or pre-populate the UI
    return {"messages": merged, "auto_conversation_starts": auto_start_indices}


def get_conversation_data():
    """Fetch conversation metadata for visualization from all databases."""
    all_conversations = []
    all_messages = []

    for db_path in DB_PATHS:
        if not os.path.exists(db_path):
            print(f"Warning: Database not found: {db_path}")
            continue

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Try to connect to Contacts database for names first
            contact_names = {}
            try:
                import glob

                contacts_dbs = glob.glob(
                    os.path.expanduser(
                        "~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb"
                    )
                )
                if contacts_dbs:
                    contacts_conn = sqlite3.connect(contacts_dbs[0])
                    contacts_cursor = contacts_conn.cursor()

                    # Get contact names from the Contacts database
                    contacts_cursor.execute("""
                        SELECT ZABCDRECORD.ZUNIQUE_ID,
                               ZABCDRECORD.ZFIRSTNAME,
                               ZABCDRECORD.ZLASTNAME
                        FROM ZABCDRECORD
                        WHERE ZABCDRECORD.ZFIRSTNAME IS NOT NULL OR ZABCDRECORD.ZLASTNAME IS NOT NULL
                    """)

                    for unique_id, first_name, last_name in contacts_cursor.fetchall():
                        full_name = " ".join(filter(None, [first_name, last_name]))
                        if full_name:
                            contact_names[unique_id] = full_name

                    contacts_conn.close()
                    print(f"   Loaded {len(contact_names)} contact names")
            except Exception as e:
                print(f"   Note: Could not load contact names: {e}")

            # Get all 1-on-1 conversations with message counts and time ranges
            # Filter out conversations with yourself
            cursor.execute(
                """
                SELECT
                    h.id as contact,
                    h.person_centric_id,
                    COUNT(m.ROWID) as message_count,
                    MIN(datetime(m.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch')) as first_message,
                    MAX(datetime(m.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch')) as last_message,
                    SUM(CASE WHEN m.is_from_me = 1 THEN 1 ELSE 0 END) as sent_count,
                    SUM(CASE WHEN m.is_from_me = 0 THEN 1 ELSE 0 END) as received_count
                FROM chat c
                JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
                JOIN handle h ON chj.handle_id = h.ROWID
                JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
                JOIN message m ON cmj.message_id = m.ROWID
                WHERE c.ROWID IN (
                    SELECT chat.ROWID
                    FROM chat
                    JOIN chat_handle_join chj ON chat.ROWID = chj.chat_id
                    GROUP BY chat.ROWID
                    HAVING COUNT(chj.handle_id) = 1
                )
                AND h.id != ?
                GROUP BY h.id
                HAVING message_count > 5 AND sent_count > 0
                ORDER BY message_count DESC
            """,
                (SELF_ADDRESS,),
            )

            conversations = []
            matched_count = 0
            for row in cursor.fetchall():
                contact, person_id, msg_count, first_msg, last_msg, sent, received = row

                # Try to get the contact name
                display_name = contact
                if person_id and person_id in contact_names:
                    display_name = f"{contact_names[person_id]} ({contact})"
                    matched_count += 1

                conversations.append(
                    {
                        "contact": display_name,
                        "messageCount": msg_count,
                        "firstMessage": first_msg,
                        "lastMessage": last_msg,
                        "sentCount": sent,
                        "receivedCount": received,
                    }
                )

            if contact_names:
                print(
                    f"   Matched {matched_count}/{len(conversations)} contacts with names"
                )

            # Also get all messages for timeline density
            cursor.execute("""
                SELECT
                    h.id as contact,
                    datetime(m.date/1000000000 + strftime('%s','2001-01-01'), 'unixepoch') as message_date,
                    m.is_from_me
                FROM chat c
                JOIN chat_handle_join chj ON c.ROWID = chj.chat_id
                JOIN handle h ON chj.handle_id = h.ROWID
                JOIN chat_message_join cmj ON c.ROWID = cmj.chat_id
                JOIN message m ON cmj.message_id = m.ROWID
                WHERE c.ROWID IN (
                    SELECT chat.ROWID
                    FROM chat
                    JOIN chat_handle_join chj ON chat.ROWID = chj.chat_id
                    GROUP BY chat.ROWID
                    HAVING COUNT(chj.handle_id) = 1
                )
                AND h.id IN (
                    SELECT h2.id
                    FROM chat c2
                    JOIN chat_handle_join chj2 ON c2.ROWID = chj2.chat_id
                    JOIN handle h2 ON chj2.handle_id = h2.ROWID
                    JOIN chat_message_join cmj2 ON c2.ROWID = cmj2.chat_id
                    JOIN message m2 ON cmj2.message_id = m2.ROWID
                    GROUP BY h2.id
                    HAVING COUNT(m2.ROWID) > 5
                )
                ORDER BY message_date ASC
            """)

            messages = []
            for row in cursor.fetchall():
                contact, msg_date, is_from_me = row
                messages.append(
                    {"contact": contact, "date": msg_date, "fromMe": bool(is_from_me)}
                )

            all_conversations.extend(conversations)
            all_messages.extend(messages)
            conn.close()

        except Exception as e:
            print(f"Error reading database {db_path}: {e}")
            continue

    # Merge conversations with the same contact
    contact_map = {}

    for conv in all_conversations:
        contact = conv["contact"]
        if contact not in contact_map:
            contact_map[contact] = conv
        else:
            # Merge with existing entry
            existing = contact_map[contact]
            existing["messageCount"] += conv["messageCount"]
            existing["sentCount"] += conv["sentCount"]
            existing["receivedCount"] += conv["receivedCount"]

            # Update first message if this one is earlier
            if conv["firstMessage"] and (
                not existing["firstMessage"]
                or conv["firstMessage"] < existing["firstMessage"]
            ):
                existing["firstMessage"] = conv["firstMessage"]

            # Update last message if this one is later
            if conv["lastMessage"] and (
                not existing["lastMessage"]
                or conv["lastMessage"] > existing["lastMessage"]
            ):
                existing["lastMessage"] = conv["lastMessage"]

    # Convert back to list and sort by message count
    merged_conversations = list(contact_map.values())
    merged_conversations.sort(key=lambda x: x["messageCount"], reverse=True)

    # Calculate total messages and add percentage to each conversation
    total_messages = sum(conv["messageCount"] for conv in merged_conversations)
    for conv in merged_conversations:
        conv["percentage"] = (
            (conv["messageCount"] / total_messages * 100) if total_messages > 0 else 0
        )

    return {"conversations": merged_conversations, "messages": all_messages}


@app.route("/")
def index():
    return render_template("visualizer.html")


@app.route("/api/data")
def data():
    return jsonify(get_conversation_data())


@app.route("/api/conversation/<path:contact_id>")
def conversation_detail(contact_id):
    """Get detailed formatted conversation for a specific contact."""
    result = get_formatted_conversation(contact_id)
    return jsonify(result)


@app.route("/api/generate-dataset", methods=["POST"])
def generate_dataset():
    """
    Generate training and validation data files based on selected conversations and custom tokens.
    Request body should contain:
    - selected_contacts: list of contact IDs to include in training
    - validation_contacts: list of contact IDs to include in validation
    - conversation_starts: dict mapping contact_id -> list of message indices where to insert <|ConversationStart|>

    Automatically inserts <|ConversationStart|> at:
    1. Beginning of each conversation
    2. After 72+ hour breaks in conversation
    """
    data = request.json
    selected_contacts = data.get("selected_contacts", [])
    validation_contacts = data.get("validation_contacts", [])
    conversation_starts = data.get("conversation_starts", {})

    # Create output directory if it doesn't exist
    output_dir = "data/text_data"
    os.makedirs(output_dir, exist_ok=True)

    train_output_file = os.path.join(output_dir, "imessages_dataset.txt")
    val_output_file = os.path.join(output_dir, "imessages_validation.txt")

    def write_conversations(contacts, output_file):
        """Helper to write conversations to a file."""
        with open(output_file, "w", encoding="utf-8") as f:
            for contact_id in contacts:
                result = get_formatted_conversation(contact_id)
                messages = result["messages"]
                auto_conversation_starts = result["auto_conversation_starts"]

                if not messages:
                    continue

                # Get manual indices where we should insert <|ConversationStart|>
                manual_start_indices = set(conversation_starts.get(contact_id, []))

                # Get auto-generated indices
                auto_start_indices = set(auto_conversation_starts)

                # Combine manual and automatic indices
                all_start_indices = manual_start_indices | auto_start_indices

                # Insert conversation start tokens
                final_messages = []
                for idx, msg in enumerate(messages):
                    # Insert conversation start token if needed
                    if idx in all_start_indices:
                        final_messages.append(
                            {"role": "", "text": "<|ConversationStart|>"}
                        )
                    final_messages.append(msg)

                # Write to file
                for msg in final_messages:
                    if msg["role"]:  # Normal message
                        f.write(f"{msg['role']}{msg['text']}")
                    else:  # Special token
                        f.write(msg["text"])

                f.write("<|endoftext|>")  # End of conversation marker

    # Write training data
    write_conversations(selected_contacts, train_output_file)

    # Write validation data if there are validation contacts
    if validation_contacts:
        write_conversations(validation_contacts, val_output_file)

    # Return success with file paths
    return jsonify(
        {
            "success": True,
            "training_file": train_output_file,
            "validation_file": val_output_file if validation_contacts else None,
            "training_count": len(selected_contacts),
            "validation_count": len(validation_contacts),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize iMessage conversations from one or more databases"
    )
    parser.add_argument(
        "--db",
        action="append",
        dest="databases",
        help="Path to chat.db file (can be specified multiple times)",
        default=[],
    )
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

    # Ensure templates directory exists
    os.makedirs("templates", exist_ok=True)
    print("🚀 Starting message visualizer...")
    print("📊 Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)
