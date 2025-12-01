#!/usr/bin/env python3
import sqlite3
import os
import argparse
from flask import Flask, render_template, jsonify
from datetime import datetime
import json

app = Flask(__name__)

# Global variable to store database paths
DB_PATHS = []

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

            # Get all 1-on-1 conversations with message counts and time ranges
            cursor.execute("""
                SELECT
                    h.id as contact,
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
                GROUP BY h.id
                HAVING message_count > 5
                ORDER BY message_count DESC
            """)

            conversations = []
            for row in cursor.fetchall():
                contact, msg_count, first_msg, last_msg, sent, received = row
                conversations.append({
                    'contact': contact,
                    'messageCount': msg_count,
                    'firstMessage': first_msg,
                    'lastMessage': last_msg,
                    'sentCount': sent,
                    'receivedCount': received
                })

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
                messages.append({
                    'contact': contact,
                    'date': msg_date,
                    'fromMe': bool(is_from_me)
                })

            all_conversations.extend(conversations)
            all_messages.extend(messages)
            conn.close()

        except Exception as e:
            print(f"Error reading database {db_path}: {e}")
            continue

    # Merge conversations with the same contact
    from datetime import datetime as dt
    contact_map = {}

    for conv in all_conversations:
        contact = conv['contact']
        if contact not in contact_map:
            contact_map[contact] = conv
        else:
            # Merge with existing entry
            existing = contact_map[contact]
            existing['messageCount'] += conv['messageCount']
            existing['sentCount'] += conv['sentCount']
            existing['receivedCount'] += conv['receivedCount']

            # Update first message if this one is earlier
            if conv['firstMessage'] and (not existing['firstMessage'] or conv['firstMessage'] < existing['firstMessage']):
                existing['firstMessage'] = conv['firstMessage']

            # Update last message if this one is later
            if conv['lastMessage'] and (not existing['lastMessage'] or conv['lastMessage'] > existing['lastMessage']):
                existing['lastMessage'] = conv['lastMessage']

    # Convert back to list and sort by message count
    merged_conversations = list(contact_map.values())
    merged_conversations.sort(key=lambda x: x['messageCount'], reverse=True)

    return {'conversations': merged_conversations, 'messages': all_messages}

@app.route('/')
def index():
    return render_template('visualizer.html')

@app.route('/api/data')
def data():
    return jsonify(get_conversation_data())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize iMessage conversations from one or more databases')
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

    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    print("🚀 Starting message visualizer...")
    print("📊 Open http://localhost:5001 in your browser")
    app.run(debug=True, port=5001)
