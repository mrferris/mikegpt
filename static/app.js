// MikeGPT Frontend

const API_URL = window.location.origin;  // Use same origin as the page
const SESSION_ID = 'session_' + Date.now();

let conversationHistory = '';
let lastUserMessageElement = null;
let messageHistorySnapshots = [];
// Running history within the current response cycle, mirrors backend's new_history
let cycleHistory = '';
// Token ID tracking for tree navigation
let cycleTokenIds = [];           // Running token IDs for current generation cycle
let messageTokenIdSnapshots = []; // Parallel to messageHistorySnapshots

// Auto-start mode: if true, MikeGPT sends the first message
let mikeStartsFirst = true;

// TEMP: force a Loved react on the first user message for testing
let hasForceReacted = false;

// Update time in status bar
function updateTime() {
    const now = new Date();
    const hours = now.getHours();
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const timeStr = `${hours}:${minutes}`;
    document.getElementById('time').textContent = timeStr;
}

updateTime();
setInterval(updateTime, 60000);

// Reaction mapping
const REACTIONS = {
    '<|Liked|>': '👍',
    '<|Laughed at|>': '😆',
    '<|Loved|>': '🩷',
    '<|Disliked|>': '👎',
    '<|Questioned|>': '❓',
    '<|Emphasized|>': '‼️',
};

// Add message to UI
function addMessage(text, isUser, tokenIds) {
    const messagesContainer = document.getElementById('messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    bubbleDiv.textContent = text;

    messageDiv.appendChild(bubbleDiv);
    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Store reference if it's a user message
    if (isUser) {
        lastUserMessageElement = messageDiv;
    } else {
        // Track the history state before this bot message (use cycleHistory which
        // mirrors the backend's running history, including prior messages in this cycle)
        const snapshotIndex = messageHistorySnapshots.length;
        messageHistorySnapshots.push(cycleHistory);
        messageTokenIdSnapshots.push([...cycleTokenIds]);
        messageDiv.dataset.snapshotIndex = snapshotIndex;
        messageDiv.dataset.tokenIds = JSON.stringify(tokenIds || []);

        // Click handler: open MikeRL drive to mark this response as negative
        bubbleDiv.style.cursor = 'pointer';
        bubbleDiv.addEventListener('click', () => {
            const historySnapshot = messageHistorySnapshots[messageDiv.dataset.snapshotIndex];
            const msgTokenIds = JSON.parse(messageDiv.dataset.tokenIds || '[]');
            const msgText = text;  // The actual message text

            // historySnapshot ends with <|Me|> - the prompt for this response
            // Skip the first token ID (the <|Me|> separator) since it's part of the prompt
            const responseTokenIds = msgTokenIds.slice(1);

            window.open('/drive?prompt=' + encodeURIComponent(historySnapshot) +
                '&token_ids=' + encodeURIComponent(responseTokenIds.join(',')) +
                '&bad_text=' + encodeURIComponent(msgText), '_blank');
        });
    }

    return messageDiv;
}

// Add typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';

    typingDiv.innerHTML = `
        <div class="typing-bubble">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;

    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Remove typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Add reaction to last user message
function addReaction(reactionToken) {
    if (!lastUserMessageElement) return;

    // Remove existing reaction if any
    const existingReaction = lastUserMessageElement.querySelector('.reaction');
    if (existingReaction) {
        existingReaction.remove();
    }

    const emoji = REACTIONS[reactionToken];
    if (!emoji) return;

    const reactionDiv = document.createElement('div');
    reactionDiv.className = 'reaction';
    reactionDiv.textContent = emoji;

    const bubble = lastUserMessageElement.querySelector('.message-bubble');
    bubble.style.position = 'relative';
    bubble.appendChild(reactionDiv);

    // Push message down so the reaction doesn't overlap the previous message
    lastUserMessageElement.style.marginTop = '16px';
}

// Send message to backend with streaming
async function sendMessage() {
    const input = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const message = input.value.trim();

    if (!message) return;

    // Disable input
    input.disabled = true;
    sendButton.disabled = true;

    // Add user message to UI
    addMessage(message, true);

    // Clear input
    input.value = '';

    // Show typing indicator
    showTypingIndicator();

    // Build running history for this cycle, mirroring backend logic
    if (!conversationHistory) {
        cycleHistory = '<|ConversationStart|><|Them|>' + message;
    } else {
        cycleHistory = conversationHistory + '<|Them|>' + message;
    }
    cycleTokenIds = [];  // Reset token IDs for new cycle

    try {
        // Call API with EventSource for streaming
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: SESSION_ID,
                history: conversationHistory
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let typingStartTime = Date.now();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete messages (SSE format: "data: {...}\n\n")
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete message in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6);
                    const data = JSON.parse(jsonStr);

                    if (data.error) {
                        hideTypingIndicator();
                        addMessage(`Error: ${data.error}`, false);
                        break;
                    }

                    if (data.response) {
                        // Ensure typing indicator shows for at least 650ms
                        const elapsed = Date.now() - typingStartTime;
                        if (elapsed < 650) {
                            await new Promise(resolve => setTimeout(resolve, 650 - elapsed));
                        }

                        hideTypingIndicator();

                        // Check if it's a reaction
                        if (data.response.startsWith('<|') && data.response.endsWith('|>')) {
                            // Update cycle history with reaction, then apply
                            cycleHistory += data.response;
                            cycleTokenIds = cycleTokenIds.concat(data.token_ids || []);
                            addReaction(data.response);
                        } else {
                            // Snapshot happens inside addMessage; update cycleHistory
                            // to include <|Me|> BEFORE the snapshot so it captures the
                            // prompt the model saw for this response
                            cycleHistory += '<|Me|>';
                            // Token IDs from backend already include leading <|Me|>
                            addMessage(data.response, false, data.token_ids);
                            // After snapshot, append the response text for next message's snapshot
                            cycleHistory += data.response;
                            cycleTokenIds = cycleTokenIds.concat(data.token_ids || []);

                            // TEMP: force Loved react on first user message for testing
                            if (!hasForceReacted && lastUserMessageElement) {
                                hasForceReacted = true;
                                addReaction('<|Loved|>');
                            }
                        }

                        // Show typing indicator again for next response
                        showTypingIndicator();
                        typingStartTime = Date.now();
                    }

                    if (data.done) {
                        conversationHistory = data.history || conversationHistory;
                    }
                }
            }
        }

    } catch (error) {
        hideTypingIndicator();
        addMessage(`Error: ${error.message}`, false);
    } finally {
        hideTypingIndicator();
        // Re-enable input
        input.disabled = false;
        sendButton.disabled = false;
        input.focus();
    }
}

// Event listeners
document.getElementById('send-button').addEventListener('click', sendMessage);

document.getElementById('message-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Focus input on load
document.getElementById('message-input').focus();

// Auto-start: MikeGPT sends the first message
async function autoStartConversation() {
    const input = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    // Disable input while generating
    input.disabled = true;
    sendButton.disabled = true;

    // Show typing indicator immediately
    showTypingIndicator();

    // Set initial history to match the prompt the model sees for auto-start
    conversationHistory = '<|ConversationStart|><|Me|>';
    cycleHistory = '<|ConversationStart|><|Me|>';
    cycleTokenIds = [];  // Reset token IDs for new cycle

    try {
        // Call API with empty message - the backend will use <|ConversationStart|><|Me|>
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: '',  // Empty message signals auto-start
                session_id: SESSION_ID,
                history: '',
                auto_start: true  // Flag to tell backend MikeGPT starts first
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let typingStartTime = Date.now();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete messages (SSE format: "data: {...}\n\n")
            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const jsonStr = line.slice(6);
                    const data = JSON.parse(jsonStr);

                    if (data.error) {
                        hideTypingIndicator();
                        addMessage(`Error: ${data.error}`, false);
                        break;
                    }

                    if (data.response) {
                        // Ensure typing indicator shows for at least 650ms
                        const elapsed = Date.now() - typingStartTime;
                        if (elapsed < 650) {
                            await new Promise(resolve => setTimeout(resolve, 650 - elapsed));
                        }

                        hideTypingIndicator();

                        // Check if it's a reaction
                        if (data.response.startsWith('<|') && data.response.endsWith('|>')) {
                            // Add to context (so model stays in react-friendly tokenspace)
                            // but only display if there's a user message to attach to
                            cycleHistory += data.response;
                            cycleTokenIds = cycleTokenIds.concat(data.token_ids || []);
                            addReaction(data.response);
                        } else {
                            // For auto-start, first response doesn't add <|Me|>,
                            // subsequent ones do (mirrors backend logic)
                            if (cycleHistory !== '<|ConversationStart|><|Me|>') {
                                cycleHistory += '<|Me|>';
                            }
                            // Token IDs from backend already include leading <|Me|>
                            addMessage(data.response, false, data.token_ids);
                            cycleHistory += data.response;
                            cycleTokenIds = cycleTokenIds.concat(data.token_ids || []);
                        }

                        // Show typing indicator again for next response
                        showTypingIndicator();
                        typingStartTime = Date.now();
                    }

                    if (data.done) {
                        conversationHistory = data.history || conversationHistory;
                    }
                }
            }
        }

    } catch (error) {
        hideTypingIndicator();
        addMessage(`Error: ${error.message}`, false);
    } finally {
        hideTypingIndicator();
        // Re-enable input
        input.disabled = false;
        sendButton.disabled = false;
        input.focus();
    }
}

// Initialize: auto-start if enabled
if (mikeStartsFirst) {
    autoStartConversation();
}

// Initialize: fetch checkpoint count for badge
fetch('/api/checkpoints')
    .then(res => res.json())
    .then(data => {
        if (data.checkpoints) {
            updateBadgeCount(data.checkpoints.length);
            currentCheckpoint = data.current;
        }
    })
    .catch(err => console.error('Failed to fetch checkpoints:', err));


// ========== Checkpoint Selection ==========

let currentCheckpoint = null;

// Load checkpoints and populate the list
async function loadCheckpoints() {
    try {
        const response = await fetch('/api/checkpoints');
        const data = await response.json();
        const list = document.getElementById('checkpoint-list');

        currentCheckpoint = data.current;

        if (!data.checkpoints || data.checkpoints.length === 0) {
            list.innerHTML = '<div class="checkpoint-empty">No checkpoints found.<br>Add .pt files to mikegpt/checkpoints/</div>';
            updateBadgeCount(0);
            return;
        }

        updateBadgeCount(data.checkpoints.length);

        list.innerHTML = data.checkpoints.map(cp => {
            const date = new Date(cp.modified * 1000);
            const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const isCurrent = cp.path === currentCheckpoint;

            return `
                <div class="checkpoint-item ${isCurrent ? 'current' : ''}" data-path="${cp.path}">
                    <div class="checkpoint-item-icon">🧠</div>
                    <div class="checkpoint-item-info">
                        <div class="checkpoint-item-name">${cp.name}</div>
                        <div class="checkpoint-item-date">${dateStr}</div>
                    </div>
                    <div class="checkpoint-item-arrow">›</div>
                </div>
            `;
        }).join('');

        // Add click handlers
        list.querySelectorAll('.checkpoint-item').forEach(item => {
            item.addEventListener('click', () => selectCheckpoint(item.dataset.path));
        });

    } catch (err) {
        console.error('Failed to load checkpoints:', err);
        document.getElementById('checkpoint-list').innerHTML =
            '<div class="checkpoint-empty">Error loading checkpoints</div>';
    }
}

// Select a checkpoint and start fresh conversation
async function selectCheckpoint(checkpointPath) {
    // If clicking the current checkpoint, just go back without reloading
    if (checkpointPath === currentCheckpoint) {
        hideCheckpointScreen();
        return;
    }

    try {
        const response = await fetch('/api/switch-model', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({checkpoint_path: checkpointPath})
        });

        if (!response.ok) {
            throw new Error('Failed to switch model');
        }

        currentCheckpoint = checkpointPath;

        // Clear conversation state
        conversationHistory = '';
        cycleHistory = '';
        cycleTokenIds = [];
        messageHistorySnapshots = [];
        messageTokenIdSnapshots = [];
        lastUserMessageElement = null;

        // Clear messages UI
        document.getElementById('messages').innerHTML = '';

        // Hide checkpoint screen
        hideCheckpointScreen();

        // Restart conversation with new model
        if (mikeStartsFirst) {
            autoStartConversation();
        }

    } catch (err) {
        console.error('Failed to switch model:', err);
        alert('Failed to switch model: ' + err.message);
    }
}

// Show checkpoint selection screen
function showCheckpointScreen() {
    loadCheckpoints();
    document.getElementById('checkpoint-screen').classList.add('visible');
    document.getElementById('chat-screen').style.display = 'none';
    document.getElementById('header-title').textContent = 'Models';
    document.querySelector('.phone-container').classList.add('checkpoint-visible');
}

// Hide checkpoint selection screen
function hideCheckpointScreen() {
    document.getElementById('checkpoint-screen').classList.remove('visible');
    document.getElementById('chat-screen').style.display = 'flex';
    document.getElementById('header-title').textContent = 'MikeGPT';
    document.querySelector('.phone-container').classList.remove('checkpoint-visible');
}

// Update badge count
function updateBadgeCount(count) {
    document.getElementById('back-badge').textContent = count;
}

// Event listener for checkpoint screen
document.getElementById('back-button').addEventListener('click', () => {
    const checkpointScreen = document.getElementById('checkpoint-screen');
    if (checkpointScreen.classList.contains('visible')) {
        // Already on checkpoint screen, clicking current model goes back
        // This is handled by selectCheckpoint when clicking current
    } else {
        showCheckpointScreen();
    }
});
