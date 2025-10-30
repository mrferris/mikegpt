// MikeGPT Frontend

const API_URL = window.location.origin;  // Use same origin as the page
const SESSION_ID = 'session_' + Date.now();

let conversationHistory = '';
let lastUserMessageElement = null;

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
    '<|Laughed|>': '😂',
    '<|Loved|>': '❤️',
    '<|Disliked|>': '👎',
    '<|Questioned|>': '❓',
    '<|Emphasized|>': '‼️',
};

// Add message to UI
function addMessage(text, isUser) {
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
                            addReaction(data.response);
                        } else {
                            addMessage(data.response, false);
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
