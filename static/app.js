// MikeGPT Frontend

const API_URL = window.location.origin;  // Use same origin as the page
const SESSION_ID = 'session_' + Date.now();

let conversationHistory = '';
let lastUserMessageElement = null;

// Auto-start mode: if true, MikeGPT sends the first message
let mikeStartsFirst = true;

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

// Auto-start: MikeGPT sends the first message
async function autoStartConversation() {
    const input = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');

    // Disable input while generating
    input.disabled = true;
    sendButton.disabled = true;

    // Show typing indicator immediately
    showTypingIndicator();

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

                        // For auto-start, responses are always bot messages (no reactions possible yet)
                        addMessage(data.response, false);

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

// Get first user message from conversation
function getFirstUserMessage() {
    const messages = document.getElementById('messages');
    const userMessage = messages.querySelector('.message.user .message-bubble');
    return userMessage ? userMessage.textContent : '';
}

// Store original content for reversing animation
let originalPhoneContent = null;
let isLightMode = false;

// Animate to MikeRL iPad mode (light mode)
async function animateToMikeRL() {
    if (isLightMode) return; // Already in light mode

    isLightMode = true;
    const phoneContainer = document.querySelector('.phone-container');
    const messagesContainer = document.getElementById('messages');
    const inputContainer = document.querySelector('.input-container');
    const chatHeader = document.querySelector('.chat-header');
    const statusBar = document.querySelector('.status-bar');

    // Store original content for later restoration
    originalPhoneContent = {
        html: phoneContainer.innerHTML,
        width: phoneContainer.style.width,
        maxWidth: phoneContainer.style.maxWidth,
        height: phoneContainer.style.height,
        maxHeight: phoneContainer.style.maxHeight
    };

    // Get the first user message to use as prompt
    const firstMessage = getFirstUserMessage();

    // Fade out content
    messagesContainer.style.transition = 'opacity 0.3s ease';
    inputContainer.style.transition = 'opacity 0.3s ease';
    chatHeader.style.transition = 'opacity 0.3s ease';
    statusBar.style.transition = 'opacity 0.3s ease';

    messagesContainer.style.opacity = '0';
    inputContainer.style.opacity = '0';
    chatHeader.style.opacity = '0';
    statusBar.style.opacity = '0';

    await new Promise(resolve => setTimeout(resolve, 300));

    // Animate phone to iPad size
    phoneContainer.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    phoneContainer.style.width = '98vw';
    phoneContainer.style.maxWidth = '1800px';
    phoneContainer.style.height = '96vh';
    phoneContainer.style.maxHeight = '1200px';

    await new Promise(resolve => setTimeout(resolve, 600));

    // Hide original content
    messagesContainer.style.display = 'none';
    inputContainer.style.display = 'none';
    chatHeader.style.display = 'none';
    statusBar.style.display = 'none';

    // Load MikeRL interface
    const response = await fetch('/drive?mode=imessage');
    const html = await response.text();

    // Extract content from the loaded page
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    // Extract and inject styles
    const styles = doc.querySelectorAll('style');
    styles.forEach(style => {
        const newStyle = document.createElement('style');
        newStyle.textContent = style.textContent;
        document.head.appendChild(newStyle);
    });

    // Inject body content
    const driveContent = doc.body.innerHTML;
    phoneContainer.innerHTML = driveContent;
    phoneContainer.classList.add('ipad-mode');

    // Add light-mode class to body
    document.body.classList.add('light-mode');

    // Execute scripts from the loaded content
    const scripts = phoneContainer.querySelectorAll('script');
    scripts.forEach(script => {
        const newScript = document.createElement('script');
        newScript.textContent = script.textContent;
        script.parentNode.replaceChild(newScript, script);
    });

    // Pre-fill the prompt if we have a first message
    setTimeout(() => {
        const promptInput = document.getElementById('prompt-imessage');
        if (promptInput && firstMessage) {
            promptInput.value = firstMessage;
        }

        // Initialize Training History dropdown
        const historyHeader = document.getElementById('history-header');
        const historyList = document.getElementById('history-list');
        const dropdownArrow = document.getElementById('dropdown-arrow');

        if (historyHeader && historyList && dropdownArrow) {
            historyHeader.addEventListener('click', () => {
                historyList.classList.toggle('open');
                dropdownArrow.classList.toggle('open');
            });
        }

        // In iMessage mode, show game container but keep loading screen visible as input bar
        const gameContainer = document.getElementById('game-container');
        if (gameContainer) {
            gameContainer.classList.add('active');
        }
    }, 100);
}

// Reverse animation back to MikeGPT
async function animateBackToMikeGPT() {
    if (!isLightMode) return; // Not in light mode

    const phoneContainer = document.querySelector('.phone-container');

    // Remove light-mode class from body
    document.body.classList.remove('light-mode');

    // Fade out the MikeRL content
    phoneContainer.style.transition = 'opacity 0.3s ease';
    phoneContainer.style.opacity = '0';

    await new Promise(resolve => setTimeout(resolve, 300));

    // Animate back to phone size
    phoneContainer.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    phoneContainer.style.width = originalPhoneContent.width || '100%';
    phoneContainer.style.maxWidth = originalPhoneContent.maxWidth || '400px';
    phoneContainer.style.height = originalPhoneContent.height || '90vh';
    phoneContainer.style.maxHeight = originalPhoneContent.maxHeight || '800px';
    phoneContainer.classList.remove('ipad-mode');

    await new Promise(resolve => setTimeout(resolve, 600));

    // Restore original content
    phoneContainer.innerHTML = originalPhoneContent.html;

    // Fade back in
    phoneContainer.style.opacity = '1';

    await new Promise(resolve => setTimeout(resolve, 300));

    // Re-attach event listeners
    document.getElementById('send-button').addEventListener('click', sendMessage);
    document.getElementById('message-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Focus the input
    document.getElementById('message-input').focus();

    isLightMode = false;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Press 'r' to enter light mode (MikeRL)
    if (e.key === 'r' || e.key === 'R') {
        // Don't trigger if typing in input
        if (document.activeElement === document.getElementById('message-input')) {
            return;
        }
        animateToMikeRL();
    }

    // Press 'Escape' to exit light mode and return to MikeGPT
    if (e.key === 'Escape') {
        animateBackToMikeGPT();
    }
});
