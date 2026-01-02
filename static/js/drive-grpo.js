// drive-grpo.js - GRPO generation, ranking, and submission functions

// Initialize GRPO view with 8 empty response rows
function renderGrpoView() {
    const promptText = document.getElementById('grpo-prompt-text');
    const responsesContainer = document.getElementById('grpo-responses');

    promptText.textContent = originalPrompt || '';

    // Create 8 response rows
    responsesContainer.innerHTML = '';
    const letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

    for (let i = 0; i < 8; i++) {
        const row = document.createElement('div');
        row.className = 'grpo-response-row';
        row.id = `grpo-row-${i}`;
        row.innerHTML = `
            <div class="grpo-letter">${letters[i]}</div>
            <div class="grpo-response-text empty" id="grpo-text-${i}">Waiting...</div>
            <div class="grpo-rank-badge" id="grpo-rank-${i}"></div>
        `;
        responsesContainer.appendChild(row);
    }
}

// Start generating 8 responses
async function startGrpoGeneration() {
    // Reset GRPO state
    grpoResponses = [];
    grpoRankings = [];
    grpoGenerating = true;
    grpoCurrentIndex = 0;

    // Render initial view
    renderGrpoView();

    // Disable instructions while generating
    const instructions = document.getElementById('grpo-instructions');
    instructions.classList.add('disabled');

    // Get sampling parameters
    const paramValue = parseFloat(document.getElementById('param-input').value);
    const useTopK = samplingMethod === 'topk';
    const topK = useTopK ? Math.round(paramValue) : 5;
    const topP = useTopK ? 0.9 : paramValue;

    try {
        const response = await fetch('/api/grpo-generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: originalPrompt,
                temperature: 1.0,
                top_k: topK,
                top_p: topP,
                use_top_k: useTopK
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // Helper function to process SSE lines
        function processLine(line) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.error) {
                        console.error('GRPO generation error:', data.error);
                        return;
                    }

                    if (data.all_done) {
                        // All responses complete
                        grpoResponses = data.responses;
                        grpoGenerating = false;
                        instructions.classList.remove('disabled');

                        // Update all rows to show complete text
                        for (let i = 0; i < 8; i++) {
                            const textEl = document.getElementById(`grpo-text-${i}`);
                            const rowEl = document.getElementById(`grpo-row-${i}`);
                            textEl.textContent = grpoResponses[i]?.text || '(empty)';
                            textEl.classList.remove('streaming', 'empty');
                            rowEl.classList.remove('generating');
                        }
                    } else if (data.done) {
                        // Single response complete
                        const textEl = document.getElementById(`grpo-text-${data.index}`);
                        const rowEl = document.getElementById(`grpo-row-${data.index}`);
                        textEl.textContent = data.full_response || '(empty)';
                        textEl.classList.remove('streaming', 'empty');
                        rowEl.classList.remove('generating');

                        // Mark next row as generating
                        if (data.index < 7) {
                            const nextRowEl = document.getElementById(`grpo-row-${data.index + 1}`);
                            const nextTextEl = document.getElementById(`grpo-text-${data.index + 1}`);
                            nextRowEl.classList.add('generating');
                            nextTextEl.classList.remove('empty');
                            nextTextEl.classList.add('streaming');
                            nextTextEl.textContent = '';
                        }
                    } else {
                        // Token received - update streaming text
                        const textEl = document.getElementById(`grpo-text-${data.index}`);
                        const rowEl = document.getElementById(`grpo-row-${data.index}`);

                        if (textEl.classList.contains('empty')) {
                            textEl.classList.remove('empty');
                            textEl.classList.add('streaming');
                            textEl.textContent = '';
                            rowEl.classList.add('generating');
                        }

                        textEl.textContent += data.token;
                    }
                } catch (e) {
                    // Skip invalid JSON
                }
            }
        }

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                processLine(line);
            }
        }

        // Process any remaining data in buffer after stream ends
        if (buffer.trim()) {
            const remainingLines = buffer.split('\n');
            for (const line of remainingLines) {
                processLine(line);
            }
        }

    } catch (error) {
        console.error('GRPO generation failed:', error);
    } finally {
        // Always ensure generating flag is cleared and UI is enabled
        grpoGenerating = false;
        instructions.classList.remove('disabled');
    }
}

// Handle ranking key press (a-h)
function handleGrpoKeyPress(letter) {
    if (grpoGenerating) return;
    if (grpoRankings.length >= 8) return;

    const index = letter.charCodeAt(0) - 'a'.charCodeAt(0);

    // Check if already ranked
    if (grpoRankings.includes(index)) return;

    // Add to rankings
    grpoRankings.push(index);

    // Update display
    updateGrpoRankDisplay();
}

// Handle backspace to undo last ranking
function handleGrpoBackspace() {
    if (grpoGenerating) return;
    if (grpoRankings.length === 0) return;

    // Remove last ranking
    grpoRankings.pop();

    // Update display
    updateGrpoRankDisplay();
}

// Update the visual display of rankings
function updateGrpoRankDisplay() {
    // Clear all rankings first
    for (let i = 0; i < 8; i++) {
        const rowEl = document.getElementById(`grpo-row-${i}`);
        const rankEl = document.getElementById(`grpo-rank-${i}`);

        // Remove all rank classes
        for (let r = 1; r <= 8; r++) {
            rowEl.classList.remove(`grpo-rank-${r}`);
        }

        rankEl.classList.remove('visible');
        rankEl.textContent = '';
    }

    // Apply rankings in order
    for (let rank = 0; rank < grpoRankings.length; rank++) {
        const index = grpoRankings[rank];
        const rowEl = document.getElementById(`grpo-row-${index}`);
        const rankEl = document.getElementById(`grpo-rank-${index}`);

        // Add rank class (1-indexed)
        rowEl.classList.add(`grpo-rank-${rank + 1}`);

        // Show rank badge
        rankEl.textContent = `${rank + 1}${getOrdinalSuffix(rank + 1)}`;
        rankEl.classList.add('visible');
    }
}

// Get ordinal suffix (st, nd, rd, th)
function getOrdinalSuffix(n) {
    if (n === 1) return 'st';
    if (n === 2) return 'nd';
    if (n === 3) return 'rd';
    return 'th';
}

// Submit rankings for training
async function submitGrpoRankings() {
    if (grpoGenerating) return;
    if (grpoRankings.length !== 8) return;

    try {
        // Build responses_ranked array: [[token_ids, rank], ...]
        // rank 1 is best, 8 is worst
        const responsesRanked = grpoRankings.map((responseIndex, rankIndex) => {
            const response = grpoResponses[responseIndex];
            return [response.tokens, rankIndex + 1];  // rank is 1-indexed
        });

        const response = await fetch('/api/grpo-train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: originalPrompt,
                responses_ranked: responsesRanked
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Training failed');
        }

        // Add to training history
        addGrpoToHistory(result);

        // Start generating new responses
        startGrpoGeneration();

    } catch (error) {
        console.error('GRPO training failed:', error);
        alert('Training error: ' + error.message);
    }
}

// Add GRPO training result to history
function addGrpoToHistory(result) {
    const letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

    // Build paths for history display
    const rankedPaths = grpoRankings.map((responseIndex, rankIndex) => {
        const response = grpoResponses[responseIndex];
        return `${rankIndex + 1}. (${letters[responseIndex]}) ${response.text.substring(0, 40)}${response.text.length > 40 ? '...' : ''}`;
    });

    trainingHistory.push({
        timestamp: Date.now(),
        type: 'grpo',
        positiveChange: result.probability_changes[0] || 0,
        negativeChange: result.probability_changes[7] || 0,
        positivePaths: rankedPaths.slice(0, 4),
        negativePaths: rankedPaths.slice(4, 8),
        allChanges: result.probability_changes
    });

    // Update the history display
    updateTrainingHistory();

    // Show history panel
    const historyHeader = document.getElementById('history-header');
    const historyList = document.getElementById('history-list');
    const dropdownArrow = document.getElementById('dropdown-arrow');

    historyHeader.style.display = 'flex';
    historyList.classList.add('open');
    dropdownArrow.classList.add('open');
}

// Reset GRPO state when switching away
function resetGrpoState() {
    grpoResponses = [];
    grpoRankings = [];
    grpoGenerating = false;
    grpoCurrentIndex = 0;
}
