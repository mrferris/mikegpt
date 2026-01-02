// drive-training.js - Training, selection, and RL method functions

function toggleSamplingMethod() {
    samplingMethod = samplingMethod === 'topk' ? 'topp' : 'topk';
    const toggle = document.getElementById('sampling-toggle');
    const prefix = document.getElementById('param-prefix');
    const input = document.getElementById('param-input');

    if (samplingMethod === 'topp') {
        toggle.classList.add('active');
        prefix.textContent = 'p=';
        input.value = '0.9';
    } else {
        toggle.classList.remove('active');
        prefix.textContent = 'k=';
        input.value = '5';
    }
}

function setRLMethod(method) {
    rlMethod = method;
    document.getElementById('dpo-btn').classList.toggle('active', method === 'dpo');
    document.getElementById('grpo-btn').classList.toggle('active', method === 'grpo');

    // Toggle views
    const tokenExplorer = document.getElementById('token-explorer');
    const rankingView = document.getElementById('ranking-view');

    if (method === 'dpo') {
        tokenExplorer.classList.add('visible');
        tokenExplorer.classList.remove('hidden');
        rankingView.classList.remove('visible');
        // Reset GRPO state when switching away
        if (typeof resetGrpoState === 'function') {
            resetGrpoState();
        }
    } else {
        tokenExplorer.classList.remove('visible');
        tokenExplorer.classList.add('hidden');
        rankingView.classList.add('visible');
        // Start GRPO generation when switching to GRPO mode
        if (typeof startGrpoGeneration === 'function') {
            startGrpoGeneration();
        }
    }
}

function markPath(type) {
    const path = getCurrentFullPath();
    const tokenIds = getCurrentTokenIds();
    selectedPaths.push({ type, path, token_ids: tokenIds, timestamp: Date.now() });
    updateSelectionsPanel();
}

function updateSelectionsPanel() {
    const selectionsList = document.getElementById('selections-list');
    const executeHint = document.getElementById('execute-hint');

    const positiveItems = selectedPaths.filter(p => p.type === 'positive');
    const negativeItems = selectedPaths.filter(p => p.type === 'negative');

    // Show execute hint if both positive and negative are selected
    if (positiveItems.length > 0 && negativeItems.length > 0) {
        executeHint.classList.add('show');
    } else {
        executeHint.classList.remove('show');
    }

    // Show selected paths or placeholders
    selectionsList.innerHTML = '';

    // Positive section
    if (positiveItems.length > 0) {
        positiveItems.forEach(item => {
            const div = document.createElement('div');
            div.className = 'selection-item positive';
            div.innerHTML = `
                <strong>✓ GOOD</strong><br>
                ${item.path.substring(0, 50)}${item.path.length > 50 ? '...' : ''}
            `;
            selectionsList.appendChild(div);
        });
    } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'selection-placeholder';
        placeholder.textContent = 'No positive example selected';
        selectionsList.appendChild(placeholder);
    }

    // Negative section
    if (negativeItems.length > 0) {
        negativeItems.forEach(item => {
            const div = document.createElement('div');
            div.className = 'selection-item negative';
            div.innerHTML = `
                <strong>✗ BAD</strong><br>
                ${item.path.substring(0, 50)}${item.path.length > 50 ? '...' : ''}
            `;
            selectionsList.appendChild(div);
        });
    } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'selection-placeholder';
        placeholder.textContent = 'No negative example selected';
        selectionsList.appendChild(placeholder);
    }
}

function formatPercent(value) {
    // For very small values, show more precision
    if (Math.abs(value) < 0.01 && value !== 0) {
        // Find how many decimal places we need
        let formatted = value.toFixed(6);
        // Remove trailing zeros
        formatted = formatted.replace(/\.?0+$/, '');
        return formatted;
    }
    return value.toFixed(2);
}

function updateTrainingHistory() {
    const historyHeader = document.getElementById('history-header');
    const historyList = document.getElementById('history-list');

    if (trainingHistory.length === 0) {
        historyHeader.style.display = 'none';
        return;
    }

    historyHeader.style.display = 'flex';
    historyList.innerHTML = '';

    trainingHistory.slice().reverse().forEach((item, idx) => {
        const div = document.createElement('div');
        div.className = 'history-item';

        const timestamp = new Date(item.timestamp).toLocaleTimeString();

        let pathsHtml = '<div class="history-paths">';
        if (item.positivePaths && item.positivePaths.length > 0) {
            pathsHtml += `<div class="history-path-label">✓ Positive:</div>`;
            item.positivePaths.forEach(path => {
                const truncated = path.length > 60 ? path.substring(0, 60) + '...' : path;
                pathsHtml += `<div>${truncated}</div>`;
            });
        }
        if (item.negativePaths && item.negativePaths.length > 0) {
            pathsHtml += `<div class="history-path-label">✗ Negative:</div>`;
            item.negativePaths.forEach(path => {
                const truncated = path.length > 60 ? path.substring(0, 60) + '...' : path;
                pathsHtml += `<div>${truncated}</div>`;
            });
        }
        pathsHtml += '</div>';

        div.innerHTML = `
            <div class="history-timestamp">#${trainingHistory.length - idx} - ${timestamp}</div>
            ${pathsHtml}
            <div class="history-changes">
                <div class="prob-change prob-positive">+${formatPercent(item.positiveChange)}%</div>
                <div class="prob-change prob-negative">${formatPercent(item.negativeChange)}%</div>
            </div>
        `;
        historyList.appendChild(div);
    });
}

async function exportSelections() {
    if (selectedPaths.length === 0) {
        alert('No paths selected yet');
        return;
    }

    // Separate positive and negative token ID sequences
    const positiveTokenIds = selectedPaths.filter(p => p.type === 'positive').map(p => p.token_ids);
    const negativeTokenIds = selectedPaths.filter(p => p.type === 'negative').map(p => p.token_ids);

    if (positiveTokenIds.length === 0 || negativeTokenIds.length === 0) {
        alert('Need at least one positive and one negative path');
        return;
    }

    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: treeData.prompt,
                positive_token_ids: positiveTokenIds,
                negative_token_ids: negativeTokenIds
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Training failed');
        }

        // Add to training history
        const positivePaths = selectedPaths.filter(p => p.type === 'positive').map(p => p.path);
        const negativePaths = selectedPaths.filter(p => p.type === 'negative').map(p => p.path);

        trainingHistory.push({
            timestamp: Date.now(),
            positiveChange: result.positive_change_percent,
            negativeChange: result.negative_change_percent,
            positivePaths: positivePaths,
            negativePaths: negativePaths
        });

        // Update displays
        updateTrainingHistory();

        // Clear selections after successful training
        selectedPaths = [];
        updateSelectionsPanel();

    } catch (error) {
        alert('Training error: ' + error.message);
    }
}
