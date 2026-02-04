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
    const wasGroup = rlMethod === 'group';
    rlMethod = method;
    document.getElementById('pair-btn').classList.toggle('active', method === 'pair');
    document.getElementById('group-btn').classList.toggle('active', method === 'group');

    // Toggle views
    const tokenExplorer = document.getElementById('token-explorer');
    const rankingView = document.getElementById('ranking-view');

    if (method === 'pair') {
        tokenExplorer.classList.add('visible');
        tokenExplorer.classList.remove('hidden');
        rankingView.classList.remove('visible');
        // Reset group state when switching away
        if (typeof resetGrpoState === 'function') {
            resetGrpoState();
        }
        // Re-render tree only when switching FROM group to pair
        if (wasGroup && typeof renderLevels === 'function') {
            renderLevels();
        }
    } else {
        tokenExplorer.classList.remove('visible');
        tokenExplorer.classList.add('hidden');
        rankingView.classList.add('visible');
        // Start group generation when switching to group mode
        if (typeof startGrpoGeneration === 'function') {
            startGrpoGeneration();
        }
    }
}

function markPath(type) {
    const fullPath = getCurrentFullPath();
    // Strip the prompt to show only the response tokens in selections panel
    const responsePath = fullPath.startsWith(treeData.prompt)
        ? fullPath.slice(treeData.prompt.length)
        : fullPath;
    const tokenIds = getCurrentTokenIds();
    selectedPaths.push({ type, path: responsePath, token_ids: tokenIds, timestamp: Date.now() });
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
            div.innerHTML = `<strong>✓ GOOD</strong><br>${item.path}`;
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
            div.innerHTML = `<strong>✗ BAD</strong><br>${item.path}`;
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

function formatMetric(value) {
    if (value === undefined || value === null) return 'N/A';
    if (Math.abs(value) < 0.001) return value.toExponential(2);
    return value.toFixed(4);
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
            <div class="history-metrics">
                <div class="metric-item">L2: ${formatMetric(item.l2Diff)}</div>
                <div class="metric-item">KL: ${formatMetric(item.klDivergence)}</div>
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
        // Use unified training endpoint with responses + rewards
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: treeData.prompt,
                responses: [positiveTokenIds[0], negativeTokenIds[0]],
                rewards: [1.0, -1.0]
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
            positiveChange: result.probability_changes[0],
            negativeChange: result.probability_changes[1],
            positivePaths: positivePaths,
            negativePaths: negativePaths,
            l2Diff: result.l2_diff,
            klDivergence: result.kl_divergence
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
