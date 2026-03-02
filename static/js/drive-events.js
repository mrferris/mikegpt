// drive-events.js - Event listeners and initialization

// Toggle history dropdown
document.addEventListener('DOMContentLoaded', () => {
    const historyHeader = document.getElementById('history-header');
    const historyList = document.getElementById('history-list');
    const dropdownArrow = document.getElementById('dropdown-arrow');

    historyHeader.addEventListener('click', () => {
        historyList.classList.toggle('open');
        dropdownArrow.classList.toggle('open');
    });

    // Fetch persistent training history
    fetch('/api/training-history')
        .then(res => res.json())
        .then(data => {
            if (data.steps && data.steps.length > 0) {
                trainingHistory = data.steps.map(step => ({
                    timestamp: new Date(step.timestamp).getTime(),
                    positiveChange: step.probability_changes?.[0] ?? 0,
                    negativeChange: step.probability_changes?.[step.probability_changes?.length - 1] ?? 0,
                    // Use response_texts (decoded text) if available, fall back to responses (token IDs) for old entries
                    positivePaths: (step.response_texts || step.responses)?.slice(0, Math.ceil((step.response_texts || step.responses)?.length / 2)) || [],
                    negativePaths: (step.response_texts || step.responses)?.slice(Math.ceil((step.response_texts || step.responses)?.length / 2)) || [],
                    type: step.type || 'pair',
                    allChanges: step.probability_changes,
                    l2Diff: step.l2_diff,
                    klDivergence: step.kl_divergence
                }));
                updateTrainingHistory();
            }
        })
        .catch(err => console.error('Failed to load training history:', err));

    // Auto-fill and auto-start if prompt was passed via query parameter
    if (promptParam) {
        const promptInput = document.getElementById('prompt');
        if (promptInput) {
            promptInput.value = promptParam;
            startDrive().then(() => {
                // If bad_text was passed, add it directly as a negative example
                // Stay at depth 0 - don't navigate into the tree
                if (badTextParam && navTokenIds && navTokenIds.length > 0) {
                    selectedPaths.push({
                        type: 'negative',
                        path: badTextParam,
                        token_ids: navTokenIds,
                        timestamp: Date.now()
                    });
                    updateSelectionsPanel();
                }
            });
        }
    }
});

// Enter to start in prompt inputs
document.getElementById('prompt').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        startDrive();
    }
});

document.getElementById('prompt-imessage').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        startDrive();
    }
});

document.getElementById('param-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
    }
});

// Keyboard controls
document.addEventListener('keydown', (e) => {
    // Don't interfere with typing in the prompt inputs or param input
    if (document.activeElement === document.getElementById('prompt') ||
        document.activeElement === document.getElementById('prompt-imessage') ||
        document.activeElement === document.getElementById('param-input')) {
        return;
    }

    if (document.getElementById('loading-screen').classList.contains('hidden')) {
        // Group mode keyboard handling
        if (rlMethod === 'group') {
            // Check for Backspace and Enter FIRST (before letter checks)
            if (e.key === 'Backspace') {
                e.preventDefault();
                handleGrpoBackspace();
                return;
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (grpoRankings.length >= 2) {
                    submitGrpoRankings();
                }
                return;
            } else if (e.key.length === 1 && e.key >= 'a' && e.key <= 'h') {
                e.preventDefault();
                handleGrpoKeyPress(e.key);
                return;
            } else if (e.key.length === 1 && e.key >= 'A' && e.key <= 'H') {
                e.preventDefault();
                handleGrpoKeyPress(e.key.toLowerCase());
                return;
            }
            return; // Don't process pair mode keys in group mode
        }

        // Pair mode keyboard handling
        switch(e.key) {
            case 'ArrowLeft':
                e.preventDefault();
                navigateLeft();
                break;
            case 'ArrowRight':
                e.preventDefault();
                navigateRight();
                break;
            case ' ':
                e.preventDefault();
                selectAndAdvance();
                break;
            case 'Backspace':
                e.preventDefault();
                goBack();
                updateMobileButtons();
                break;
            case 'g':
            case 'G':
                markPath('positive');
                break;
            case 'b':
            case 'B':
                markPath('negative');
                break;
            case 'e':
            case 'E':
                exportSelections();
                break;
            case 'Escape':
                // Undo good selection first, then bad if no good exists
                const goodIdx = selectedPaths.findLastIndex(p => p.type === 'positive');
                if (goodIdx !== -1) {
                    selectedPaths.splice(goodIdx, 1);
                    updateSelectionsPanel();
                } else {
                    const badIdx = selectedPaths.findLastIndex(p => p.type === 'negative');
                    if (badIdx !== -1) {
                        selectedPaths.splice(badIdx, 1);
                        updateSelectionsPanel();
                    }
                }
                break;
        }
    }
});

// ========== Mobile Touch Controls ==========

// Button handlers — toggle good/bad, greyed-out train until both selected
function updateMobileButtons() {
    const hasGood = selectedPaths.some(p => p.type === 'positive');
    const hasBad = selectedPaths.some(p => p.type === 'negative');
    const btnGood = document.getElementById('btn-good');
    const btnBad = document.getElementById('btn-bad');
    const btnExec = document.getElementById('btn-execute');
    if (btnGood) btnGood.classList.toggle('selected', hasGood);
    if (btnBad) btnBad.classList.toggle('selected', hasBad);
    if (btnExec) btnExec.classList.toggle('ready', hasGood && hasBad);

    // Show/hide back button based on depth
    const navGroup = document.getElementById('mobile-nav-group');
    if (navGroup) navGroup.style.display = currentPath.length > 0 ? 'flex' : 'none';
}

document.getElementById('btn-good')?.addEventListener('click', () => {
    const hasGood = selectedPaths.some(p => p.type === 'positive');
    if (hasGood) {
        const idx = selectedPaths.findLastIndex(p => p.type === 'positive');
        selectedPaths.splice(idx, 1);
        updateSelectionsPanel();
    } else {
        markPath('positive');
    }
    updateMobileButtons();
});

document.getElementById('btn-bad')?.addEventListener('click', () => {
    const hasBad = selectedPaths.some(p => p.type === 'negative');
    if (hasBad) {
        const idx = selectedPaths.findLastIndex(p => p.type === 'negative');
        selectedPaths.splice(idx, 1);
        updateSelectionsPanel();
    } else {
        markPath('negative');
    }
    updateMobileButtons();
});

document.getElementById('btn-execute')?.addEventListener('click', () => {
    const hasGood = selectedPaths.some(p => p.type === 'positive');
    const hasBad = selectedPaths.some(p => p.type === 'negative');
    if (hasGood && hasBad) exportSelections();
});

document.getElementById('btn-back')?.addEventListener('click', () => {
    goBack();
    updateMobileButtons();
});

document.getElementById('btn-grpo-execute')?.addEventListener('click', () => {
    if (grpoRankings.length >= 2) submitGrpoRankings();
});

// Swipe to navigate, double tap to go deeper
(function() {
    let touchStartX = 0;
    let touchStartY = 0;
    let touchActive = false;

    // Reset touch state when returning from another tab
    document.addEventListener('visibilitychange', () => {
        touchActive = false;
        lastTapTime = 0;
    });
    let lastTapTime = 0;

    let touchOnTopRow = false;

    document.addEventListener('touchstart', (e) => {
        // Ignore touches on buttons/inputs
        if (e.target.closest('.mobile-controls, .mobile-grpo-controls, .selections-panel, input, button')) return;
        touchStartX = e.touches[0].clientX;
        touchStartY = e.touches[0].clientY;
        touchOnTopRow = true;
        touchActive = true;
    }, { passive: true });

    document.addEventListener('touchend', (e) => {
        if (!touchActive) return;
        touchActive = false;
        if (e.target.closest('.mobile-controls, .mobile-grpo-controls, .selections-panel, input, button')) return;
        if (rlMethod !== 'pair') return;

        const touch = e.changedTouches[0];
        const rawDx = touch.clientX - touchStartX;
        const rawDy = touch.clientY - touchStartY;

        // After 90deg CSS rotation, physical horizontal swipe = vertical delta
        // Swap axes so swipe logic works in rotated coordinates
        const isMobileRotated = window.innerWidth <= 768 && window.innerHeight > window.innerWidth;
        const dx = isMobileRotated ? -rawDy : rawDx;
        const dy = isMobileRotated ? rawDx : rawDy;

        // Swipe detection — jump to next/previous page
        if (touchOnTopRow && Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) {
            e.preventDefault();
            const oldPage = currentPage;
            if (dx > 0) {
                // Swipe right → next page
                const nextPageStart = (currentPage + 1) * initialK;
                // Navigate right until we reach the next page
                for (let i = currentIndex; i < nextPageStart && i < totalK - 1; i++) {
                    navigateRight();
                }
            } else {
                // Swipe left → previous page
                if (currentPage > 0) {
                    const prevPageStart = (currentPage - 1) * initialK;
                    currentIndex = prevPageStart;
                    currentPage = currentPage - 1;
                    animatePageTransition(oldPage, currentPage);
                    updatePathDisplay();
                    ensureCurrentTokenHasDepth();
                }
            }
            return;
        }

    }, { passive: false });
})();
