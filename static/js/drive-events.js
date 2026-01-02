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
        // GRPO mode keyboard handling
        if (rlMethod === 'grpo') {
            // Check for Backspace and Enter FIRST (before letter checks)
            if (e.key === 'Backspace') {
                e.preventDefault();
                handleGrpoBackspace();
                return;
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if (grpoRankings.length === 8) {
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
            return; // Don't process DPO keys in GRPO mode
        }

        // DPO mode keyboard handling
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
        }
    }
});
