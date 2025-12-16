// drive-render.js - All rendering and display functions

function createTokenCard(node, idx, depth, selectedIdx) {
    const card = document.createElement('div');
    card.className = 'token-card';

    if (depth === 0) {
        if (idx === currentIndex) {
            card.classList.add('active');
        } else {
            card.classList.add('dimmed');
        }
    } else {
        // Background layers - all get dimmed class
        card.classList.add('dimmed');
    }

    const tokenText = document.createElement('div');
    tokenText.className = 'token-text';
    tokenText.textContent = node.token_str.replace(/\n/g, '↵').replace(/\t/g, '→');

    const probText = document.createElement('div');
    probText.className = 'token-prob';
    probText.textContent = `${(node.probability * 100).toFixed(1)}%`;

    const cumulativeText = document.createElement('div');
    cumulativeText.className = 'cumulative-prob';
    cumulativeText.textContent = `Σ ${(node.cumulative_prob * 100).toFixed(3)}%`;

    card.appendChild(tokenText);
    card.appendChild(probText);
    card.appendChild(cumulativeText);

    if (depth === 0) {
        card.addEventListener('click', () => {
            currentIndex = idx;
            renderLevels();
            updatePathDisplay();
        });
    }

    return card;
}

function updateDepthZeroSelection(container) {
    // Update active/dimmed classes on depth-0 cards without rebuilding
    const level0 = container.querySelector('.level-0');
    if (!level0) return;

    const cards = level0.querySelectorAll('.token-card');
    cards.forEach((card, idx) => {
        if (idx === currentIndex) {
            card.classList.remove('dimmed');
            card.classList.add('active');
        } else {
            card.classList.remove('active');
            card.classList.add('dimmed');
        }
    });
}

function rebuildDeeperLayers(container) {
    // Remove existing deeper level containers (preserve level-0)
    container.querySelectorAll('.level-1, .level-2, .level-3').forEach(el => el.remove());

    // Navigate to current position in tree
    let nodes = treeData.children;
    for (const selection of currentPath) {
        if (!nodes || !nodes[selection.index]) return;
        nodes = nodes[selection.index].children;
    }
    if (!nodes || nodes.length === 0) return;

    // Get the selected node at depth 0 and traverse to its children
    const selectedNode = nodes[currentIndex];
    if (!selectedNode || !selectedNode.children) return;

    let childNodes = selectedNode.children;

    // Render depth 1, 2, 3
    for (let depth = 1; depth < 4; depth++) {
        if (!childNodes || childNodes.length === 0) break;

        const levelDiv = document.createElement('div');
        levelDiv.className = `level-container level-${depth}`;

        const carousel = document.createElement('div');
        carousel.className = 'carousel';
        carousel.style.justifyContent = 'center';

        const track = document.createElement('div');
        track.className = 'carousel-track';

        const visibleNodes = childNodes.slice(0, initialK);
        visibleNodes.forEach((node, idx) => {
            const card = createTokenCard(node, idx, depth, 0);
            track.appendChild(card);
        });

        carousel.appendChild(track);
        levelDiv.appendChild(carousel);
        container.appendChild(levelDiv);

        // Move to next depth level (always follow index 0 for deeper layers)
        if (!childNodes[0] || !childNodes[0].children) break;
        childNodes = childNodes[0].children;
    }
}

function renderLevels(animateFromPage = null) {
    const container = document.getElementById('levels-container');

    // During page transition animation, don't rebuild any layers - just update depth 0 selection
    // All layers (including deeper ones) have slide animations that would be interrupted by rebuilding
    // Everything will be fully updated when the animation timeout completes
    if (isAnimating && animateFromPage === null) {
        updateDepthZeroSelection(container);
        updateDepth();
        return;
    }

    container.innerHTML = '';

    // First, navigate to the current position in the tree based on currentPath
    let nodes = treeData.children;
    for (const selection of currentPath) {
        if (!nodes || !nodes[selection.index]) return;
        nodes = nodes[selection.index].children;
    }

    // Now show current level and up to 4 levels ahead from this position
    if (!nodes || nodes.length === 0) return;

    let depthPath = [...currentPath]; // Track path as we go deeper
    let lastRenderedNodes = null;
    let lastDepthPath = null;

    for (let depth = 0; depth < 4; depth++) {
        if (!nodes || nodes.length === 0) break;

        const levelDiv = document.createElement('div');
        levelDiv.className = `level-container level-${depth}`;

        const carousel = document.createElement('div');
        carousel.className = 'carousel';
        if (depth > 0) carousel.style.justifyContent = 'center';

        const selectedIdx = depth === 0 ? currentIndex : 0;

        // Create track
        const track = document.createElement('div');
        track.className = 'carousel-track';
        if (depth === 0) track.id = 'carousel-track-0';

        if (depth === 0) {
            // Depth 0: show all pages with separators
            // Use totalK for root level, nodes.length for deeper levels
            const maxVisible = currentPath.length === 0 ? totalK : nodes.length;
            const visibleNodes = nodes.slice(0, maxVisible);

            visibleNodes.forEach((node, idx) => {
                // Add separator before each page boundary (except first)
                if (idx > 0 && idx % initialK === 0) {
                    const separator = document.createElement('div');
                    separator.className = 'page-separator';

                    // Calculate cumulative probability up to this boundary
                    let cumulativeP = 0;
                    for (let i = 0; i < idx; i++) {
                        cumulativeP += visibleNodes[i].probability;
                    }

                    const topLabel = document.createElement('div');
                    topLabel.className = 'page-separator-label';
                    topLabel.textContent = `k=${idx}`;
                    separator.appendChild(topLabel);

                    const line = document.createElement('div');
                    line.className = 'page-separator-line';
                    separator.appendChild(line);

                    const bottomLabel = document.createElement('div');
                    bottomLabel.className = 'page-separator-label';
                    bottomLabel.textContent = `p=${cumulativeP.toFixed(3)}`;
                    separator.appendChild(bottomLabel);

                    track.appendChild(separator);
                }

                const card = createTokenCard(node, idx, depth, selectedIdx);
                if (idx >= initialK) {
                    card.classList.add('out-of-policy');
                }
                track.appendChild(card);
            });
        } else {
            // Deeper levels: just show first k tokens, centered
            const visibleNodes = nodes.slice(0, initialK);

            visibleNodes.forEach((node, idx) => {
                const card = createTokenCard(node, idx, depth, selectedIdx);
                track.appendChild(card);
            });
        }

        carousel.appendChild(track);
        levelDiv.appendChild(carousel);
        container.appendChild(levelDiv);

        // Position the track - depth 0 gets page-based positioning,
        // deeper layers follow the same animation pattern
        if (depth === 0) {
            requestAnimationFrame(() => {
                positionTrackForPage(track, currentPage, animateFromPage);
            });
        } else if (animateFromPage !== null && animateFromPage !== currentPage) {
            // During page animation, deeper layers should slide in from the same direction
            // as the top layer to create a unified movement effect
            const direction = currentPage > animateFromPage ? 1 : -1; // Same direction as top layer movement
            const carousel = track.parentElement;
            const viewportWidth = carousel ? carousel.offsetWidth : 800;
            const offsetAmount = viewportWidth * 0.3; // Start 30% viewport off to the side

            track.style.transform = `translateX(${direction * offsetAmount}px)`;
            track.style.opacity = '0.3';

            // Use double-RAF to avoid forced reflow jank
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    track.classList.add('animating');
                    track.style.transform = 'translateX(0)';
                    track.style.opacity = '1';
                    setTimeout(() => track.classList.remove('animating'), 400);
                });
            });
        }

        lastRenderedNodes = nodes;
        lastDepthPath = depthPath;

        // Move to children of selected node for next level
        if (!nodes[selectedIdx] || !nodes[selectedIdx].children) {
            // Don't try to load deeper children until the current page's depths are fully loaded
            // This prevents race conditions between loadPageDepths and loadNextLayer
            if (!loadedPageDepths.has(currentPage) || loadingPageDepths.has(currentPage)) {
                // Page depths still loading - wait for completion and re-render
                break;
            }

            const pathKey = depthPath.map(p => p.token_id).join(',');
            if (!loadedDepths.has(pathKey) && lastRenderedNodes) {
                loadedDepths.add(pathKey);
                loadNextLayer(lastRenderedNodes, lastDepthPath);
            }
            break;
        }

        if (depth === 0) {
            depthPath = [...currentPath, { token_id: nodes[selectedIdx].token_id, index: selectedIdx }];
        } else {
            depthPath = [...depthPath, { token_id: nodes[selectedIdx].token_id, index: selectedIdx }];
        }

        nodes = nodes[selectedIdx].children;
    }

    updateDepth();
}

function positionTrackForPage(track, targetPage, animateFromPage) {
    const cards = track.querySelectorAll('.token-card');
    const separators = track.querySelectorAll('.page-separator');
    const carousel = track.parentElement;

    if (cards.length === 0 || !carousel) return;

    const viewportWidth = carousel.offsetWidth;
    const viewportCenter = viewportWidth / 2;
    const gap = 30; // gap between cards

    // Calculate the center X position of a page within the track
    const calculatePageCenterX = (page) => {
        let x = 0;
        const pageStartIdx = page * initialK;
        const pageEndIdx = Math.min(pageStartIdx + initialK, cards.length);

        // Add width of all cards before this page
        for (let i = 0; i < pageStartIdx && i < cards.length; i++) {
            x += cards[i].offsetWidth + gap;
        }

        // Add separator widths before this page
        for (let i = 0; i < page && i < separators.length; i++) {
            x += separators[i].offsetWidth + gap;
        }

        // Now x is at the left edge of this page
        // Calculate the width of this page
        let pageWidth = 0;
        for (let i = pageStartIdx; i < pageEndIdx && i < cards.length; i++) {
            pageWidth += cards[i].offsetWidth;
            if (i < pageEndIdx - 1) pageWidth += gap;
        }

        // Return the center of this page
        return x + pageWidth / 2;
    };

    const targetCenterX = calculatePageCenterX(targetPage);
    const targetOffset = viewportCenter - targetCenterX;

    if (animateFromPage !== null && animateFromPage !== targetPage) {
        // Animate from old position to new
        const fromCenterX = calculatePageCenterX(animateFromPage);
        const fromOffset = viewportCenter - fromCenterX;

        track.style.transform = `translateX(${fromOffset}px)`;

        // Use double-RAF to avoid forced reflow jank
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                track.classList.add('animating');
                track.style.transform = `translateX(${targetOffset}px)`;
                setTimeout(() => track.classList.remove('animating'), 400);
            });
        });
    } else {
        // Just set position
        track.style.transform = `translateX(${targetOffset}px)`;
    }
}

function updatePathDisplay() {
    const pathText = document.getElementById('path-text');
    const fullPath = getCurrentFullPath();
    pathText.textContent = fullPath || '(start)';
}

function updateDepth() {
    document.getElementById('depth-value').textContent = currentPath.length;
}
