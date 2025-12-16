// drive-api.js - API calls and data loading functions

async function startDrive() {
    // Check if we're in light mode and use the appropriate input
    const isLightMode = document.body.classList.contains('light-mode');
    const promptInput = isLightMode ?
        document.getElementById('prompt-imessage') :
        document.getElementById('prompt');
    const prompt = promptInput.value;

    // Get k or p value based on sampling method
    const paramValue = parseFloat(document.getElementById('param-input').value);
    const userK = samplingMethod === 'topk' ? Math.round(paramValue) : 20; // User's k for top-p threshold
    const k = userK; // Only fetch k initially for fast load
    const n = parseInt(document.getElementById('n-value').value);

    // Store the policy boundary
    initialK = userK;

    if (!prompt.trim()) {
        return;
    }

    const errorContainer = document.getElementById('error-container');
    errorContainer.innerHTML = '<div class="loading-text">Building token tree...</div>';

    try {
        // Request k+1 tokens so the separator is visible immediately with first screen
        const response = await fetch('/api/beam-tree', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt, k: k + 1, n })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate tree');
        }

        treeData = data;
        originalPrompt = prompt; // Store the original user prompt
        initialK = k;
        initialN = n;
        console.log('Initial tree loaded with', k, 'tokens');
        startGame();

    } catch (error) {
        showError(error.message);
    }
}

function showError(message) {
    const container = document.getElementById('error-container');
    container.innerHTML = `<div class="error-message">${message}</div>`;
}

function startGame() {
    document.getElementById('loading-screen').classList.add('hidden');
    document.getElementById('rl-method-toggle').classList.add('visible');
    document.getElementById('token-explorer').classList.add('visible');
    currentPath = [];
    currentIndex = 0;
    currentPage = 0;
    selectedPaths = [];
    loadedDepths = new Set();
    loadedPageDepths = new Set(); // Will be topped up to k+1 by loadPageDepths
    loadingPageDepths = new Set(); // No pages currently loading
    loadedPages = new Set([0]); // Only page 0 is fully loaded initially
    isLoadingNextLayer = false;
    isAnimating = false;
    animationTimeoutId = null;
    loadingTokenChildren = new Set(); // Reset token loading tracker
    pendingDepthLoadTimeout = null;
    totalK = initialK + 1; // Show k+1 tokens (k in-policy + 1 for separator)
    console.log(`[startGame] initialK=${initialK}, totalK=${totalK}, treeData.children.length=${treeData.children.length}`);

    // Set initial RL method state
    setRLMethod('dpo');

    // Render immediately
    renderLevels();
    updatePathDisplay();

    // Load second page tokens (first layer) in parallel - shows k+1 token quickly
    loadSecondPageTokens();

    // Top up page 0's children to k+1 in parallel
    loadPageDepths(0);
}

async function loadSecondPageTokens() {
    // Load second k tokens (first layer only) immediately
    try {
        console.log('Loading second page tokens (first layer)...');

        const response = await fetch('/api/beam-tree', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: originalPrompt,
                k: initialK * 2 + 1,  // Load 2k tokens + 1 for page 2 separator
                n: 1  // Only first layer for new tokens
            })
        });

        const newData = await response.json();
        if (response.ok) {
            console.log('✓ Loaded second page first layer');

            // Merge: keep first k+1 tokens' deep children, add remaining tokens' first layer
            // Start at initialK + 1 to preserve the k+1 token loaded with full depth initially
            for (let i = initialK + 1; i < newData.children.length; i++) {
                // Add new tokens (they only have first layer)
                treeData.children[i] = newData.children[i];
            }

            // Set totalK to actual tokens returned, not requested
            const oldTotalK = totalK;
            totalK = Math.max(totalK, newData.children.length);
            console.log(`[loadSecondPageTokens] totalK updated: ${oldTotalK} -> ${totalK}, newData.children.length=${newData.children.length}`);
            loadedPages.add(1); // Mark page 1 first layer as loaded
            renderLevels();

            // Preload full depth for page 1 in background (for smooth scrolling)
            preloadPageFullDepth(1);
        }
    } catch (error) {
        console.error('Error loading second page tokens:', error);
    }
}

// Preload full depth (all 4 layers) for a page in the background
async function preloadPageFullDepth(pageNum) {
    // Skip if page doesn't exist yet
    if (pageNum * initialK >= totalK) return;

    // First ensure root children are loaded (depth 1)
    await loadPageDepths(pageNum);

    const pageStart = pageNum * initialK;
    const pageEnd = Math.min(pageStart + initialK, totalK);

    // Load depth 2: grandchildren (children of children)
    await preloadDepthLevel(pageStart, pageEnd, 2);

    // Load depth 3: great-grandchildren
    await preloadDepthLevel(pageStart, pageEnd, 3);

    // Load depth 4: great-great-grandchildren (we display 4 layers total)
    await preloadDepthLevel(pageStart, pageEnd, 4);

    // Re-render to show the newly loaded depth data (no "pop in" effect)
    renderLevels();
}

async function preloadDepthLevel(pageStart, pageEnd, targetDepth) {
    const nodesToExpand = [];

    // Collect all nodes at targetDepth-1 that need children loaded
    for (let i = pageStart; i < pageEnd && i < treeData.children.length; i++) {
        const rootToken = treeData.children[i];
        if (!rootToken) continue;
        collectNodesNeedingChildren(rootToken, [rootToken.token_id], 1, targetDepth, nodesToExpand);
    }

    if (nodesToExpand.length === 0) return;

    // Mark as loading
    for (const n of nodesToExpand) {
        loadingTokenChildren.add(n.tokenKey);
    }

    try {
        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: treeData.prompt,
                nodes: nodesToExpand.map(n => ({ path: n.path, token_id: n.token_id })),
                k: initialK + 1  // +1 for separator visibility when this becomes depth 0
            })
        });

        const data = await response.json();
        if (!response.ok) return;

        // Merge children into tree (without re-rendering)
        const childrenMap = data.children_map;
        for (const nodeInfo of nodesToExpand) {
            if (childrenMap[nodeInfo.tokenKey]) {
                if (nodeInfo.node && !nodeInfo.node.children) {
                    const children = childrenMap[nodeInfo.tokenKey];
                    for (const child of children) {
                        child.cumulative_prob = (nodeInfo.node.cumulative_prob || nodeInfo.node.probability) * child.probability;
                    }
                    nodeInfo.node.children = children;
                }
            }
        }
    } catch (error) {
        console.error('Error preloading depth level:', error);
    } finally {
        // Clear loading state
        for (const n of nodesToExpand) {
            loadingTokenChildren.delete(n.tokenKey);
        }
    }
}

async function loadPageDepths(pageNum) {
    // Skip if already loaded or currently loading
    if (loadedPageDepths.has(pageNum) || loadingPageDepths.has(pageNum)) return;
    loadingPageDepths.add(pageNum);

    // Load deeper layers for a specific page's tokens using expand-depth
    const pageStart = pageNum * initialK;
    const pageEnd = Math.min(pageStart + initialK, totalK);

    console.log(`Loading depths for page ${pageNum} (tokens ${pageStart}-${pageEnd})...`);

    try {
        // Collect the tokens on this page that need children loaded or topped up
        const nodesToExpand = [];
        for (let i = pageStart; i < pageEnd && i < treeData.children.length; i++) {
            const node = treeData.children[i];
            if (!node) continue;

            const tokenKey = node.token_id.toString();

            // Skip if already loading or already has enough children
            if (loadingTokenChildren.has(tokenKey)) continue;
            if (node.children && node.children.length >= initialK + 1) continue;

            nodesToExpand.push({
                path: [], // Root level - empty path
                token_id: node.token_id,
                tokenKey: tokenKey,
                node: node
            });
        }

        if (nodesToExpand.length === 0) {
            console.log(`Page ${pageNum} tokens already have k+1 children`);
            loadingPageDepths.delete(pageNum);
            loadedPageDepths.add(pageNum);
            return;
        }

        // Mark as loading
        for (const n of nodesToExpand) {
            loadingTokenChildren.add(n.tokenKey);
        }

        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: treeData.prompt,
                nodes: nodesToExpand.map(n => ({ path: n.path, token_id: n.token_id })),
                k: initialK + 1  // +1 for separator visibility when this becomes depth 0
            })
        });

        const data = await response.json();
        if (response.ok) {
            // Merge children into the tree
            const childrenMap = data.children_map;

            for (const nodeInfo of nodesToExpand) {
                if (childrenMap[nodeInfo.tokenKey]) {
                    const newChildren = childrenMap[nodeInfo.tokenKey];
                    for (const child of newChildren) {
                        child.cumulative_prob = nodeInfo.node.cumulative_prob * child.probability;
                    }

                    if (!nodeInfo.node.children) {
                        nodeInfo.node.children = newChildren;
                    } else {
                        const existingIds = new Set(nodeInfo.node.children.map(c => c.token_id));
                        for (const child of newChildren) {
                            if (!existingIds.has(child.token_id)) {
                                nodeInfo.node.children.push(child);
                            }
                        }
                    }
                }
            }

            console.log(`✓ Loaded depths for page ${pageNum}`);
            loadedPageDepths.add(pageNum);
            renderLevels();
        }
    } catch (error) {
        console.error(`Error loading depths for page ${pageNum}:`, error);
    } finally {
        loadingPageDepths.delete(pageNum);
        // Clear token loading state
        for (let i = pageStart; i < pageEnd && i < treeData.children.length; i++) {
            const node = treeData.children[i];
            if (node) loadingTokenChildren.delete(node.token_id.toString());
        }
    }
}

async function loadNextPage(pageNum) {
    // Load the next page's first layer tokens (+1 to include first token of following page for separator)
    const newTotalK = (pageNum + 1) * initialK + 1;

    console.log(`[loadNextPage] pageNum=${pageNum}, newTotalK=${newTotalK}, totalK=${totalK}, loadingNextPage=${loadingNextPage}`);

    // Skip if we already have enough tokens
    if (newTotalK <= totalK) {
        console.log(`[loadNextPage] SKIP: already have enough tokens`);
        return;
    }

    // If already loading, wait for that load to complete instead of starting a new one
    if (loadingNextPage && loadNextPagePromise) {
        console.log(`[loadNextPage] Already loading, waiting for current load to complete...`);
        await loadNextPagePromise;
        // After waiting, recheck if we need to load more
        if (newTotalK <= totalK) {
            console.log(`[loadNextPage] After wait: already have enough tokens`);
            return;
        }
    }

    loadingNextPage = true;

    console.log(`[loadNextPage] Loading page ${pageNum} tokens (requesting k=${newTotalK})...`);

    // Create a promise that will resolve when this load completes
    let resolveLoadPromise;
    loadNextPagePromise = new Promise(resolve => { resolveLoadPromise = resolve; });

    try {
        const response = await fetch('/api/beam-tree', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: originalPrompt,
                k: newTotalK,
                n: 1  // First layer only
            })
        });

        const newData = await response.json();
        if (response.ok) {
            // Add new tokens
            for (let i = totalK; i < newData.children.length; i++) {
                treeData.children[i] = newData.children[i];
            }

            // Set totalK to actual tokens returned, not requested (API may return fewer)
            totalK = Math.max(totalK, newData.children.length);
            renderLevels(); // Always re-render after loading new data

            console.log(`✓ Loaded page ${pageNum} first layer (totalK now ${totalK})`);
        }
    } catch (error) {
        console.error(`Error loading page ${pageNum}:`, error);
    }

    loadingNextPage = false;
    loadNextPagePromise = null;
    resolveLoadPromise(); // Signal that load is complete
}

// Load more children for the current node when navigating at deeper layers
async function loadMoreChildrenForCurrentNode() {
    if (currentPath.length === 0) return;

    // If already loading, wait for that load to complete
    if (loadingMoreChildren && loadMoreChildrenPromise) {
        console.log(`[loadMoreChildrenForCurrentNode] Already loading, waiting...`);
        await loadMoreChildrenPromise;
        return; // After waiting, the data should be available
    }

    loadingMoreChildren = true;

    // Create a promise that will resolve when this load completes
    let resolveLoadPromise;
    loadMoreChildrenPromise = new Promise(resolve => { resolveLoadPromise = resolve; });

    // Get the current node (the one whose children we're navigating)
    let currentNode = treeData.children[currentPath[0].index];
    for (let i = 1; i < currentPath.length; i++) {
        if (!currentNode || !currentNode.children) break;
        currentNode = currentNode.children[currentPath[i].index];
    }

    if (!currentNode || !currentNode.children) {
        loadingMoreChildren = false;
        loadMoreChildrenPromise = null;
        resolveLoadPromise();
        return;
    }

    const currentChildCount = currentNode.children.length;
    const targetChildCount = currentChildCount + initialK + 1; // Load next page + 1

    console.log(`Loading more children for current node (${currentChildCount} -> ${targetChildCount})...`);

    try {
        // Build the path for the API call
        const pathTokenIds = currentPath.map(p => p.token_id);

        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: treeData.prompt,
                nodes: [{
                    path: pathTokenIds.slice(0, -1), // Parent path
                    token_id: pathTokenIds[pathTokenIds.length - 1] // Current node's token_id
                }],
                k: targetChildCount // Request this many children total
            })
        });

        const data = await response.json();
        if (response.ok) {
            const childrenMap = data.children_map;
            const pathKey = pathTokenIds.join(',');

            if (childrenMap[pathKey]) {
                const newChildren = childrenMap[pathKey];

                // Merge: add any children we don't already have
                const existingIds = new Set(currentNode.children.map(c => c.token_id));
                for (const child of newChildren) {
                    if (!existingIds.has(child.token_id)) {
                        child.cumulative_prob = (currentNode.cumulative_prob || currentNode.probability) * child.probability;
                        currentNode.children.push(child);
                    }
                }

                console.log(`✓ Loaded more children (now ${currentNode.children.length})`);
                renderLevels(); // Always re-render after loading new data

                // Also load depth for the newly loaded children
                topUpCurrentLevelChildren();
            }
        }
    } catch (error) {
        console.error('Error loading more children:', error);
    }

    loadingMoreChildren = false;
    loadMoreChildrenPromise = null;
    resolveLoadPromise(); // Signal that load is complete
}

// Top up children for the current level's tokens to k+1
async function topUpCurrentLevelChildren() {
    const nodes = getCurrentNodes();
    if (!nodes || nodes.length === 0) return;

    // Find nodes that need children topped up on CURRENT PAGE (not just first k+1)
    const nodesToExpand = [];
    const pathTokenIds = currentPath.map(p => p.token_id);
    const pathPrefix = pathTokenIds.join(',');

    // Calculate which tokens to check based on current page
    const pageStart = currentPage * initialK;
    const pageEnd = Math.min(pageStart + initialK + 1, nodes.length); // +1 for separator token

    for (let i = pageStart; i < pageEnd; i++) {
        const node = nodes[i];
        if (!node) continue;

        const tokenKey = pathPrefix ? `${pathPrefix},${node.token_id}` : `${node.token_id}`;

        // Skip if already loading or already has enough children
        if (loadingTokenChildren.has(tokenKey)) continue;
        if (node.children && node.children.length >= initialK + 1) continue;

        nodesToExpand.push({
            path: pathTokenIds,
            token_id: node.token_id,
            tokenKey: tokenKey,
            node: node
        });
    }

    if (nodesToExpand.length === 0) return;

    // Mark as loading
    for (const n of nodesToExpand) {
        loadingTokenChildren.add(n.tokenKey);
    }

    console.log(`Topping up ${nodesToExpand.length} nodes' children to k+1...`);

    try {
        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: treeData.prompt,
                nodes: nodesToExpand.map(n => ({ path: n.path, token_id: n.token_id })),
                k: initialK + 1
            })
        });

        const data = await response.json();
        if (response.ok) {
            const childrenMap = data.children_map;

            for (const nodeInfo of nodesToExpand) {
                if (childrenMap[nodeInfo.tokenKey]) {
                    const newChildren = childrenMap[nodeInfo.tokenKey];
                    for (const child of newChildren) {
                        child.cumulative_prob = (nodeInfo.node.cumulative_prob || nodeInfo.node.probability) * child.probability;
                    }

                    if (!nodeInfo.node.children) {
                        nodeInfo.node.children = newChildren;
                    } else {
                        const existingIds = new Set(nodeInfo.node.children.map(c => c.token_id));
                        for (const child of newChildren) {
                            if (!existingIds.has(child.token_id)) {
                                nodeInfo.node.children.push(child);
                            }
                        }
                    }
                }
            }

            console.log(`✓ Topped up children to k+1`);
            renderLevels();
        }
    } catch (error) {
        console.error('Error topping up children:', error);
    } finally {
        // Clear loading state
        for (const n of nodesToExpand) {
            loadingTokenChildren.delete(n.tokenKey);
        }
    }
}

// Debounced depth loading - collects all needed tokens and loads in one batch
function ensureCurrentTokenHasDepth() {
    // Cancel any pending depth load and reschedule
    if (pendingDepthLoadTimeout) {
        clearTimeout(pendingDepthLoadTimeout);
    }
    // Debounce by 50ms to batch rapid navigation requests
    pendingDepthLoadTimeout = setTimeout(() => {
        pendingDepthLoadTimeout = null;
        doEnsureCurrentTokenHasDepth();
    }, 50);
}

// Actual depth loading implementation
async function doEnsureCurrentTokenHasDepth() {
    const nodes = getCurrentNodes();
    if (!nodes || nodes.length === 0) return;

    const pathTokenIds = currentPath.map(p => p.token_id);
    const pathPrefix = pathTokenIds.join(',');

    // Check current token and adjacent tokens (for smooth scrolling)
    const indicesToCheck = [
        currentIndex - 1,
        currentIndex,
        currentIndex + 1,
        currentIndex + 2
    ].filter(i => i >= 0 && i < nodes.length);

    const nodesToLoad = [];
    for (const idx of indicesToCheck) {
        const node = nodes[idx];
        if (!node) continue;

        // Create a unique key for this token
        const tokenKey = pathPrefix ? `${pathPrefix},${node.token_id}` : `${node.token_id}`;

        // Skip if already loading or already has enough children
        if (loadingTokenChildren.has(tokenKey)) continue;
        if (node.children && node.children.length >= initialK + 1) continue;

        nodesToLoad.push({
            path: pathTokenIds,
            token_id: node.token_id,
            node: node,
            tokenKey: tokenKey
        });
    }

    if (nodesToLoad.length > 0) {
        // Mark these tokens as loading
        for (const n of nodesToLoad) {
            loadingTokenChildren.add(n.tokenKey);
        }

        console.log(`Loading children for ${nodesToLoad.length} tokens...`);

        try {
            const response = await fetch('/api/expand-depth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: treeData.prompt,
                    nodes: nodesToLoad.map(n => ({ path: n.path, token_id: n.token_id })),
                    k: initialK + 1
                })
            });

            const data = await response.json();
            if (response.ok) {
                const childrenMap = data.children_map;

                for (const nodeInfo of nodesToLoad) {
                    const pathKey = [...nodeInfo.path, nodeInfo.token_id].join(',');

                    if (childrenMap[pathKey]) {
                        const newChildren = childrenMap[pathKey];
                        for (const child of newChildren) {
                            child.cumulative_prob = (nodeInfo.node.cumulative_prob || nodeInfo.node.probability) * child.probability;
                        }

                        if (!nodeInfo.node.children) {
                            nodeInfo.node.children = newChildren;
                        } else {
                            const existingIds = new Set(nodeInfo.node.children.map(c => c.token_id));
                            for (const child of newChildren) {
                                if (!existingIds.has(child.token_id)) {
                                    nodeInfo.node.children.push(child);
                                }
                            }
                        }
                    }
                }

                console.log(`✓ Loaded children for ${nodesToLoad.length} tokens`);
                renderLevels();
            }
        } catch (error) {
            console.error('Error loading children for tokens:', error);
        } finally {
            // Clear loading state
            for (const n of nodesToLoad) {
                loadingTokenChildren.delete(n.tokenKey);
            }
        }
    }

    // After ensuring current token has children, load preview layers
    await ensurePreviewLayersLoaded();
}

// Load preview layers for the current token (single pass)
// Only loads children for nodes we can reach that don't have children yet
async function ensurePreviewLayersLoaded() {
    const nodes = getCurrentNodes();
    if (!nodes || nodes.length === 0) return;
    if (currentIndex >= nodes.length) return;

    const currentNode = nodes[currentIndex];
    if (!currentNode || !currentNode.children || currentNode.children.length === 0) return;

    // Collect all nodes in the preview chain that need children loaded
    const nodesToExpand = [];
    const basePath = currentPath.map(p => p.token_id);

    // Walk down the preview chain (first child at each level)
    let previewNode = currentNode;
    let previewPath = [...basePath, currentNode.token_id];

    for (let depth = 0; depth < 4; depth++) {
        if (!previewNode.children || previewNode.children.length === 0) break;

        const firstChild = previewNode.children[0];
        if (!firstChild) break;

        const tokenKey = [...previewPath, firstChild.token_id].join(',');

        // Check if first child needs children loaded and isn't already loading
        if (!loadingTokenChildren.has(tokenKey) &&
            (!firstChild.children || firstChild.children.length < initialK + 1)) {
            nodesToExpand.push({
                path: previewPath,
                token_id: firstChild.token_id,
                tokenKey: tokenKey,
                node: firstChild
            });
        }

        // Move to next level
        previewPath = [...previewPath, firstChild.token_id];
        previewNode = firstChild;
    }

    if (nodesToExpand.length === 0) return;

    // Mark as loading
    for (const n of nodesToExpand) {
        loadingTokenChildren.add(n.tokenKey);
    }

    try {
        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: treeData.prompt,
                nodes: nodesToExpand.map(n => ({ path: n.path, token_id: n.token_id })),
                k: initialK + 1
            })
        });

        const data = await response.json();
        if (response.ok) {
            const childrenMap = data.children_map;

            for (const nodeInfo of nodesToExpand) {
                if (childrenMap[nodeInfo.tokenKey]) {
                    const newChildren = childrenMap[nodeInfo.tokenKey];
                    for (const child of newChildren) {
                        child.cumulative_prob = (nodeInfo.node.cumulative_prob || nodeInfo.node.probability) * child.probability;
                    }

                    if (!nodeInfo.node.children) {
                        nodeInfo.node.children = newChildren;
                    } else {
                        const existingIds = new Set(nodeInfo.node.children.map(c => c.token_id));
                        for (const child of newChildren) {
                            if (!existingIds.has(child.token_id)) {
                                nodeInfo.node.children.push(child);
                            }
                        }
                    }
                }
            }

            renderLevels();
        }
    } catch (error) {
        console.error('Error loading preview layers:', error);
    } finally {
        // Clear loading state
        for (const n of nodesToExpand) {
            loadingTokenChildren.delete(n.tokenKey);
        }
    }
}

async function loadNextLayer(nodes, pathToNodes) {
    if (isLoadingNextLayer) return; // Prevent cascading
    isLoadingNextLayer = true;

    try {
        // Collect all nodes that need children loaded
        const nodesToExpand = nodes.map(node => ({
            path: pathToNodes.map(p => p.token_id),
            token_id: node.token_id
        }));

        const response = await fetch('/api/expand-depth', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: treeData.prompt,  // expand-depth needs full prompt with special tokens
                nodes: nodesToExpand,
                k: initialK + 1  // +1 for separator visibility when this becomes depth 0
            })
        });

        const data = await response.json();

        if (!response.ok) {
            console.error('Failed to load depth:', data.error);
            isLoadingNextLayer = false;
            return;
        }

        // Merge children into the tree
        const childrenMap = data.children_map;

        for (const node of nodes) {
            // Only set children if node doesn't already have them
            if (node.children) continue;

            const pathKey = [...pathToNodes.map(p => p.token_id), node.token_id].join(',');
            if (childrenMap[pathKey]) {
                // Fix cumulative probability for loaded children
                const children = childrenMap[pathKey];
                for (const child of children) {
                    child.cumulative_prob = node.cumulative_prob * child.probability;
                }
                node.children = children;
            }
        }

        isLoadingNextLayer = false;
        // Re-render to show newly loaded layers (unless animating)
        renderLevels(); // Always re-render after loading new data
    } catch (error) {
        console.error('Error loading depth:', error);
        isLoadingNextLayer = false;
    }
}
