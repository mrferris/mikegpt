// drive-navigation.js - Navigation functions and tree traversal utilities

function collectNodesNeedingChildren(node, pathSoFar, currentDepth, targetDepth, result) {
    if (!node) return;

    if (currentDepth === targetDepth - 1) {
        // We're at the parent level - collect children that need their own children loaded
        if (node.children) {
            for (const child of node.children) {
                if (!child) continue;
                const tokenKey = [...pathSoFar, child.token_id].join(',');
                // Skip if already loading or already has children
                if (loadingTokenChildren.has(tokenKey)) continue;
                if (child.children) continue;

                result.push({
                    path: pathSoFar,
                    token_id: child.token_id,
                    tokenKey: tokenKey,
                    node: child
                });
            }
        }
        return;
    }

    // Go deeper
    if (node.children) {
        for (const child of node.children) {
            if (child) {
                collectNodesNeedingChildren(child, [...pathSoFar, child.token_id], currentDepth + 1, targetDepth, result);
            }
        }
    }
}

function findNodeByPath(tokenIds) {
    if (tokenIds.length === 0) return null;

    // First token ID is a root token
    let node = treeData.children.find(c => c && c.token_id === tokenIds[0]);

    for (let i = 1; i < tokenIds.length && node; i++) {
        if (!node.children) return null;
        node = node.children.find(c => c && c.token_id === tokenIds[i]);
    }

    return node;
}

function getCurrentNodes() {
    let nodes = treeData.children;
    for (const selection of currentPath) {
        if (!nodes || !nodes[selection.index]) return [];
        nodes = nodes[selection.index].children;
        if (!nodes) break;
    }
    return nodes || [];
}

function getCurrentFullPath() {
    let path = treeData.prompt;
    let nodes = treeData.children;

    for (const selection of currentPath) {
        if (!nodes || !nodes[selection.index]) break;
        path += nodes[selection.index].token_str;
        nodes = nodes[selection.index].children;
    }

    // Add current selection
    const currentNodes = getCurrentNodes();
    if (currentNodes && currentNodes[currentIndex]) {
        path += currentNodes[currentIndex].token_str;
    }

    return path;
}

function getCurrentTokenIds() {
    // Collect the token IDs from the current path
    let tokenIds = [];
    let nodes = treeData.children;

    for (const selection of currentPath) {
        if (!nodes || !nodes[selection.index]) break;
        tokenIds.push(nodes[selection.index].token_id);
        nodes = nodes[selection.index].children;
    }

    // Add current selection
    const currentNodes = getCurrentNodes();
    if (currentNodes && currentNodes[currentIndex]) {
        tokenIds.push(currentNodes[currentIndex].token_id);
    }

    return tokenIds;
}

function navigateLeft() {
    const nodes = getCurrentNodes();
    if (nodes.length === 0 || totalK === 0) return;

    // Don't cycle - stop at beginning
    if (currentIndex <= 0) return;

    const oldPage = currentPage;
    currentIndex = currentIndex - 1;
    currentPage = Math.floor(currentIndex / initialK);

    if (currentPage !== oldPage) {
        animatePageTransition(oldPage, currentPage);
    } else {
        renderLevels();
    }
    updatePathDisplay();

    // Ensure the new current token has depth loaded
    ensureCurrentTokenHasDepth();
}

async function navigateRight() {
    const nodes = getCurrentNodes();
    console.log(`[navigateRight] nodes.length=${nodes.length}, currentIndex=${currentIndex}, totalK=${totalK}, currentPath.length=${currentPath.length}`);
    if (nodes.length === 0) return;

    const oldPage = currentPage;
    const newIndex = currentIndex + 1;
    const newPage = Math.floor(newIndex / initialK);

    // Different behavior for root level vs deeper levels
    const isRootLevel = currentPath.length === 0;
    let maxLoaded = isRootLevel ? totalK : nodes.length;

    console.log(`[navigateRight] newIndex=${newIndex}, maxLoaded=${maxLoaded}, isRootLevel=${isRootLevel}`);

    // If trying to go to unloaded tokens, trigger load and WAIT for it
    if (newIndex >= maxLoaded) {
        console.log(`[navigateRight] BLOCKED: newIndex >= maxLoaded, triggering load and waiting`);
        if (isRootLevel) {
            await loadNextPage(newPage);
            // Recheck maxLoaded after load completes
            maxLoaded = totalK;
        } else {
            await loadMoreChildrenForCurrentNode();
            // Recheck maxLoaded after load completes
            const updatedNodes = getCurrentNodes();
            maxLoaded = updatedNodes.length;
        }

        // If still can't advance, return (data might not be available yet)
        if (newIndex >= maxLoaded) {
            console.log(`[navigateRight] Still blocked after load: newIndex=${newIndex}, maxLoaded=${maxLoaded}`);
            return;
        }
    }

    currentIndex = newIndex;
    currentPage = newPage;

    // Preload next page when approaching end of current page (fire-and-forget)
    if (currentIndex % initialK >= initialK - 2) {
        if (isRootLevel) {
            loadNextPage(currentPage + 1);
            preloadPageFullDepth(currentPage + 1); // start full depth preload early
        } else {
            loadMoreChildrenForCurrentNode();
            preloadDeeperLevelFullDepth(); // start full depth preload early
        }
    }

    if (currentPage !== oldPage) {
        animatePageTransition(oldPage, currentPage);
    } else {
        renderLevels();
    }
    updatePathDisplay();

    // Ensure the new current token has depth loaded
    ensureCurrentTokenHasDepth();
}

function animatePageTransition(fromPage, toPage) {
    // Cancel any previous animation timeout
    if (animationTimeoutId) {
        clearTimeout(animationTimeoutId);
        animationTimeoutId = null;
    }

    isAnimating = true; // Prevent rebuilds during animation
    renderLevels(fromPage);

    const isRootLevel = currentPath.length === 0;

    // Fire ALL data loading immediately — not in the animation timeout.
    // Previously, preloadPageFullDepth lived inside the 420ms timeout,
    // which got cancelled during fast scrolling. Now preloading is
    // decoupled from the animation lifecycle entirely.
    let deeperPreloadPromise = null;
    if (isRootLevel) {
        loadNextPage(toPage + 1);         // first layer for page after next
        preloadPageFullDepth(toPage);     // full depth (1-4) for target page
        preloadPageFullDepth(toPage + 1); // full depth for page after next
    } else {
        loadMoreChildrenForCurrentNode();
        // Fire full depth preload immediately — loads depths 1-4 for ALL page
        // tokens by walking the first-child preview chain. O(pageSize × 3) nodes.
        deeperPreloadPromise = preloadDeeperLevelFullDepth();
    }

    // Animation timeout: unlock rendering and do a final render.
    animationTimeoutId = setTimeout(async () => {
        animationTimeoutId = null;
        isAnimating = false;

        // At deeper levels, wait for the full depth preload to finish before
        // rendering so all 4 layers appear at once (no pop-in).
        if (!isRootLevel && deeperPreloadPromise) {
            await deeperPreloadPromise;
        }
        renderLevels();
        ensureCurrentTokenHasDepth();
    }, 420);
}

function selectAndAdvance() {
    const currentNodes = getCurrentNodes();
    console.log(`[selectAndAdvance] currentNodes.length=${currentNodes?.length}, currentIndex=${currentIndex}`);
    if (!currentNodes || currentNodes.length === 0) return;
    if (currentIndex >= currentNodes.length) return;

    const selected = currentNodes[currentIndex];
    console.log(`[selectAndAdvance] selected.children?.length=${selected?.children?.length}`);
    if (!selected.children || selected.children.length === 0) {
        console.log(`[selectAndAdvance] BLOCKED: no children, triggering load`);
        // No children yet - trigger load and wait
        ensureCurrentTokenHasDepth();
        return;
    }

    currentPath.push({
        index: currentIndex,
        token: selected.token_str,
        probability: selected.probability,
        token_id: selected.token_id
    });

    currentIndex = 0;
    currentPage = 0; // Reset to first page when going deeper
    renderLevels();
    updatePathDisplay();

    // Preload all 4 depth layers for the new deeper level's first page.
    // This replaces the old topUpCurrentLevelChildren + ensureCurrentTokenHasDepth
    // combo which only loaded depth 1 for page tokens and depth 1-4 for the
    // current token. Now ALL page tokens get full depth preloaded.
    preloadDeeperLevelFullDepth();

    // Also ensure the current token's preview chain is loaded (covers edge cases)
    ensureCurrentTokenHasDepth();
}

function goBack() {
    if (currentPath.length === 0) return;

    const last = currentPath.pop();
    currentIndex = last.index;
    currentPage = Math.floor(currentIndex / initialK);
    renderLevels();
    updatePathDisplay();
}
