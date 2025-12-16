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
        } else {
            loadMoreChildrenForCurrentNode();
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

    // Start loading next page's tokens immediately (in parallel with animation)
    // This ensures the separator and first token of the following page are ready
    let loadPromise;
    if (isRootLevel) {
        const nextPageToLoad = toPage + 1;
        loadPromise = loadNextPage(nextPageToLoad);
    } else {
        // For deeper levels, load more children for the current node
        loadPromise = loadMoreChildrenForCurrentNode();
    }

    // Delay rendering until after the animation completes (400ms)
    animationTimeoutId = setTimeout(async () => {
        animationTimeoutId = null;
        isAnimating = false; // Animation complete, allow re-renders

        if (isRootLevel) {
            // Wait for both depth loading and next page tokens to complete
            await Promise.all([
                loadPageDepths(toPage),
                loadPromise
            ]);
        } else {
            // For deeper levels, wait for children load then top up
            await loadPromise;
            await topUpCurrentLevelChildren();
        }
        renderLevels(); // Re-render now that data is loaded

        // Ensure current token has depth (important for deeper levels)
        ensureCurrentTokenHasDepth();

        if (isRootLevel) {
            // Preload full depth for next page in background (for smooth future navigation)
            const nextPageToLoad = toPage + 1;
            preloadPageFullDepth(nextPageToLoad);
        }
    }, 420); // Slightly longer than the 400ms animation
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

    // Top up the new level's children to k+1 for separator visibility
    topUpCurrentLevelChildren();

    // Ensure the new current token (at the deeper level) has depth loaded
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
