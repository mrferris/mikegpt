// drive-state.js - Global state variables and configuration

// Check for light mode from URL params
const urlParams = new URLSearchParams(window.location.search);
const isLightMode = urlParams.get('mode') === 'imessage';

if (isLightMode) {
    document.body.classList.add('light-mode');
}

// Core data
let treeData = null;
let originalPrompt = null; // Store the original user prompt (without special tokens)

// Navigation state
let currentPath = [];
let currentIndex = 0;
let currentPage = 0; // Current page of tokens being viewed (0-indexed)

// Selection and training state
let selectedPaths = [];
let trainingHistory = [];

// Configuration
let initialK = 5; // Policy boundary (user's k value)
let totalK = 5; // Total tokens currently visible
let initialN = 4; // Initial N value (static)
let samplingMethod = 'topk'; // 'topk' or 'topp'
let rlMethod = 'dpo'; // 'dpo' or 'grpo'

// Loading state tracking
let loadedDepths = new Set(); // Track which depth layers we've already loaded
let loadedPageDepths = new Set(); // Track which pages have COMPLETED depth loading
let loadingPageDepths = new Set(); // Track which pages are CURRENTLY loading depths
let loadedPages = new Set(); // Track which pages have been loaded
let loadingTokenChildren = new Set(); // Track token paths currently being loaded (prevents duplicate requests)

// Animation and loading flags
let isLoadingNextLayer = false; // Prevent cascading loads
let isAnimating = false; // Prevent rebuilds during page transition animation
let animationTimeoutId = null; // Track animation timeout for cancellation
let pendingDepthLoadTimeout = null; // Debounce timer for depth loading

// Page loading state
let loadingNextPage = false;
let loadNextPagePromise = null; // Promise that resolves when current load completes

// Children loading state
let loadingMoreChildren = false;
let loadMoreChildrenPromise = null; // Promise that resolves when current load completes
