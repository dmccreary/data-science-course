// Global variables
let network = null;
let allNodes = null;
let allEdges = null;
let graphData = null;

// Category configuration with colors and labels
const categories = {
    'found': { label: 'Foundational Concepts', color: 'red', textColor: 'white' },
    'python': { label: 'Python Programming', color: 'gold', textColor: 'black' },
    'libs': { label: 'Python Libraries', color: 'green', textColor: 'white' },
    'data': { label: 'Data Manipulation', color: 'blue', textColor: 'white' },
    'stats': { label: 'Statistical Analysis', color: 'purple', textColor: 'white' },
    'ml': { label: 'Machine Learning', color: 'pink', textColor: 'black' },
    'vis': { label: 'Data Visualization', color: 'gray', textColor: 'white' },
    'nlp': { label: 'Natural Language Processing', color: 'brown', textColor: 'white' },
    'genai': { label: 'Generative AI', color: 'teal', textColor: 'white' },
    'proj': { label: 'Projects', color: 'cyan', textColor: 'black' },
    'soft': { label: 'Soft Skills', color: 'lightgreen', textColor: 'black' },
    'goal': { label: 'Learning Goals', color: 'gold', textColor: 'black' }
};

// Initialize the graph
function initGraph() {
    fetch('../learning-graph/data-science.json')
        .then(response => response.json())
        .then(data => {
            graphData = data;

            // Process nodes
            const nodes = data.nodes.map(node => {
                const processed = { ...node };
                if (node.group === "found") {
                    processed.x = -1200;
                    processed.fixed = { x: true, y: false };
                    processed.shape = "box";
                    processed.color = "red";
                    processed.font = { color: "white" };
                } else if (node.group === "goal") {
                    processed.x = 1200;
                    processed.fixed = { x: true, y: false };
                    processed.shape = "star";
                    processed.color = "gold";
                }
                return processed;
            });

            allNodes = new vis.DataSet(nodes);
            allEdges = new vis.DataSet(data.edges);

            // Create the network
            const container = document.getElementById('mynetwork');
            const networkData = {
                nodes: allNodes,
                edges: allEdges
            };

            const options = {
                physics: {
                    enabled: true,
                    solver: 'forceAtlas2Based',
                    stabilization: {
                        iterations: 1000,
                        updateInterval: 25
                    }
                },
                layout: {
                    improvedLayout: false
                },
                edges: {
                    arrows: {
                        to: { enabled: true, type: 'arrow' }
                    },
                    smooth: { type: 'continuous' }
                },
                nodes: {
                    shape: 'dot',
                    size: 16,
                    font: { size: 20, color: 'black' },
                    borderWidth: 2,
                    borderWidthSelected: 4
                },
                groups: {
                    foundation: { shape: "box", color: { background: 'red' }, font: { color: "white" } },
                    python: { color: { background: 'gold' } },
                    libs: { color: { background: 'green' } },
                    ml: { color: { background: 'pink' } },
                    nlp: { color: { background: 'brown' } },
                    data: { color: { background: 'blue' } },
                    proj: { color: { background: 'cyan' } },
                    soft: { color: { background: 'lightgreen' } },
                    stats: { color: { background: 'purple' } },
                    vis: { color: { background: 'gray' } },
                    genai: { color: { background: 'teal' } },
                    goal: { shape: "star", color: { background: 'gold' }, font: { size: 16 } }
                }
            };

            network = new vis.Network(container, networkData, options);

            // Build category filters
            buildCategoryFilters();

            // Setup search
            setupSearch();

            // Update statistics
            updateStats();
        })
        .catch(error => {
            console.error("Error loading graph data:", error);
        });
}

// Build category filter checkboxes
function buildCategoryFilters() {
    const filtersContainer = document.getElementById('category-filters');
    const usedCategories = new Set();

    // Find which categories are actually used
    allNodes.forEach(node => {
        if (node.group) {
            usedCategories.add(node.group);
        }
    });

    // Create checkboxes for used categories
    usedCategories.forEach(groupKey => {
        const cat = categories[groupKey] || { label: groupKey, color: '#999', textColor: 'white' };

        const filterItem = document.createElement('div');
        filterItem.className = 'filter-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `filter-${groupKey}`;
        checkbox.checked = true;
        checkbox.addEventListener('change', applyFilters);

        const label = document.createElement('label');
        label.htmlFor = `filter-${groupKey}`;
        label.innerHTML = `<span class="color-box" style="background-color: ${cat.color};"></span> ${cat.label}`;

        filterItem.appendChild(checkbox);
        filterItem.appendChild(label);
        filtersContainer.appendChild(filterItem);
    });
}

// Apply category filters
function applyFilters() {
    const checkedCategories = new Set();

    document.querySelectorAll('#category-filters input[type="checkbox"]').forEach(cb => {
        if (cb.checked) {
            checkedCategories.add(cb.id.replace('filter-', ''));
        }
    });

    // Update node visibility
    const updates = [];
    allNodes.forEach(node => {
        const shouldShow = checkedCategories.has(node.group);
        updates.push({ id: node.id, hidden: !shouldShow });
    });
    allNodes.update(updates);

    // Update edge visibility based on connected nodes
    const edgeUpdates = [];
    allEdges.forEach(edge => {
        const fromNode = allNodes.get(edge.from);
        const toNode = allNodes.get(edge.to);
        const shouldShow = fromNode && toNode && !fromNode.hidden && !toNode.hidden;
        edgeUpdates.push({ id: edge.id, hidden: !shouldShow });
    });
    allEdges.update(edgeUpdates);

    updateStats();
}

// Check all categories
function checkAll() {
    document.querySelectorAll('#category-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = true;
    });
    applyFilters();
}

// Uncheck all categories
function uncheckAll() {
    document.querySelectorAll('#category-filters input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    applyFilters();
}

// Setup search functionality
function setupSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase().trim();
        searchResults.innerHTML = '';

        if (query.length < 2) {
            searchResults.style.display = 'none';
            return;
        }

        const matches = [];
        allNodes.forEach(node => {
            if (node.label && node.label.toLowerCase().includes(query)) {
                matches.push(node);
            }
        });

        if (matches.length > 0) {
            matches.slice(0, 10).forEach(node => {
                const item = document.createElement('div');
                item.className = 'search-result-item';
                const cat = categories[node.group] || { label: node.group };
                item.innerHTML = `<strong>${node.label}</strong><br><small>${cat.label}</small>`;
                item.addEventListener('click', () => {
                    selectNode(node.id);
                    searchResults.style.display = 'none';
                    searchInput.value = node.label;
                });
                searchResults.appendChild(item);
            });
            searchResults.style.display = 'block';
        } else {
            searchResults.style.display = 'none';
        }
    });

    // Close search results when clicking outside
    document.addEventListener('click', function(e) {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });
}

// Select and focus on a node
function selectNode(nodeId) {
    network.selectNodes([nodeId]);
    network.focus(nodeId, {
        scale: 1.5,
        animation: {
            duration: 500,
            easingFunction: 'easeInOutQuad'
        }
    });
}

// Update statistics
function updateStats() {
    let visibleNodes = 0;
    let visibleEdges = 0;
    let orphanNodes = 0;

    // Count visible nodes
    allNodes.forEach(node => {
        if (!node.hidden) {
            visibleNodes++;
        }
    });

    // Count visible edges
    allEdges.forEach(edge => {
        if (!edge.hidden) {
            visibleEdges++;
        }
    });

    // Count orphan nodes (nodes with no connections)
    const connectedNodes = new Set();
    allEdges.forEach(edge => {
        if (!edge.hidden) {
            connectedNodes.add(edge.from);
            connectedNodes.add(edge.to);
        }
    });

    allNodes.forEach(node => {
        if (!node.hidden && !connectedNodes.has(node.id)) {
            orphanNodes++;
        }
    });

    document.getElementById('visible-nodes').textContent = visibleNodes;
    document.getElementById('visible-edges').textContent = visibleEdges;
    document.getElementById('orphan-nodes').textContent = orphanNodes;
}

// Toggle sidebar visibility
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const mainContent = document.getElementById("main");
    const toggleButton = document.getElementById("toggle-button");

    if (sidebar.style.display === "none") {
        sidebar.style.display = "block";
        toggleButton.innerHTML = "&#9776;";
        mainContent.style.marginLeft = "auto";
    } else {
        sidebar.style.display = "none";
        toggleButton.innerHTML = "&#8594;";
        mainContent.style.marginLeft = "0";
    }
}
