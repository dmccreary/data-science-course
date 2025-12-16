// Chapter 1 Concept Map
// Interactive learning map showing relationships between all Chapter 1 concepts

// Node category definitions with colors and shapes
const categories = {
    core: {
        name: 'Core Concepts',
        color: '#FFD700',
        border: '#B8860B',
        shape: 'circle',
        size: 35,
        font: { color: '#333', size: 14 }
    },
    dataType: {
        name: 'Data Types',
        color: '#4A90D9',
        border: '#2E5B8A',
        shape: 'box',
        size: 25,
        font: { color: '#333', size: 12 }
    },
    variableRole: {
        name: 'Variable Roles',
        color: '#27AE60',
        border: '#1E8449',
        shape: 'diamond',
        size: 25,
        font: { color: '#333', size: 11 }
    },
    workflow: {
        name: 'Workflow',
        color: '#9B59B6',
        border: '#7D3C98',
        shape: 'hexagon',
        size: 25,
        font: { color: '#333', size: 11 }
    },
    tools: {
        name: 'Tools & Practices',
        color: '#E67E22',
        border: '#BA6318',
        shape: 'box',
        size: 22,
        font: { color: '#333', size: 11 }
    },
    structure: {
        name: 'Structure',
        color: '#17A2B8',
        border: '#117A8B',
        shape: 'triangle',
        size: 25,
        font: { color: '#333', size: 11 }
    }
};

// Edge type styles
const edgeStyles = {
    isTypeOf: { color: '#666', dashes: false, width: 2 },
    contains: { color: '#4A90D9', dashes: [5, 5], width: 2 },
    uses: { color: '#27AE60', dashes: [2, 2], width: 2 },
    sameAs: { color: '#E74C3C', dashes: false, width: 3 }
};

// Global variables
let nodes, edges, network;
let selectedNode = null;
let nodeData = [];
let edgeData = [];
let examples = {};
let editMode = false;

// Check URL parameters for edit mode
function checkEditMode() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('edit') === 'true';
}

// Environment detection
function isInIframe() {
    try {
        return window.self !== window.top;
    } catch (e) {
        return true;
    }
}

// Load data from JSON file
async function loadData() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        nodeData = data.nodes;
        edgeData = data.edges;
        examples = data.examples;
        return true;
    } catch (error) {
        console.error('Error loading data:', error);
        return false;
    }
}

// Create vis.js node from data
function createVisNode(node) {
    const cat = categories[node.category];
    return {
        id: node.id,
        label: node.label,
        x: node.x,
        y: node.y,
        shape: cat.shape,
        size: cat.size,
        color: {
            background: cat.color,
            border: cat.border,
            highlight: {
                background: cat.color,
                border: '#333'
            }
        },
        font: cat.font,
        borderWidth: 3,
        shadow: true,
        title: node.definition,
        category: node.category
    };
}

// Create vis.js edge from data
function createVisEdge(edge, index) {
    const style = edgeStyles[edge.type];
    return {
        id: index,
        from: edge.from,
        to: edge.to,
        label: edge.label,
        arrows: edge.type === 'sameAs' ? { to: false, from: false } : { to: { enabled: true, scaleFactor: 0.8 } },
        color: { color: style.color, highlight: '#E74C3C' },
        width: style.width,
        dashes: style.dashes,
        font: { size: 10, color: '#666', strokeWidth: 2, strokeColor: '#fff' },
        smooth: { type: 'curvedCW', roundness: 0.1 },
        edgeType: edge.type
    };
}

// Initialize the network
function initializeNetwork() {
    const visNodes = nodeData.map(createVisNode);
    const visEdges = edgeData.map(createVisEdge);

    nodes = new vis.DataSet(visNodes);
    edges = new vis.DataSet(visEdges);

    const enableMouseInteraction = !isInIframe();

    const options = {
        layout: { improvedLayout: false },
        physics: { enabled: false },
        interaction: {
            hover: true,
            tooltipDelay: 100,
            zoomView: enableMouseInteraction,
            dragView: enableMouseInteraction,
            dragNodes: editMode,
            navigationButtons: true
        },
        nodes: {
            borderWidth: 3,
            shadow: {
                enabled: true,
                color: 'rgba(0,0,0,0.2)',
                size: 5,
                x: 2,
                y: 2
            }
        },
        edges: {
            smooth: { type: 'curvedCW', roundness: 0.1 }
        }
    };

    const container = document.getElementById('network');
    const data = { nodes: nodes, edges: edges };
    network = new vis.Network(container, data, options);

    // Event handlers
    network.on('click', handleClick);
    network.on('doubleClick', handleDoubleClick);
    network.on('hoverNode', handleHoverNode);
    network.on('blurNode', handleBlurNode);

    setTimeout(positionView, 200);
}

// Position the view
function positionView() {
    if (network) {
        network.moveTo({
            position: { x: 0, y: 50 },
            scale: 0.9,
            animation: false
        });
    }
}

// Handle node click - highlight connected nodes
function handleClick(params) {
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        highlightConnections(nodeId);
        showNodeInfo(nodeId);
    } else {
        resetHighlights();
        hideNodeInfo();
    }
}

// Handle double-click - show examples
function handleDoubleClick(params) {
    if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        showExamples(nodeId);
    }
}

// Handle hover
function handleHoverNode(params) {
    document.body.style.cursor = 'pointer';
}

function handleBlurNode(params) {
    document.body.style.cursor = 'default';
}

// Highlight connected nodes and edges
function highlightConnections(nodeId) {
    const connectedNodes = network.getConnectedNodes(nodeId);
    const connectedEdges = network.getConnectedEdges(nodeId);

    // Dim all nodes
    nodes.forEach(node => {
        if (node.id === nodeId || connectedNodes.includes(node.id)) {
            nodes.update({ id: node.id, opacity: 1 });
        } else {
            nodes.update({ id: node.id, opacity: 0.2 });
        }
    });

    // Dim all edges
    edges.forEach(edge => {
        if (connectedEdges.includes(edge.id)) {
            edges.update({ id: edge.id, color: { color: '#E74C3C' }, width: 3 });
        } else {
            edges.update({ id: edge.id, color: { color: '#ddd' }, width: 1 });
        }
    });

    selectedNode = nodeId;
}

// Reset highlights
function resetHighlights() {
    nodes.forEach(node => {
        nodes.update({ id: node.id, opacity: 1 });
    });

    edgeData.forEach((edge, index) => {
        const style = edgeStyles[edge.type];
        edges.update({
            id: index,
            color: { color: style.color },
            width: style.width
        });
    });

    selectedNode = null;
}

// Show node info in panel
function showNodeInfo(nodeId) {
    const node = nodeData.find(n => n.id === nodeId);
    if (!node) return;

    const infoPanel = document.getElementById('info-content');
    const connectedNodes = network.getConnectedNodes(nodeId);
    const connections = connectedNodes.map(id => {
        const n = nodeData.find(nd => nd.id === id);
        return n ? n.label.replace('\n', ' ') : id;
    }).join(', ');

    infoPanel.innerHTML = `
        <div class="info-title">${node.label.replace('\n', ' ')}</div>
        <div class="info-category">${categories[node.category].name}</div>
        <div class="info-definition">${node.definition}</div>
        <div class="info-connections"><strong>Connected to:</strong> ${connections || 'None'}</div>
    `;
}

// Hide node info
function hideNodeInfo() {
    const infoPanel = document.getElementById('info-content');
    infoPanel.innerHTML = '<div class="info-hint">Click a concept to see details.<br>Double-click for examples.</div>';
}

// Show examples for a concept
function showExamples(nodeId) {
    const exampleList = examples[nodeId] || ['No examples available'];
    const node = nodeData.find(n => n.id === nodeId);

    const infoPanel = document.getElementById('info-content');
    infoPanel.innerHTML = `
        <div class="info-title">${node.label.replace('\n', ' ')}</div>
        <div class="info-subtitle">Examples:</div>
        <ul class="example-list">
            ${exampleList.map(ex => `<li>${ex}</li>`).join('')}
        </ul>
    `;
}

// Save updated node positions to JSON and download
function saveData() {
    // Get current positions from the network
    const positions = network.getPositions();

    // Update nodeData with new positions
    const updatedNodes = nodeData.map(node => {
        const pos = positions[node.id];
        return {
            ...node,
            x: Math.round(pos.x),
            y: Math.round(pos.y)
        };
    });

    // Create the data object
    const dataToSave = {
        nodes: updatedNodes,
        edges: edgeData,
        examples: examples
    };

    // Convert to JSON with nice formatting
    const jsonString = JSON.stringify(dataToSave, null, 4);

    // Create blob and download link
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'data.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show edit mode UI
function showEditModeUI() {
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) {
        saveBtn.style.display = 'block';
        saveBtn.addEventListener('click', saveData);
    }

    // Show edit mode indicator
    const title = document.querySelector('.title');
    if (title) {
        title.innerHTML += ' <span style="color: #E74C3C; font-size: 14px;">[EDIT MODE]</span>';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', async function() {
    editMode = checkEditMode();
    const loaded = await loadData();
    if (loaded) {
        initializeNetwork();
        window.addEventListener('resize', positionView);
        if (editMode) {
            showEditModeUI();
        }
    } else {
        document.getElementById('network').innerHTML = '<p style="padding: 20px; color: red;">Error loading concept map data.</p>';
    }
});
