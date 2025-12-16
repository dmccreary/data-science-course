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

// Node definitions with positions, categories, and definitions
const nodeData = [
    // Core Concepts (top level)
    { id: 'ds', label: 'Data\nScience', category: 'core', x: 0, y: -200,
      definition: 'The interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge from data.' },
    { id: 'data', label: 'Data', category: 'core', x: -150, y: -80,
      definition: 'Raw facts, figures, and observations collected for analysis.' },
    { id: 'variables', label: 'Variables', category: 'core', x: 150, y: -80,
      definition: 'Characteristics or attributes that can take different values across observations.' },
    { id: 'dataset', label: 'Dataset', category: 'core', x: 0, y: -80,
      definition: 'A structured collection of data organized into rows (observations) and columns (variables).' },

    // Data Types
    { id: 'numerical', label: 'Numerical\nData', category: 'dataType', x: -280, y: 50,
      definition: 'Data that represents quantities and can be measured. Math operations are meaningful.' },
    { id: 'categorical', label: 'Categorical\nData', category: 'dataType', x: -80, y: 50,
      definition: 'Data that represents groups or categories. Math operations are not meaningful.' },
    { id: 'ordinal', label: 'Ordinal\nData', category: 'dataType', x: -150, y: 150,
      definition: 'Categorical data with a meaningful order or ranking (e.g., Low < Medium < High).' },
    { id: 'nominal', label: 'Nominal\nData', category: 'dataType', x: -10, y: 150,
      definition: 'Categorical data with no inherent order (e.g., colors, countries).' },

    // Variable Roles
    { id: 'independent', label: 'Independent\nVariable', category: 'variableRole', x: 200, y: 50,
      definition: 'The variable manipulated or used to predict another variable (the input).' },
    { id: 'dependent', label: 'Dependent\nVariable', category: 'variableRole', x: 350, y: 50,
      definition: 'The variable being predicted or explained (the output).' },
    { id: 'feature', label: 'Feature', category: 'variableRole', x: 200, y: 150,
      definition: 'In machine learning, an input variable used to make predictions. Same as independent variable.' },
    { id: 'target', label: 'Target\nVariable', category: 'variableRole', x: 350, y: 150,
      definition: 'In machine learning, the variable we want to predict. Same as dependent variable.' },

    // Workflow
    { id: 'workflow', label: 'Data Science\nWorkflow', category: 'workflow', x: -350, y: -150,
      definition: 'The systematic process of solving problems with data: Define → Collect → Clean → Analyze → Communicate.' },
    { id: 'problem', label: 'Problem\nDefinition', category: 'workflow', x: -400, y: -50,
      definition: 'The first step: clearly articulating the question you want to answer with data.' },
    { id: 'collection', label: 'Data\nCollection', category: 'workflow', x: -400, y: 50,
      definition: 'Gathering relevant data from various sources to address the defined problem.' },

    // Tools & Practices
    { id: 'python', label: 'Python\nProgramming', category: 'tools', x: 100, y: -200,
      definition: 'A popular programming language for data science with rich libraries like pandas and numpy.' },
    { id: 'documentation', label: 'Documentation', category: 'tools', x: -100, y: -200,
      definition: 'Recording methods, decisions, and findings to ensure reproducibility and clarity.' },

    // Structure
    { id: 'scales', label: 'Measurement\nScales', category: 'structure', x: -200, y: 250,
      definition: 'The levels at which variables are measured: nominal, ordinal, interval, and ratio.' },
    { id: 'observation', label: 'Observation', category: 'structure', x: 80, y: 250,
      definition: 'A single row in a dataset representing one instance or record of data.' }
];

// Edge definitions with types
const edgeData = [
    // "works with" / main relationships
    { from: 'ds', to: 'data', label: 'works with', type: 'uses' },
    { from: 'ds', to: 'python', label: 'uses', type: 'uses' },
    { from: 'ds', to: 'workflow', label: 'follows', type: 'uses' },

    // "organized into" / contains
    { from: 'data', to: 'variables', label: 'organized into', type: 'contains' },
    { from: 'dataset', to: 'observation', label: 'contains', type: 'contains' },
    { from: 'dataset', to: 'variables', label: 'contains', type: 'contains' },

    // "classified by" / is a type of
    { from: 'variables', to: 'numerical', label: 'classified as', type: 'isTypeOf' },
    { from: 'variables', to: 'categorical', label: 'classified as', type: 'isTypeOf' },
    { from: 'categorical', to: 'ordinal', label: 'type', type: 'isTypeOf' },
    { from: 'categorical', to: 'nominal', label: 'type', type: 'isTypeOf' },

    // Variable roles
    { from: 'variables', to: 'independent', label: 'can be', type: 'isTypeOf' },
    { from: 'variables', to: 'dependent', label: 'can be', type: 'isTypeOf' },

    // "same as" relationships
    { from: 'feature', to: 'independent', label: 'same as', type: 'sameAs' },
    { from: 'target', to: 'dependent', label: 'same as', type: 'sameAs' },

    // Workflow sequence
    { from: 'workflow', to: 'problem', label: 'starts with', type: 'contains' },
    { from: 'problem', to: 'collection', label: 'then', type: 'contains' },

    // Documentation
    { from: 'ds', to: 'documentation', label: 'requires', type: 'uses' },

    // Measurement scales
    { from: 'numerical', to: 'scales', label: 'measured by', type: 'uses' },
    { from: 'categorical', to: 'scales', label: 'measured by', type: 'uses' }
];

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

// Environment detection
function isInIframe() {
    try {
        return window.self !== window.top;
    } catch (e) {
        return true;
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
            dragNodes: false,
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
    const examples = {
        'ds': ['Predicting house prices', 'Analyzing customer behavior', 'Medical diagnosis from symptoms'],
        'data': ['Survey responses', 'Sensor readings', 'Transaction records'],
        'variables': ['Age, Income, Education Level', 'Temperature, Pressure', 'Satisfaction Rating'],
        'dataset': ['Titanic passenger data', 'Iris flower measurements', 'MNIST handwritten digits'],
        'numerical': ['Height: 5.9 feet', 'Temperature: 72.5F', 'Age: 25 years'],
        'categorical': ['Color: Red, Blue, Green', 'Size: S, M, L, XL', 'Country: USA, Canada, Mexico'],
        'ordinal': ['Education: HS < BS < MS < PhD', 'Rating: Poor < Fair < Good < Excellent'],
        'nominal': ['Eye color: Blue, Brown, Green', 'Blood type: A, B, AB, O'],
        'independent': ['Study hours (predicting test score)', 'Ad spend (predicting sales)'],
        'dependent': ['Test score', 'Sales revenue', 'Customer churn'],
        'feature': ['Pixel values in image classification', 'Word frequencies in text analysis'],
        'target': ['Image label (cat/dog)', 'Sentiment (positive/negative)'],
        'workflow': ['CRISP-DM methodology', 'Iterative refinement process'],
        'problem': ['What factors predict customer churn?', 'How can we reduce delivery times?'],
        'collection': ['Web scraping', 'API calls', 'Database queries', 'Surveys'],
        'python': ['pandas for data manipulation', 'matplotlib for visualization', 'scikit-learn for ML'],
        'documentation': ['Jupyter notebooks', 'README files', 'Code comments'],
        'scales': ['Nominal, Ordinal, Interval, Ratio', 'Stevens\' levels of measurement'],
        'observation': ['One patient record', 'One transaction', 'One survey response']
    };

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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeNetwork();
    window.addEventListener('resize', positionView);
});
