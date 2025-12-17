// Measurement Scales Pyramid MicroSim
// Interactive infographic showing the four measurement scales hierarchy

// Canvas dimensions
let canvasWidth = 500;
let drawHeight = 450;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;

// Pyramid dimensions
let pyramidBaseWidth;
let pyramidHeight;
let pyramidX;
let pyramidY;
let layerHeight;

// Layer data
const layers = [
    {
        name: 'RATIO',
        operations: '= ≠ < > + − × ÷',
        example: 'Height in centimeters',
        definition: 'Has a true zero point; ratios are meaningful.',
        color: [255, 215, 0],      // Gold
        glowColor: [255, 235, 100],
        examples: [
            'Height (cm)',
            'Weight (kg)',
            'Age (years)',
            'Income ($)',
            'Distance (miles)'
        ],
        tests: [
            'All statistical tests',
            'Geometric mean',
            'Coefficient of variation',
            'Ratio comparisons',
            'Percentage calculations'
        ]
    },
    {
        name: 'INTERVAL',
        operations: '= ≠ < > + −',
        example: 'Temperature in °F',
        definition: 'Equal intervals between values; no true zero.',
        color: [144, 238, 144],    // Light green
        glowColor: [180, 255, 180],
        examples: [
            'Temperature (°F/°C)',
            'IQ scores',
            'SAT scores',
            'Calendar dates',
            'Credit scores'
        ],
        tests: [
            'Mean, Std deviation',
            't-tests, ANOVA',
            'Pearson correlation',
            'Linear regression',
            'Factor analysis'
        ]
    },
    {
        name: 'ORDINAL',
        operations: '= ≠ < >',
        example: 'Race finishing positions',
        definition: 'Categories with a meaningful order or rank.',
        color: [173, 216, 230],    // Light blue
        glowColor: [200, 235, 255],
        examples: [
            '1st, 2nd, 3rd place',
            'Likert scale (1-5)',
            'Education level',
            'Pain scale (0-10)',
            'Movie ratings (★★★)'
        ],
        tests: [
            'Median, Percentiles',
            'Spearman correlation',
            'Mann-Whitney U test',
            'Kruskal-Wallis test',
            'Ordinal regression'
        ]
    },
    {
        name: 'NOMINAL',
        operations: '= ≠',
        example: 'Jersey numbers',
        definition: 'Categories with no inherent order.',
        color: [200, 200, 200],    // Light gray
        glowColor: [230, 230, 230],
        examples: [
            'Jersey numbers',
            'Eye color',
            'Gender',
            'Country of birth',
            'Blood type (A/B/AB/O)'
        ],
        tests: [
            'Mode only',
            'Chi-square test',
            'Fisher exact test',
            'Frequency tables',
            'Contingency tables'
        ]
    }
];

// State
let hoveredLayer = -1;
let selectedLayer = -1;
let animFrame = 0;

function setup() {
    updateCanvasSize();
    const canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent(document.querySelector('main'));
    textFont('Arial');

    describe('Interactive pyramid showing four measurement scales: Nominal, Ordinal, Interval, and Ratio. Hover to see examples, click to see statistical tests.', LABEL);
}

function draw() {
    updateCanvasSize();
    animFrame++;

    // Drawing area
    background(240, 248, 255); // aliceblue
    stroke(192);
    strokeWeight(1);
    noFill();
    rect(0, 0, canvasWidth - 1, drawHeight - 1);

    // Control area
    fill(255);
    noStroke();
    rect(0, drawHeight, canvasWidth, controlHeight);

    // Calculate pyramid dimensions
    pyramidBaseWidth = min(canvasWidth * 0.65, 400);
    pyramidHeight = drawHeight * 0.7;
    pyramidX = canvasWidth * 0.45 + 50;
    pyramidY = drawHeight * 0.88;
    layerHeight = pyramidHeight / 4;

    // Title
    fill(50);
    noStroke();
    textSize(22);
    textAlign(CENTER, TOP);
    text('Measurement Scales Pyramid', canvasWidth / 2, 12);

    // Draw side annotations
    drawAnnotations();

    // Draw pyramid layers (bottom to top)
    for (let i = 3; i >= 0; i--) {
        drawPyramidLayer(i);
    }

    // Draw info panel
    drawInfoPanel();

    // Draw instructions in control area
    fill(100);
    textSize(14);
    textAlign(CENTER, CENTER);
    text('Hover over a layer to see examples • Click to see statistical tests', canvasWidth / 2, drawHeight + controlHeight / 2);
}

function drawPyramidLayer(index) {
    const layer = layers[index];
    const layerIndex = 3 - index; // 0 = bottom, 3 = top

    // Calculate trapezoid dimensions
    const topRatio = (layerIndex + 1) / 4;
    const bottomRatio = layerIndex / 4;

    const topWidth = pyramidBaseWidth * (1 - topRatio * 0.7);
    const bottomWidth = pyramidBaseWidth * (1 - bottomRatio * 0.7);

    const yTop = pyramidY - (layerIndex + 1) * layerHeight;
    const yBottom = pyramidY - layerIndex * layerHeight;

    // Check if mouse is over this layer
    const isHovered = hoveredLayer === index;
    const isSelected = selectedLayer === index;

    // Draw glow effect if hovered or selected
    if (isHovered || isSelected) {
        const glowIntensity = sin(animFrame * 0.1) * 20 + 40;
        for (let g = 15; g > 0; g--) {
            const alpha = map(g, 15, 0, 0, glowIntensity);
            stroke(layer.glowColor[0], layer.glowColor[1], layer.glowColor[2], alpha);
            strokeWeight(g * 2);
            noFill();
            drawTrapezoid(pyramidX, yTop, yBottom, topWidth, bottomWidth);
        }
    }

    // Draw 3D effect (darker side)
    const darken = 40;
    const depth = 15;

    // Bottom face (only for bottom layer)
    if (layerIndex === 0) {
        fill(layer.color[0] - darken - 20, layer.color[1] - darken - 20, layer.color[2] - darken - 20);
        stroke(80);
        strokeWeight(2);
        beginShape();
        vertex(pyramidX - bottomWidth / 2, yBottom);
        vertex(pyramidX + bottomWidth / 2, yBottom);
        vertex(pyramidX + bottomWidth / 2 + depth, yBottom + depth * 0.5);
        vertex(pyramidX - bottomWidth / 2 + depth, yBottom + depth * 0.5);
        endShape(CLOSE);
    }

    // Right side 3D effect
    fill(layer.color[0] - darken, layer.color[1] - darken, layer.color[2] - darken);
    stroke(80);
    strokeWeight(2);
    beginShape();
    vertex(pyramidX + topWidth / 2, yTop);
    vertex(pyramidX + topWidth / 2 + depth, yTop + depth * 0.5);
    vertex(pyramidX + bottomWidth / 2 + depth, yBottom + depth * 0.5);
    vertex(pyramidX + bottomWidth / 2, yBottom);
    endShape(CLOSE);

    // Main face
    fill(layer.color[0], layer.color[1], layer.color[2]);
    stroke(80);
    strokeWeight(2);
    drawTrapezoid(pyramidX, yTop, yBottom, topWidth, bottomWidth);

    // Highlight edge if hovered
    if (isHovered || isSelected) {
        stroke(255, 255, 255, 200);
        strokeWeight(3);
        noFill();
        drawTrapezoid(pyramidX, yTop, yBottom, topWidth, bottomWidth);
    }

    // Layer text
    const centerY = (yTop + yBottom) / 2;
    fill(40);
    noStroke();
    textAlign(CENTER, CENTER);

    // Name
    textSize(16);
    textStyle(BOLD);
    text(layer.name, pyramidX, centerY - 18);

    // Operations
    textSize(13);
    textStyle(NORMAL);
    fill(60);
    text(layer.operations, pyramidX, centerY + 2);

    // Example
    textSize(11);
    fill(80);
    textStyle(ITALIC);
    text('"' + layer.example + '"', pyramidX, centerY + 20);
    textStyle(NORMAL);
}

function drawTrapezoid(centerX, yTop, yBottom, topWidth, bottomWidth) {
    beginShape();
    vertex(centerX - topWidth / 2, yTop);
    vertex(centerX + topWidth / 2, yTop);
    vertex(centerX + bottomWidth / 2, yBottom);
    vertex(centerX - bottomWidth / 2, yBottom);
    endShape(CLOSE);
}

function drawAnnotations() {
    const arrowX = pyramidX + pyramidBaseWidth / 2 + 60;
    const arrowTop = pyramidY - pyramidHeight + 20;
    const arrowBottom = pyramidY - 20;

    // Up arrow
    stroke(100);
    strokeWeight(2);
    line(arrowX, arrowBottom, arrowX, arrowTop);

    // Arrow head up
    fill(100);
    noStroke();
    triangle(arrowX, arrowTop - 5, arrowX - 6, arrowTop + 8, arrowX + 6, arrowTop + 8);

    // Arrow head down
    triangle(arrowX, arrowBottom + 5, arrowX - 6, arrowBottom - 8, arrowX + 6, arrowBottom - 8);

    // Labels
    push();
    textSize(11);
    fill(80);
    textAlign(CENTER, CENTER);

    // Top label
    translate(arrowX + 25, arrowTop + 50);
    rotate(-PI / 2);
    text('More operations', 0, 0);
    pop();

    push();
    translate(arrowX + 25, arrowBottom - 50);
    rotate(-PI / 2);
    textSize(11);
    fill(80);
    textAlign(CENTER, CENTER);
    text('More variables fit', 0, 0);
    pop();
}

function drawInfoPanel() {
    const panelX = 15;
    const panelY = 55;
    const panelWidth = canvasWidth * 0.28;
    const panelHeight = 210;

    // Panel background
    fill(255, 255, 255, 240);
    stroke(180);
    strokeWeight(1);
    rect(panelX, panelY, panelWidth, panelHeight, 8);

    // Panel content
    let displayLayer = selectedLayer >= 0 ? selectedLayer : hoveredLayer;

    if (displayLayer >= 0) {
        const layer = layers[displayLayer];
        const isShowingTests = selectedLayer >= 0;

        // Title
        fill(layer.color[0] - 30, layer.color[1] - 30, layer.color[2] - 30);
        noStroke();
        textSize(14);
        textStyle(BOLD);
        textAlign(LEFT, TOP);
        text(layer.name, panelX + 10, panelY + 10);

        // Definition
        textStyle(ITALIC);
        textSize(11);
        fill(80);
        text(layer.definition, panelX + 10, panelY + 28, panelWidth - 20);

        // Subtitle
        textStyle(NORMAL);
        textSize(11);
        fill(100);
        text(isShowingTests ? 'Statistical Tests:' : 'Example Variables:', panelX + 10, panelY + 58);

        // Items
        const items = isShowingTests ? layer.tests : layer.examples;
        textSize(12);
        fill(60);
        for (let i = 0; i < items.length; i++) {
            text('• ' + items[i], panelX + 12, panelY + 78 + i * 24);
        }
    } else {
        // Default message
        fill(120);
        textSize(13);
        textAlign(LEFT, TOP);
        textStyle(ITALIC);
        text('Hover over a pyramid\nlayer to see example\nvariables.\n\nClick a layer to see\nappropriate statistical\ntests.', panelX + 12, panelY + 15);
        textStyle(NORMAL);
    }
}

function mouseMoved() {
    updateHoveredLayer();
    return false;
}

function mousePressed() {
    updateHoveredLayer();
    if (hoveredLayer >= 0) {
        if (selectedLayer === hoveredLayer) {
            selectedLayer = -1; // Deselect if clicking same layer
        } else {
            selectedLayer = hoveredLayer;
        }
    } else {
        selectedLayer = -1;
    }
    return false;
}

function updateHoveredLayer() {
    hoveredLayer = -1;

    for (let i = 0; i < 4; i++) {
        const layerIndex = 3 - i;
        const topRatio = (layerIndex + 1) / 4;
        const bottomRatio = layerIndex / 4;

        const topWidth = pyramidBaseWidth * (1 - topRatio * 0.7);
        const bottomWidth = pyramidBaseWidth * (1 - bottomRatio * 0.7);

        const yTop = pyramidY - (layerIndex + 1) * layerHeight;
        const yBottom = pyramidY - layerIndex * layerHeight;

        if (mouseY >= yTop && mouseY <= yBottom) {
            // Interpolate width at mouse Y position
            const t = (mouseY - yTop) / (yBottom - yTop);
            const widthAtY = lerp(topWidth, bottomWidth, t);

            if (mouseX >= pyramidX - widthAtY / 2 && mouseX <= pyramidX + widthAtY / 2) {
                hoveredLayer = i;
                break;
            }
        }
    }

    // Update cursor
    if (hoveredLayer >= 0) {
        cursor(HAND);
    } else {
        cursor(ARROW);
    }
}

function windowResized() {
    updateCanvasSize();
    resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
    const container = document.querySelector('main');
    if (container) {
        canvasWidth = container.offsetWidth;
    }
}
