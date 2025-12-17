// Independent vs Dependent Variables MicroSim
// Demonstrates cause-and-effect relationships through scatter plot exploration

// Canvas dimensions
let canvasWidth = 700;
let drawHeight = 400;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let margin = 25;
let defaultTextSize = 16;

// Plot dimensions
let plotLeft = 70;
let plotRight; // Calculated based on canvasWidth
let plotTop = 50;
let plotBottom = 360;
let statsWidth = 180;

// Data
let students = [];
let relationshipStrength = 70;

// UI Controls
let strengthSlider;
let addStudentBtn;
let generateClassBtn;
let clearBtn;
let showTrendCheckbox;
let showPredictionCheckbox;
let sliderLeftMargin = 200;

// State
let showTrendLine = true;
let showPrediction = true;
let hoveredStudent = null;
let predictX = null;
let predictY = null;
let lastClickMessage = "";
let messageTimer = 0;

function setup() {
    updateCanvasSize();
    const canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent(document.querySelector('main'));

    // Create controls
    createControls();

    // Generate initial class of 10
    generateStudents(10);

    describe('Interactive scatter plot showing relationship between study time and test scores', LABEL);
}

function createControls() {
    // Row 1: Buttons
    addStudentBtn = createButton('Add Student');
    addStudentBtn.position(10, drawHeight + 10);
    addStudentBtn.mousePressed(addRandomStudent);

    generateClassBtn = createButton('Generate Class of 30');
    generateClassBtn.position(105, drawHeight + 10);
    generateClassBtn.mousePressed(() => {
        students = [];
        generateStudents(30);
    });

    clearBtn = createButton('Clear All');
    clearBtn.position(260, drawHeight + 10);
    clearBtn.mousePressed(() => {
        students = [];
        lastClickMessage = "";
    });

    // Row 2: Relationship Strength slider
    strengthSlider = createSlider(0, 100, 70, 1);
    strengthSlider.position(sliderLeftMargin, drawHeight + 45);
    strengthSlider.size(canvasWidth - 360 - margin);
    strengthSlider.input(() => {
        relationshipStrength = strengthSlider.value();
    });

    // Row 2: Checkboxes (right side)
    showTrendCheckbox = createCheckbox('Show Trend Line', true);
    showTrendCheckbox.position(canvasWidth - 170, drawHeight + 40);
    showTrendCheckbox.changed(() => {
        showTrendLine = showTrendCheckbox.checked();
    });

    showPredictionCheckbox = createCheckbox('Show Prediction', true);
    showPredictionCheckbox.position(canvasWidth - 170, drawHeight + 65);
    showPredictionCheckbox.changed(() => {
        showPrediction = showPredictionCheckbox.checked();
    });
}

function draw() {
    updateCanvasSize();

    // Drawing area background
    fill('aliceblue');
    stroke('silver');
    strokeWeight(1);
    rect(0, 0, canvasWidth, drawHeight);

    // Control area background
    fill('white');
    noStroke();
    rect(0, drawHeight, canvasWidth, controlHeight);

    // Update plot dimensions
    plotRight = canvasWidth - statsWidth - 20;

    // Draw title
    fill('black');
    textSize(20);
    textAlign(CENTER, TOP);
    noStroke();
    text('Independent vs Dependent Variables', (plotLeft + plotRight) / 2, 10);

    // Reset text settings
    textSize(defaultTextSize);
    textAlign(LEFT, CENTER);

    // Draw plot area
    drawPlotArea();

    // Draw confidence band and trend line
    if (showTrendLine && students.length >= 2) {
        drawConfidenceBand();
        drawTrendLine();
    }

    // Draw data points
    drawDataPoints();

    // Draw prediction line on hover
    if (showPrediction) {
        drawPrediction();
    }

    // Draw stats panel
    drawStatsPanel();

    // Draw control labels
    drawControlLabels();

    // Update message timer
    if (messageTimer > 0) {
        messageTimer--;
    }
}

function drawPlotArea() {
    // Plot background
    fill(255);
    stroke('silver');
    strokeWeight(1);
    rect(plotLeft, plotTop, plotRight - plotLeft, plotBottom - plotTop);

    // Grid lines
    stroke(230);
    strokeWeight(0.5);

    // Vertical grid (hours)
    for (let h = 0; h <= 10; h++) {
        let x = map(h, 0, 10, plotLeft, plotRight);
        line(x, plotTop, x, plotBottom);
    }

    // Horizontal grid (scores)
    for (let s = 0; s <= 100; s += 10) {
        let y = map(s, 0, 100, plotBottom, plotTop);
        line(plotLeft, y, plotRight, y);
    }

    // Axes
    stroke(0);
    strokeWeight(2);
    // X-axis
    line(plotLeft, plotBottom, plotRight, plotBottom);
    // Y-axis
    line(plotLeft, plotTop, plotLeft, plotBottom);

    // Axis labels
    fill(0);
    noStroke();
    textSize(14);
    textAlign(CENTER, TOP);

    // X-axis label
    text('Hours Studied (Independent Variable)', (plotLeft + plotRight) / 2, plotBottom + 25);

    // X-axis tick labels
    textSize(12);
    for (let h = 0; h <= 10; h += 2) {
        let x = map(h, 0, 10, plotLeft, plotRight);
        text(h, x, plotBottom + 5);
    }

    // Y-axis label
    push();
    translate(20, (plotTop + plotBottom) / 2);
    rotate(-HALF_PI);
    textAlign(CENTER, CENTER);
    textSize(14);
    text('Test Score (Dependent Variable)', 0, 0);
    pop();

    // Y-axis tick labels
    textSize(12);
    textAlign(RIGHT, CENTER);
    for (let s = 0; s <= 100; s += 20) {
        let y = map(s, 0, 100, plotBottom, plotTop);
        text(s, plotLeft - 5, y);
    }

    // Instruction text in upper right of plot
    fill('black');
    textSize(12);
    textAlign(RIGHT, TOP);
    textStyle(ITALIC);
    text('Click to add new student', canvasWidth - 100, 15);
    textStyle(NORMAL);
}

function drawTrendLine() {
    if (students.length < 2) return;

    let { slope, intercept } = calculateRegression();

    stroke(0, 100, 200);
    strokeWeight(3);

    let x1 = 0;
    let y1 = intercept;
    let x2 = 10;
    let y2 = slope * 10 + intercept;

    // Clamp to plot area
    y1 = constrain(y1, 0, 100);
    y2 = constrain(y2, 0, 100);

    let px1 = map(x1, 0, 10, plotLeft, plotRight);
    let py1 = map(y1, 0, 100, plotBottom, plotTop);
    let px2 = map(x2, 0, 10, plotLeft, plotRight);
    let py2 = map(y2, 0, 100, plotBottom, plotTop);

    line(px1, py1, px2, py2);
}

function drawConfidenceBand() {
    if (students.length < 2) return;

    let { slope, intercept } = calculateRegression();
    let stdError = calculateStdError(slope, intercept);

    fill(0, 100, 200, 40);
    noStroke();

    beginShape();
    // Top edge of band
    for (let h = 0; h <= 10; h += 0.5) {
        let predictedY = slope * h + intercept;
        let upperY = constrain(predictedY + stdError * 1.5, 0, 100);
        let px = map(h, 0, 10, plotLeft, plotRight);
        let py = map(upperY, 0, 100, plotBottom, plotTop);
        vertex(px, py);
    }
    // Bottom edge of band (reverse direction)
    for (let h = 10; h >= 0; h -= 0.5) {
        let predictedY = slope * h + intercept;
        let lowerY = constrain(predictedY - stdError * 1.5, 0, 100);
        let px = map(h, 0, 10, plotLeft, plotRight);
        let py = map(lowerY, 0, 100, plotBottom, plotTop);
        vertex(px, py);
    }
    endShape(CLOSE);
}

function drawDataPoints() {
    hoveredStudent = null;

    textSize(20);
    textAlign(CENTER, CENTER);

    // Get current regression for dynamic outlier calculation
    let { slope, intercept } = calculateRegression();
    let stdError = students.length >= 3 ? calculateStdError(slope, intercept) : 15;
    let outlierThreshold = stdError * 1.5;

    for (let i = 0; i < students.length; i++) {
        let s = students[i];
        let px = map(s.hours, 0, 10, plotLeft, plotRight);
        let py = map(s.score, 0, 100, plotBottom, plotTop);

        // Check if mouse is hovering
        let d = dist(mouseX, mouseY, px, py);
        if (d < 15) {
            hoveredStudent = { student: s, x: px, y: py, index: i };
        }

        // Dynamically calculate if student is outlier based on current regression
        let expectedScore = slope * s.hours + intercept;
        let residual = Math.abs(s.score - expectedScore);
        let isOutlier = residual > outlierThreshold;

        // Draw student emoji
        text(isOutlier ? '🌟' : '🎓', px, py);
    }

    // Draw hover tooltip
    if (hoveredStudent) {
        let s = hoveredStudent.student;
        fill(255, 255, 255, 240);
        stroke(100);
        strokeWeight(1);
        rectMode(CENTER);
        rect(hoveredStudent.x, hoveredStudent.y - 35, 120, 30, 5);
        rectMode(CORNER);

        fill(0);
        noStroke();
        textSize(12);
        text(`${s.hours.toFixed(1)}h → ${s.score.toFixed(0)} pts`, hoveredStudent.x, hoveredStudent.y - 35);
    }
}

function drawPrediction() {
    if (students.length < 2) return;

    let { slope, intercept } = calculateRegression();

    // Convert mouse position to data coordinates
    let dataX = map(mouseX, plotLeft, plotRight, 0, 10);

    // Only show prediction when mouse is in plot area
    if (mouseX < plotLeft || mouseX > plotRight || mouseY < plotTop || mouseY > plotBottom) {
        predictX = null;
        return;
    }

    dataX = constrain(dataX, 0, 10);
    let dataY = slope * dataX + intercept;
    dataY = constrain(dataY, 0, 100);

    predictX = dataX;
    predictY = dataY;

    let px = map(dataX, 0, 10, plotLeft, plotRight);
    let py = map(dataY, 0, 100, plotBottom, plotTop);

    // Vertical line to trend
    stroke(255, 100, 0, 150);
    strokeWeight(2);
    setLineDash([5, 5]);
    line(px, plotBottom, px, py);

    // Horizontal line from trend to axis
    line(plotLeft, py, px, py);
    setLineDash([]);

    // Prediction point
    fill(255, 100, 0);
    noStroke();
    ellipse(px, py, 12, 12);

    // Prediction text
    fill(0);
    textSize(12);
    textAlign(LEFT, BOTTOM);
    let predText = `Study ${dataX.toFixed(1)}h → ~${dataY.toFixed(0)} pts`;

    // Position text to stay in bounds
    let textX = px + 10;
    let textY = py - 10;
    if (textX > plotRight - 100) textX = px - 110;
    if (textY < plotTop + 20) textY = py + 20;

    fill(255, 255, 255, 230);
    stroke(255, 100, 0);
    strokeWeight(1);
    rectMode(CORNER);
    rect(textX - 5, textY - 14, 130, 18, 3);

    fill(0);
    noStroke();
    text(predText, textX, textY);
}

function setLineDash(pattern) {
    drawingContext.setLineDash(pattern);
}

function drawStatsPanel() {
    let panelX = plotRight + 10;
    let panelY = plotTop;
    let panelWidth = statsWidth;
    let panelHeight = plotBottom - plotTop;

    // Panel background
    fill(255, 255, 255, 230);
    stroke(200);
    strokeWeight(1);
    rect(panelX, panelY, panelWidth, panelHeight, 10);

    // Stats content
    fill(0);
    noStroke();
    textSize(14);
    textAlign(LEFT, TOP);

    let y = panelY + 15;
    let lineHeight = 28;

    // Number of students
    textStyle(BOLD);
    text('Students:', panelX + 10, y);
    textStyle(NORMAL);
    textAlign(RIGHT, TOP);
    text(students.length, panelX + panelWidth - 10, y);
    textAlign(LEFT, TOP);
    y += lineHeight;

    // Average study time
    let avgHours = students.length > 0 ? students.reduce((sum, s) => sum + s.hours, 0) / students.length : 0;
    textStyle(BOLD);
    text('Avg Study:', panelX + 10, y);
    textStyle(NORMAL);
    textAlign(RIGHT, TOP);
    text(avgHours.toFixed(1) + 'h', panelX + panelWidth - 10, y);
    textAlign(LEFT, TOP);
    y += lineHeight;

    // Average score
    let avgScore = students.length > 0 ? students.reduce((sum, s) => sum + s.score, 0) / students.length : 0;
    textStyle(BOLD);
    text('Avg Score:', panelX + 10, y);
    textStyle(NORMAL);
    textAlign(RIGHT, TOP);
    text(avgScore.toFixed(0), panelX + panelWidth - 10, y);
    textAlign(LEFT, TOP);
    y += lineHeight;

    // Correlation
    let corr = calculateCorrelation();
    let corrLabel = getCorrelationLabel(corr);
    textStyle(BOLD);
    text('Correlation:', panelX + 10, y);
    textStyle(NORMAL);
    y += lineHeight - 5;
    textAlign(CENTER, TOP);
    textSize(24);
    text(corrLabel.emoji, panelX + panelWidth / 2, y);
    textSize(12);
    y += 30;
    text(corrLabel.text, panelX + panelWidth / 2, y);
    text(`(r = ${corr.toFixed(2)})`, panelX + panelWidth / 2, y + 15);
    textAlign(LEFT, TOP);

    // Educational message
    y = panelY + panelHeight - 100;
    textSize(11);
    textWrap(WORD);
    textAlign(LEFT, TOP);
    fill(60, 60, 100);
    text('INDEPENDENT: what you control (study time)', panelX + 8, y, panelWidth - 16);
    y += 35;
    text('DEPENDENT: what you measure (score)', panelX + 8, y, panelWidth - 16);

    // Show click message
    if (lastClickMessage && messageTimer > 0) {
        y = panelY + panelHeight - 25;
        fill(200, 100, 0);
        textSize(10);
        textAlign(CENTER, TOP);
        text(lastClickMessage, panelX + panelWidth / 2, y);
    }
}

function drawControlLabels() {
    fill(0);
    noStroke();
    textSize(14);
    textAlign(LEFT, CENTER);
    textStyle(NORMAL);

    // Relationship strength label
    text('Relationship Strength: ' + relationshipStrength + '%', 10, drawHeight + 53);

    // Legend at bottom of control area
    textSize(12);
    textAlign(LEFT, CENTER);
    text('🎓 Normal', 10, drawHeight + 85);
    text('🌟 Outlier', 100, drawHeight + 85);
}

function calculateRegression() {
    if (students.length < 2) return { slope: 7, intercept: 30 };

    let n = students.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (let s of students) {
        sumX += s.hours;
        sumY += s.score;
        sumXY += s.hours * s.score;
        sumX2 += s.hours * s.hours;
    }

    let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    let intercept = (sumY - slope * sumX) / n;

    // Handle edge cases
    if (!isFinite(slope)) slope = 7;
    if (!isFinite(intercept)) intercept = 30;

    return { slope, intercept };
}

function calculateStdError(slope, intercept) {
    if (students.length < 3) return 10;

    let residuals = students.map(s => s.score - (slope * s.hours + intercept));
    let sumSquares = residuals.reduce((sum, r) => sum + r * r, 0);
    return Math.sqrt(sumSquares / (students.length - 2));
}

function calculateCorrelation() {
    if (students.length < 2) return 0;

    let n = students.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (let s of students) {
        sumX += s.hours;
        sumY += s.score;
        sumXY += s.hours * s.score;
        sumX2 += s.hours * s.hours;
        sumY2 += s.score * s.score;
    }

    let num = n * sumXY - sumX * sumY;
    let den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    if (den === 0) return 0;
    return num / den;
}

function getCorrelationLabel(r) {
    let absR = Math.abs(r);
    if (absR >= 0.7) return { emoji: '💪', text: 'Strong' };
    if (absR >= 0.4) return { emoji: '👍', text: 'Moderate' };
    if (absR >= 0.2) return { emoji: '🤷', text: 'Weak' };
    return { emoji: '😴', text: 'Very Weak' };
}

function generateStudents(count) {
    for (let i = 0; i < count; i++) {
        addRandomStudent();
    }
}

function addRandomStudent() {
    // Random hours between 0.5 and 9.5
    let hours = random(0.5, 9.5);

    // Base expected score: more hours = higher score
    // Perfect relationship: score = 30 + 7 * hours (ranges from 33.5 to 96.5)
    let expectedScore = 30 + 7 * hours;

    // Add noise based on relationship strength
    // At 100% strength, very little noise. At 0%, huge noise.
    let noiseLevel = map(relationshipStrength, 0, 100, 35, 3);
    let noise = randomGaussian(0, noiseLevel);

    let score = expectedScore + noise;
    score = constrain(score, 0, 100);

    // Check if this is an outlier (beat the odds significantly)
    let residual = Math.abs(score - expectedScore);
    let isOutlier = residual > noiseLevel * 2;

    if (isOutlier) {
        if (score > expectedScore) {
            lastClickMessage = "Beat the odds! 🌟";
        } else {
            lastClickMessage = "Unexpected result! 🌟";
        }
        messageTimer = 120; // Show for 2 seconds
    }

    students.push({
        hours: hours,
        score: score,
        isOutlier: isOutlier
    });
}

function mousePressed() {
    // Check if click is in plot area
    if (mouseX >= plotLeft && mouseX <= plotRight && mouseY >= plotTop && mouseY <= plotBottom) {
        // Add student at clicked location
        let hours = map(mouseX, plotLeft, plotRight, 0, 10);
        let score = map(mouseY, plotBottom, plotTop, 0, 100);

        hours = constrain(hours, 0, 10);
        score = constrain(score, 0, 100);

        // Check if outlier compared to expected
        let expectedScore = 30 + 7 * hours;
        let residual = Math.abs(score - expectedScore);
        let noiseLevel = map(relationshipStrength, 0, 100, 35, 3);
        let isOutlier = residual > noiseLevel * 1.5;

        if (isOutlier) {
            if (score > expectedScore) {
                lastClickMessage = "Beat the odds! 🌟";
            } else {
                lastClickMessage = "Below expected! 🌟";
            }
            messageTimer = 120;
        } else {
            lastClickMessage = "Added student!";
            messageTimer = 60;
        }

        students.push({
            hours: hours,
            score: score,
            isOutlier: isOutlier
        });
    }
}

function windowResized() {
    updateCanvasSize();
    resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
    const container = document.querySelector('main');
    if (container) {
        canvasWidth = min(container.offsetWidth, 800);
        plotRight = canvasWidth - statsWidth - 20;

        // Reposition controls
        if (typeof strengthSlider !== 'undefined') {
            strengthSlider.size(canvasWidth - 360 - margin);
        }
        if (typeof showTrendCheckbox !== 'undefined') {
            showTrendCheckbox.position(canvasWidth - 170, drawHeight + 40);
        }
        if (typeof showPredictionCheckbox !== 'undefined') {
            showPredictionCheckbox.position(canvasWidth - 170, drawHeight + 65);
        }
    }
}
