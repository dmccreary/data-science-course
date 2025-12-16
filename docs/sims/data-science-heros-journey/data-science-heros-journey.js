// Data Science Hero's Journey MicroSim
// An interactive circular visualization of the data science workflow

// Canvas dimensions
let canvasWidth = 700;
let drawHeight = 550;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let margin = 25;
let defaultTextSize = 16;

// Circle layout
let centerX, centerY;
let circleRadius = 180;
let nodeRadius = 45;

// Animation
let glowingStage = 0;
let glowTimer = 0;
let glowInterval = 120; // frames between stage changes
let glowPulse = 0;

// Interaction state
let hoveredStage = -1;
let selectedStage = -1;
let showExamples = false;

// Journey stages data
const stages = [
  {
    title: "PROBLEM DEFINITION",
    subtitle: "The Call to Adventure",
    icon: "Define\nProblem",
    color: [128, 0, 128], // Purple
    hoverText: "A mystery emerges! What question burns in your mind?",
    examples: [
      "Which factors predict student success?",
      "Can we forecast product demand?",
      "What causes customer churn?"
    ]
  },
  {
    title: "DATA COLLECTION",
    subtitle: "Gathering Allies",
    icon: "Gather\nData",
    color: [218, 165, 32], // Gold
    hoverText: "Seek out the data you need from databases, surveys, and APIs",
    examples: [
      "Scraping weather data from APIs",
      "Conducting customer surveys",
      "Extracting sales from databases"
    ]
  },
  {
    title: "DATA CLEANING",
    subtitle: "Trials and Tribulations",
    icon: "Clean\nData",
    color: [255, 140, 0], // Orange
    hoverText: "Face the chaos! Fix errors, handle missing values, tame the mess",
    examples: [
      "Removing duplicate records",
      "Imputing missing values",
      "Fixing inconsistent formats"
    ]
  },
  {
    title: "EXPLORATORY ANALYSIS",
    subtitle: "The Revelation",
    icon: "Explore\nData",
    color: [30, 144, 255], // Blue
    hoverText: "Visualize and explore. Patterns begin to emerge from the fog",
    examples: [
      "Creating distribution plots",
      "Finding correlations",
      "Detecting outliers visually"
    ]
  },
  {
    title: "MODELING",
    subtitle: "Forging the Weapon",
    icon: "Build\nModel",
    color: [220, 20, 60], // Red
    hoverText: "Build your predictive model - your weapon against uncertainty",
    examples: [
      "Training a regression model",
      "Building a neural network",
      "Creating decision trees"
    ]
  },
  {
    title: "EVALUATION",
    subtitle: "The Ultimate Test",
    icon: "Evaluate\nModel",
    color: [34, 139, 34], // Green
    hoverText: "Does your model actually work? Time to find out!",
    examples: [
      "Calculating accuracy metrics",
      "Cross-validation testing",
      "A/B testing in production"
    ]
  },
  {
    title: "COMMUNICATION",
    subtitle: "Return with the Elixir",
    icon: "Share\nResults",
    color: [0, 139, 139], // Teal
    hoverText: "Share your discoveries! Tell the story your data revealed",
    examples: [
      "Creating executive dashboards",
      "Writing technical reports",
      "Presenting to stakeholders"
    ]
  }
];

// Return arrows data
const returnArrows = [
  { from: 5, to: 1, label: "Need more data?" },
  { from: 4, to: 2, label: "Model not working?" },
  { from: 6, to: 0, label: "New questions?" }
];

let animateCheckbox;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  centerX = canvasWidth / 2;
  centerY = drawHeight / 2 + 20;

  // Create animation toggle checkbox
  animateCheckbox = createCheckbox(' Auto-animate stages', true);
  animateCheckbox.position(10, drawHeight + 15);
  animateCheckbox.style('font-size', '16px');

  describe('Interactive circular diagram showing the 7 stages of the data science workflow as a hero\'s journey. Hover over stages for descriptions, click for examples.', LABEL);
}

function draw() {
  updateCanvasSize();

  // Drawing area
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Control area
  fill('white');
  noStroke();
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Update center position for responsive design
  centerX = canvasWidth / 2;

  // Adjust circle radius based on canvas width
  circleRadius = min(180, canvasWidth / 4);
  nodeRadius = min(45, canvasWidth / 16);

  // Draw title
  fill(50);
  noStroke();
  textSize(24);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text("Data Science Hero's Journey", canvasWidth / 2, 10);
  textStyle(NORMAL);

  // Draw center quest label
  drawCenterQuest();

  // Draw return arrows (dotted, behind nodes)
  drawReturnArrows();

  // Draw forward arrows between stages
  drawForwardArrows();

  // Draw stage nodes
  drawStageNodes();

  // Draw hover infobox
  if (hoveredStage >= 0) {
    drawInfoBox(hoveredStage);
  }

  // Draw examples panel if a stage is selected
  if (selectedStage >= 0 && showExamples) {
    drawExamplesPanel(selectedStage);
  }

  // Update animation
  if (animateCheckbox.checked()) {
    glowTimer++;
    glowPulse = sin(frameCount * 0.1) * 0.3 + 0.7;
    if (glowTimer >= glowInterval) {
      glowTimer = 0;
      glowingStage = (glowingStage + 1) % stages.length;
    }
  }

  // Draw instructions in control area
  fill(80);
  textSize(14);
  textAlign(RIGHT, CENTER);
  text("Hover for details | Click for examples", canvasWidth - 15, drawHeight + 30);
  textAlign(LEFT, CENTER);
}

function drawCenterQuest() {
  // Draw glowing center circle
  let pulseSize = 60 + sin(frameCount * 0.05) * 5;

  // Outer glow
  noStroke();
  for (let i = 3; i > 0; i--) {
    fill(100, 100, 200, 30);
    ellipse(centerX, centerY, pulseSize + i * 15, pulseSize + i * 15);
  }

  // Main circle
  fill(70, 70, 140);
  stroke(100, 100, 180);
  strokeWeight(3);
  ellipse(centerX, centerY, pulseSize, pulseSize);

  // Text
  fill(255);
  noStroke();
  textSize(11);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  text("DATA", centerX, centerY - 10);
  text("SCIENCE", centerX, centerY + 2);
  text("QUEST", centerX, centerY + 14);
  textStyle(NORMAL);
}

function drawStageNodes() {
  for (let i = 0; i < stages.length; i++) {
    let angle = -PI / 2 + (TWO_PI / stages.length) * i;
    let x = centerX + cos(angle) * circleRadius;
    let y = centerY + sin(angle) * circleRadius;

    let stage = stages[i];
    let isHovered = (hoveredStage === i);
    let isGlowing = (glowingStage === i && animateCheckbox.checked());
    let isSelected = (selectedStage === i);

    // Draw glow effect for animated stage
    if (isGlowing) {
      noStroke();
      for (let g = 4; g > 0; g--) {
        fill(stage.color[0], stage.color[1], stage.color[2], 40 * glowPulse);
        ellipse(x, y, nodeRadius * 2 + g * 10, nodeRadius * 2 + g * 10);
      }
    }

    // Draw node circle
    let nodeSizeMultiplier = isHovered ? 1.15 : 1;

    // Shadow
    noStroke();
    fill(0, 0, 0, 30);
    ellipse(x + 3, y + 3, nodeRadius * 2 * nodeSizeMultiplier);

    // Main circle
    fill(stage.color[0], stage.color[1], stage.color[2]);
    stroke(255);
    strokeWeight(isHovered || isSelected ? 4 : 2);
    ellipse(x, y, nodeRadius * 2 * nodeSizeMultiplier);

    // Icon
    fill(255);
    noStroke();
    // Use smaller text for multi-line icons
    let iconSize = stage.icon.includes('\n') ? nodeRadius * 0.35 : nodeRadius * 0.5;
    textSize(iconSize);
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    text(stage.icon, x, y);
    textStyle(NORMAL);

    // Stage number badge
    fill(50);
    stroke(255);
    strokeWeight(2);
    let badgeX = x + nodeRadius * 0.7;
    let badgeY = y - nodeRadius * 0.7;
    ellipse(badgeX, badgeY, 22, 22);
    fill(255);
    noStroke();
    textSize(12);
    text(i + 1, badgeX, badgeY);

    // Label below node
    fill(50);
    textSize(11);
    textAlign(CENTER, TOP);
    let labelY = y + nodeRadius + 8;

    // Adjust label position for bottom nodes
    if (i >= 2 && i <= 4) {
      labelY = y + nodeRadius + 5;
    }

    textStyle(BOLD);
    text(stage.subtitle, x, labelY);
    textStyle(NORMAL);
  }
}

function drawForwardArrows() {
  for (let i = 0; i < stages.length; i++) {
    let nextI = (i + 1) % stages.length;

    let angle1 = -PI / 2 + (TWO_PI / stages.length) * i;
    let angle2 = -PI / 2 + (TWO_PI / stages.length) * nextI;

    let x1 = centerX + cos(angle1) * circleRadius;
    let y1 = centerY + sin(angle1) * circleRadius;
    let x2 = centerX + cos(angle2) * circleRadius;
    let y2 = centerY + sin(angle2) * circleRadius;

    // Calculate arrow start and end points (at edge of nodes)
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = sqrt(dx * dx + dy * dy);
    let nx = dx / dist;
    let ny = dy / dist;

    let startX = x1 + nx * (nodeRadius + 5);
    let startY = y1 + ny * (nodeRadius + 5);
    let endX = x2 - nx * (nodeRadius + 10);
    let endY = y2 - ny * (nodeRadius + 10);

    // Draw curved arrow
    stroke(100, 100, 100);
    strokeWeight(2);
    noFill();

    // Calculate control point for curve (slight outward bend)
    let midAngle = (angle1 + angle2) / 2;
    let ctrlX = centerX + cos(midAngle) * (circleRadius + 25);
    let ctrlY = centerY + sin(midAngle) * (circleRadius + 25);

    beginShape();
    vertex(startX, startY);
    quadraticVertex(ctrlX, ctrlY, endX, endY);
    endShape();

    // Arrowhead
    let arrowAngle = atan2(endY - ctrlY, endX - ctrlX);
    fill(100);
    noStroke();
    push();
    translate(endX, endY);
    rotate(arrowAngle);
    triangle(0, 0, -10, -5, -10, 5);
    pop();
  }
}

function drawReturnArrows() {
  for (let arr of returnArrows) {
    let angle1 = -PI / 2 + (TWO_PI / stages.length) * arr.from;
    let angle2 = -PI / 2 + (TWO_PI / stages.length) * arr.to;

    let x1 = centerX + cos(angle1) * circleRadius;
    let y1 = centerY + sin(angle1) * circleRadius;
    let x2 = centerX + cos(angle2) * circleRadius;
    let y2 = centerY + sin(angle2) * circleRadius;

    // Calculate control point for inner curved arrow
    let midAngle = (angle1 + angle2) / 2;
    // Adjust for wraparound
    if (abs(angle1 - angle2) > PI) {
      midAngle += PI;
    }
    let ctrlDist = circleRadius * 0.4;
    let ctrlX = centerX + cos(midAngle) * ctrlDist;
    let ctrlY = centerY + sin(midAngle) * ctrlDist;

    // Calculate start and end points
    let dx1 = ctrlX - x1;
    let dy1 = ctrlY - y1;
    let dist1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let startX = x1 + (dx1 / dist1) * (nodeRadius + 5);
    let startY = y1 + (dy1 / dist1) * (nodeRadius + 5);

    let dx2 = ctrlX - x2;
    let dy2 = ctrlY - y2;
    let dist2 = sqrt(dx2 * dx2 + dy2 * dy2);
    let endX = x2 + (dx2 / dist2) * (nodeRadius + 10);
    let endY = y2 + (dy2 / dist2) * (nodeRadius + 10);

    // Draw dotted curved line
    stroke(150, 100, 100);
    strokeWeight(2);
    drawingContext.setLineDash([5, 5]);
    noFill();

    beginShape();
    vertex(startX, startY);
    quadraticVertex(ctrlX, ctrlY, endX, endY);
    endShape();

    drawingContext.setLineDash([]);

    // Arrowhead
    let arrowAngle = atan2(endY - ctrlY, endX - ctrlX);
    fill(150, 100, 100);
    noStroke();
    push();
    translate(endX, endY);
    rotate(arrowAngle);
    triangle(0, 0, -8, -4, -8, 4);
    pop();

    // Label on arrow
    fill(120, 80, 80);
    textSize(10);
    textAlign(CENTER, CENTER);
    noStroke();

    // Position label near control point
    let labelX = ctrlX;
    let labelY = ctrlY;

    // Background for label
    let labelWidth = textWidth(arr.label) + 8;
    fill(255, 255, 255, 220);
    stroke(200);
    strokeWeight(1);
    rectMode(CENTER);
    rect(labelX, labelY, labelWidth, 16, 3);
    rectMode(CORNER);

    fill(120, 80, 80);
    noStroke();
    textStyle(ITALIC);
    text(arr.label, labelX, labelY);
    textStyle(NORMAL);
  }
}

function drawInfoBox(stageIndex) {
  let stage = stages[stageIndex];
  let angle = -PI / 2 + (TWO_PI / stages.length) * stageIndex;
  let nodeX = centerX + cos(angle) * circleRadius;
  let nodeY = centerY + sin(angle) * circleRadius;

  // Determine box position
  let boxWidth = 220;
  let boxHeight = 85;
  let boxX = nodeX;
  let boxY = nodeY;

  // Adjust position to stay on screen
  if (nodeX > canvasWidth / 2) {
    boxX = nodeX - boxWidth - nodeRadius - 10;
  } else {
    boxX = nodeX + nodeRadius + 10;
  }

  if (nodeY < 100) {
    boxY = nodeY + 20;
  } else if (nodeY > drawHeight - 100) {
    boxY = nodeY - boxHeight - 20;
  } else {
    boxY = nodeY - boxHeight / 2;
  }

  // Keep box on screen
  boxX = constrain(boxX, 10, canvasWidth - boxWidth - 10);
  boxY = constrain(boxY, 50, drawHeight - boxHeight - 10);

  // Draw box with shadow
  noStroke();
  fill(0, 0, 0, 30);
  rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 10);

  // Main box
  fill(255, 255, 255, 245);
  stroke(stage.color[0], stage.color[1], stage.color[2]);
  strokeWeight(3);
  rect(boxX, boxY, boxWidth, boxHeight, 10);

  // Title
  fill(stage.color[0], stage.color[1], stage.color[2]);
  noStroke();
  textSize(13);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  text(stage.title, boxX + 12, boxY + 10);

  // Subtitle
  fill(100);
  textSize(11);
  textStyle(ITALIC);
  text(stage.subtitle, boxX + 12, boxY + 28);
  textStyle(NORMAL);

  // Description
  fill(60);
  textSize(12);
  textStyle(NORMAL);

  // Word wrap the hover text
  let words = stage.hoverText.split(' ');
  let line = '';
  let lineY = boxY + 48;
  let maxWidth = boxWidth - 24;

  for (let word of words) {
    let testLine = line + word + ' ';
    if (textWidth(testLine) > maxWidth) {
      text(line, boxX + 12, lineY);
      line = word + ' ';
      lineY += 16;
    } else {
      line = testLine;
    }
  }
  text(line, boxX + 12, lineY);
}

function drawExamplesPanel(stageIndex) {
  let stage = stages[stageIndex];

  let panelWidth = 250;
  let panelHeight = 130;
  let panelX = canvasWidth - panelWidth - 15;
  let panelY = drawHeight - panelHeight - 15;

  // Shadow
  noStroke();
  fill(0, 0, 0, 30);
  rect(panelX + 4, panelY + 4, panelWidth, panelHeight, 10);

  // Panel
  fill(255, 255, 255, 250);
  stroke(stage.color[0], stage.color[1], stage.color[2]);
  strokeWeight(3);
  rect(panelX, panelY, panelWidth, panelHeight, 10);

  // Close button
  fill(200);
  noStroke();
  ellipse(panelX + panelWidth - 15, panelY + 15, 20, 20);
  fill(100);
  textSize(14);
  textAlign(CENTER, CENTER);
  text("x", panelX + panelWidth - 15, panelY + 14);

  // Title
  fill(stage.color[0], stage.color[1], stage.color[2]);
  textSize(13);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  text("Real-World Examples:", panelX + 12, panelY + 12);
  textStyle(NORMAL);

  // Examples list
  fill(60);
  textSize(12);
  for (let i = 0; i < stage.examples.length; i++) {
    let bullet = String.fromCharCode(8226); // bullet character
    text(bullet + " " + stage.examples[i], panelX + 15, panelY + 38 + i * 28);
  }
}

function mouseMoved() {
  hoveredStage = -1;

  for (let i = 0; i < stages.length; i++) {
    let angle = -PI / 2 + (TWO_PI / stages.length) * i;
    let x = centerX + cos(angle) * circleRadius;
    let y = centerY + sin(angle) * circleRadius;

    let d = dist(mouseX, mouseY, x, y);
    if (d < nodeRadius) {
      hoveredStage = i;
      break;
    }
  }
}

function mousePressed() {
  // Check if clicking close button on examples panel
  if (selectedStage >= 0 && showExamples) {
    let panelWidth = 250;
    let panelHeight = 130;
    let panelX = canvasWidth - panelWidth - 15;
    let panelY = drawHeight - panelHeight - 15;
    let closeX = panelX + panelWidth - 15;
    let closeY = panelY + 15;

    if (dist(mouseX, mouseY, closeX, closeY) < 12) {
      showExamples = false;
      selectedStage = -1;
      return;
    }
  }

  // Check if clicking on a stage
  for (let i = 0; i < stages.length; i++) {
    let angle = -PI / 2 + (TWO_PI / stages.length) * i;
    let x = centerX + cos(angle) * circleRadius;
    let y = centerY + sin(angle) * circleRadius;

    let d = dist(mouseX, mouseY, x, y);
    if (d < nodeRadius) {
      if (selectedStage === i && showExamples) {
        showExamples = false;
        selectedStage = -1;
      } else {
        selectedStage = i;
        showExamples = true;
        glowingStage = i;
      }
      return;
    }
  }

  // Click elsewhere closes examples
  showExamples = false;
  selectedStage = -1;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    centerX = canvasWidth / 2;
  }
}
