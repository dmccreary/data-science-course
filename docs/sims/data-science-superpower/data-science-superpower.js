// Data Science Superpower Concept Map MicroSim
// Hub-and-spoke visualization showing data science superpowers and applications

// Canvas dimensions
let canvasWidth = 700;
let drawHeight = 550;
let controlHeight = 50;
let canvasHeight = drawHeight + controlHeight;
let margin = 25;
let defaultTextSize = 16;

// Layout
let centerX, centerY;
let innerRadius = 120;  // Distance from center to superpowers
let outerRadius = 220;  // Distance from center to applications
let nodeSize = 58;      // Base size for superpower nodes (30% larger than original 45)

// Animation
let pulsePhase = 0;
let actionLinePhase = 0;

// Interaction state
let hoveredSuperpower = -1;
let hoveredApplication = -1;

// Color scheme
const colors = {
  core: [30, 100, 200],        // Blue core
  coreGlow: [100, 150, 255],   // Light blue glow
  spoke: [255, 200, 50],       // Yellow/gold spokes
  spokeHover: [255, 220, 100], // Brighter gold on hover
  app: [50, 180, 100],         // Green application nodes
  appHover: [80, 220, 130],    // Brighter green on hover
  actionLine: [255, 100, 50],  // Orange action lines
  text: [40, 40, 40]
};

// Superpowers data
const superpowers = [
  {
    name: "Pattern\nDetection",
    description: "Find hidden patterns and anomalies in complex data",
    applications: [
      { name: "Fraud\nDetection", companies: "PayPal, Stripe, Visa" },
      { name: "Disease\nDiagnosis", companies: "IBM Watson, Google Health" }
    ]
  },
  {
    name: "Prediction",
    description: "Forecast future events and trends from historical data",
    applications: [
      { name: "Weather\nForecasting", companies: "The Weather Company, AccuWeather" },
      { name: "Stock\nMarkets", companies: "Bloomberg, Renaissance Technologies" }
    ]
  },
  {
    name: "Optimization",
    description: "Find the best solution among many possibilities",
    applications: [
      { name: "Route\nPlanning", companies: "Google Maps, UPS, FedEx" },
      { name: "Resource\nAllocation", companies: "Amazon, Uber, Airbnb" }
    ]
  },
  {
    name: "Automation",
    description: "Enable machines to perform tasks intelligently",
    applications: [
      { name: "Self-Driving\nCars", companies: "Tesla, Waymo, Cruise" },
      { name: "Smart\nAssistants", companies: "Amazon Alexa, Google Assistant" }
    ]
  },
  {
    name: "Insight\nDiscovery",
    description: "Uncover meaningful insights from raw information",
    applications: [
      { name: "Customer\nBehavior", companies: "Netflix, Spotify, Amazon" },
      { name: "Scientific\nResearch", companies: "DeepMind, OpenAI" }
    ]
  },
  {
    name: "Decision\nSupport",
    description: "Provide evidence-based recommendations for choices",
    applications: [
      { name: "Business\nStrategy", companies: "McKinsey, Palantir" },
      { name: "Policy\nMaking", companies: "World Bank, CDC" }
    ]
  }
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  describe('Interactive hub-and-spoke diagram showing Data Science at the center with six superpowers radiating outward to real-world applications. Hover for details.', LABEL);
}

function draw() {
  updateCanvasSize();

  // Drawing area with gradient-like background
  drawBackground();

  // Control area
  fill('white');
  noStroke();
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Update animation phases
  pulsePhase += 0.03;
  actionLinePhase += 0.05;

  // Draw action lines (comic book style)
  drawActionLines();

  // Draw spokes (connections)
  drawSpokes();

  // Draw application nodes
  drawApplicationNodes();

  // Draw superpower nodes
  drawSuperpowerNodes();

  // Draw central core
  drawCentralCore();

  // Draw title
  fill(colors.text);
  noStroke();
  textSize(22);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text("Data Science Superpowers", canvasWidth / 2, 8);
  textStyle(NORMAL);

  // Draw hover info box
  if (hoveredSuperpower >= 0) {
    drawSuperpowerInfo(hoveredSuperpower);
  } else if (hoveredApplication >= 0) {
    drawApplicationInfo(hoveredApplication);
  }

  // Draw instructions
  fill(80);
  textSize(14);
  textAlign(CENTER, CENTER);
  text("Hover over nodes to learn more", canvasWidth / 2, drawHeight + 25);
}

function drawBackground() {
  // Light gradient-like background
  fill(240, 245, 255);
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Subtle radial lines for comic effect
  stroke(220, 230, 245);
  strokeWeight(1);
  for (let i = 0; i < 24; i++) {
    let angle = (TWO_PI / 24) * i;
    let x1 = centerX + cos(angle) * 50;
    let y1 = centerY + sin(angle) * 50;
    let x2 = centerX + cos(angle) * max(canvasWidth, drawHeight);
    let y2 = centerY + sin(angle) * max(canvasWidth, drawHeight);
    line(x1, y1, x2, y2);
  }
}

function drawActionLines() {
  // Comic book style action lines radiating from center
  stroke(colors.actionLine[0], colors.actionLine[1], colors.actionLine[2], 30);
  strokeWeight(3);

  for (let i = 0; i < 12; i++) {
    let angle = (TWO_PI / 12) * i + actionLinePhase;
    let len = 30 + sin(actionLinePhase * 2 + i) * 10;
    let startDist = 45 + sin(pulsePhase + i) * 5;
    let x1 = centerX + cos(angle) * startDist;
    let y1 = centerY + sin(angle) * startDist;
    let x2 = centerX + cos(angle) * (startDist + len);
    let y2 = centerY + sin(angle) * (startDist + len);
    line(x1, y1, x2, y2);
  }
}

function drawCentralCore() {
  let pulseSize = 40 + sin(pulsePhase) * 5;

  // Outer glow rings
  noStroke();
  for (let i = 4; i > 0; i--) {
    let alpha = 40 - i * 8;
    fill(colors.coreGlow[0], colors.coreGlow[1], colors.coreGlow[2], alpha);
    ellipse(centerX, centerY, pulseSize * 2 + i * 20, pulseSize * 2 + i * 20);
  }

  // Main core circle
  fill(colors.core);
  stroke(255);
  strokeWeight(4);
  ellipse(centerX, centerY, pulseSize * 2, pulseSize * 2);

  // Inner highlight
  noStroke();
  fill(255, 255, 255, 60);
  ellipse(centerX - 10, centerY - 10, pulseSize * 0.8, pulseSize * 0.8);

  // Text
  fill(255);
  noStroke();
  textSize(13);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  text("DATA", centerX, centerY - 10);
  text("SCIENCE", centerX, centerY + 8);
  textStyle(NORMAL);
}

function drawSpokes() {
  for (let i = 0; i < superpowers.length; i++) {
    let angle = -PI / 2 + (TWO_PI / superpowers.length) * i;
    let superpowerX = centerX + cos(angle) * innerRadius;
    let superpowerY = centerY + sin(angle) * innerRadius;

    // Draw spoke from center to superpower
    let isHovered = (hoveredSuperpower === i);
    strokeWeight(isHovered ? 4 : 3);
    stroke(colors.spoke[0], colors.spoke[1], colors.spoke[2], isHovered ? 255 : 180);
    line(centerX, centerY, superpowerX, superpowerY);

    // Draw connections to applications
    for (let j = 0; j < 2; j++) {
      let appAngle = angle + (j === 0 ? -0.25 : 0.25);
      let appX = centerX + cos(appAngle) * outerRadius;
      let appY = centerY + sin(appAngle) * outerRadius;

      let appIndex = i * 2 + j;
      let appHovered = (hoveredApplication === appIndex);

      strokeWeight(appHovered ? 3 : 2);
      stroke(colors.app[0], colors.app[1], colors.app[2], appHovered ? 255 : 150);
      line(superpowerX, superpowerY, appX, appY);
    }
  }
}

function drawSuperpowerNodes() {
  for (let i = 0; i < superpowers.length; i++) {
    let angle = -PI / 2 + (TWO_PI / superpowers.length) * i;
    let x = centerX + cos(angle) * innerRadius;
    let y = centerY + sin(angle) * innerRadius;

    let isHovered = (hoveredSuperpower === i);
    let baseSize = nodeSize * 1.1;  // Superpower nodes are 10% larger
    let currentSize = isHovered ? baseSize * 1.1 : baseSize;

    // Shadow
    noStroke();
    fill(0, 0, 0, 30);
    ellipse(x + 3, y + 3, currentSize, currentSize);

    // Node with comic book style border
    let col = isHovered ? colors.spokeHover : colors.spoke;
    fill(col);
    stroke(50);
    strokeWeight(3);
    ellipse(x, y, currentSize, currentSize);

    // Text
    fill(40);
    noStroke();
    textSize(10);
    textAlign(CENTER, CENTER);
    textStyle(BOLD);
    text(superpowers[i].name, x, y);
    textStyle(NORMAL);
  }
}

function drawApplicationNodes() {
  for (let i = 0; i < superpowers.length; i++) {
    let baseAngle = -PI / 2 + (TWO_PI / superpowers.length) * i;

    for (let j = 0; j < 2; j++) {
      let appAngle = baseAngle + (j === 0 ? -0.25 : 0.25);
      let x = centerX + cos(appAngle) * outerRadius;
      let y = centerY + sin(appAngle) * outerRadius;

      let appIndex = i * 2 + j;
      let isHovered = (hoveredApplication === appIndex);
      let appNodeSize = nodeSize * 0.9;  // Application nodes are 90% of base nodeSize
      let currentSize = isHovered ? appNodeSize * 1.1 : appNodeSize;

      // Shadow
      noStroke();
      fill(0, 0, 0, 25);
      ellipse(x + 2, y + 2, currentSize, currentSize);

      // Node
      let col = isHovered ? colors.appHover : colors.app;
      fill(col);
      stroke(30);
      strokeWeight(2);
      ellipse(x, y, currentSize, currentSize);

      // Text
      fill(255);
      noStroke();
      textSize(8);
      textAlign(CENTER, CENTER);
      textStyle(BOLD);
      text(superpowers[i].applications[j].name, x, y);
      textStyle(NORMAL);
    }
  }
}

function drawSuperpowerInfo(index) {
  let sp = superpowers[index];

  let boxWidth = 240;
  let boxHeight = 75;
  let boxX = canvasWidth / 2 - boxWidth / 2;
  let boxY = drawHeight - boxHeight - 15;

  // Shadow
  noStroke();
  fill(0, 0, 0, 30);
  rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 10);

  // Box
  fill(255, 250, 230);
  stroke(colors.spoke);
  strokeWeight(3);
  rect(boxX, boxY, boxWidth, boxHeight, 10);

  // Title
  fill(colors.spoke[0] - 50, colors.spoke[1] - 50, 0);
  noStroke();
  textSize(14);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  let titleText = sp.name.replace('\n', ' ');
  text(titleText, boxX + boxWidth / 2, boxY + 10);
  textStyle(NORMAL);

  // Description with word wrap
  fill(60);
  textSize(11);
  textAlign(LEFT, TOP);
  textWrap(WORD);
  text(sp.description, boxX + 10, boxY + 32, boxWidth - 20);
}

function drawApplicationInfo(index) {
  let spIndex = floor(index / 2);
  let appIndex = index % 2;
  let app = superpowers[spIndex].applications[appIndex];

  let boxWidth = 240;
  let boxHeight = 75;
  let boxX = canvasWidth / 2 - boxWidth / 2;
  let boxY = drawHeight - boxHeight - 15;

  // Shadow
  noStroke();
  fill(0, 0, 0, 30);
  rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 10);

  // Box
  fill(230, 255, 240);
  stroke(colors.app);
  strokeWeight(3);
  rect(boxX, boxY, boxWidth, boxHeight, 10);

  // Title
  fill(20, 100, 60);
  noStroke();
  textSize(13);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  let titleText = app.name.replace('\n', ' ');
  text(titleText, boxX + boxWidth / 2, boxY + 10);
  textStyle(NORMAL);

  // Companies with word wrap
  fill(60);
  textSize(11);
  textAlign(LEFT, TOP);
  textWrap(WORD);
  text("Examples: " + app.companies, boxX + 10, boxY + 32, boxWidth - 20);
}

function mouseMoved() {
  hoveredSuperpower = -1;
  hoveredApplication = -1;

  // Check superpowers
  for (let i = 0; i < superpowers.length; i++) {
    let angle = -PI / 2 + (TWO_PI / superpowers.length) * i;
    let x = centerX + cos(angle) * innerRadius;
    let y = centerY + sin(angle) * innerRadius;

    if (dist(mouseX, mouseY, x, y) < 25) {
      hoveredSuperpower = i;
      return;
    }
  }

  // Check applications
  for (let i = 0; i < superpowers.length; i++) {
    let baseAngle = -PI / 2 + (TWO_PI / superpowers.length) * i;

    for (let j = 0; j < 2; j++) {
      let appAngle = baseAngle + (j === 0 ? -0.25 : 0.25);
      let x = centerX + cos(appAngle) * outerRadius;
      let y = centerY + sin(appAngle) * outerRadius;

      let appIndex = i * 2 + j;
      if (dist(mouseX, mouseY, x, y) < 20) {
        hoveredApplication = appIndex;
        return;
      }
    }
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
    updateLayoutDimensions();
  }
}

function updateLayoutDimensions() {
  centerX = canvasWidth / 2;
  centerY = drawHeight / 2 + 10;
  innerRadius = min(120, canvasWidth * 0.17);
  outerRadius = min(220, canvasWidth * 0.31);
  nodeSize = min(58, canvasWidth * 0.083);  // 30% larger than original 45
}
