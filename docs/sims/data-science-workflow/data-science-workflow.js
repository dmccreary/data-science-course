// Interactive infographic showing a linear six-step data science workflow

// Global layout variables
let containerWidth;
let containerHeight = 290;
let canvasWidth = 800;

// Box and diagram styling
let boxes = [];
let currentHover = -1;
let lineStrokeWeight = 3;
let arrowSize = 10;

// Box dimensions - updated in updateLayout()
let boxHeight = 80;
// this will be reset to a percent of the drawing width (13%) for six steps
let boxWidth = 100;
const defaultTextSize = 14;
// the distance down to display the workflow
const workflowY = 60;

// Colors for each step
const stepColors = [
  "#e74c3c", // Red - Identify Problem
  "#f39c12", // Orange - Source Data
  "#f1c40f", // Yellow - Prepare Data
  "#2ecc71", // Green - Analyze Data
  "#3498db", // Blue - Visualize Data
  "#9b59b6", // Purple - Present Findings
];

// Workflow step definitions
const workflowSteps = [
  {
    label: "Identify\nProblem",
    color: stepColors[0],
    tcolor: "white",
    description:
      "Define the business question or research objective that needs to be answered. This involves understanding stakeholder needs, identifying success metrics, determining scope and constraints, and clarifying what specific insights or predictions are required. A well-defined problem statement guides all subsequent analysis decisions.",
  },
  {
    label: "Source\nData",
    color: stepColors[1],
    tcolor: "white",
    description:
      "Identify and acquire relevant data sources that can help answer the defined problem. This includes evaluating internal databases, external APIs, public datasets, surveys, or web scraping. Consider data quality, accessibility, legal restrictions, and cost. Document data lineage and establish data governance protocols.",
  },
  {
    label: "Prepare\nData",
    color: stepColors[2],
    tcolor: "black",
    description:
      "Clean, transform, and structure raw data for analysis. This involves handling missing values, removing duplicates, standardizing formats, creating derived variables, and merging datasets. Data preparation often takes 60-80% of project time but is crucial for reliable results.",
  },
  {
    label: "Analyze\nData",
    color: stepColors[3],
    tcolor: "white",
    description:
      "Apply statistical methods, machine learning algorithms, or other analytical techniques to extract insights from the prepared data. This includes exploratory data analysis, hypothesis testing, predictive modeling, clustering, or other methods appropriate to the problem type and data characteristics.",
  },
  {
    label: "Visualize\nData",
    color: stepColors[4],
    tcolor: "white",
    description:
      "Create charts, graphs, dashboards, and other visual representations that effectively communicate patterns, trends, and insights discovered in the analysis. Good visualizations make complex data accessible to stakeholders and support data-driven decision making.",
  },
  {
    label: "Present\nFindings",
    color: stepColors[5],
    tcolor: "white",
    description:
      "Communicate results to stakeholders through reports, presentations, or interactive tools. Translate technical findings into business language, provide actionable recommendations, discuss limitations and assumptions, and outline next steps. Tailor communication style to the audience.",
  },
];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector("main"));

  updateLayout();

  describe(
    "Linear workflow diagram showing 6 steps of data science process: Identify Problem, Source Data, Prepare Data, Analyze Data, Visualize Data, and Present Findings. Hover over steps for detailed descriptions.", LABEL);
}

function updateLayout() {
  // Responsive dimensions
  const margin = max(20, containerWidth * 0.03);
  
  // this needs to be adjusted so that the arrows are visible
  // it should be adjusted based on the number of steps
  boxWidth = containerWidth * 0.13;
  boxHeight = 70;

  if (containerWidth < 600) {
    boxWidth = containerWidth * 0.18;
    boxHeight = 60;
  }

  // Calculate spacing to distribute boxes evenly
  const totalBoxWidth = boxWidth * workflowSteps.length;
  const availableSpace = containerWidth - totalBoxWidth - margin * 2;
  const spacing = availableSpace / (workflowSteps.length - 1);

  // Position boxes
  boxes = [];
  for (let i = 0; i < workflowSteps.length; i++) {
    boxes.push({
      x: margin + i * (boxWidth + spacing),
      y: workflowY,
      w: boxWidth,
      h: boxHeight,
      ...workflowSteps[i],
    });
  }
}

function draw() {
  // Background
  fill("aliceblue");
  stroke("lightgray");
  strokeWeight(1);
  rect(0, 0, containerWidth, containerHeight);

  // Title
  let titleSize = constrain(containerWidth * 0.03, 16, 24);
  textSize(titleSize);
  textAlign(CENTER, TOP);
  fill(0);
  noStroke();
  text("Six-Step Data Science Workflow", containerWidth / 2, 20);

  // Draw workflow arrows
  drawWorkflowArrows();

  // Draw boxes
  drawBoxes();

  // Description area
  renderDescriptionBox();
}

function drawBoxes() {
  const fontSize = constrain(containerWidth * 0.025, 11, 18);

  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    const isHovered =
      mouseX >= b.x &&
      mouseX <= b.x + b.w &&
      mouseY >= b.y &&
      mouseY <= b.y + b.h;

    // Box styling
    stroke(isHovered ? "#2c3e50" : "#34495e");
    strokeWeight(isHovered ? 3 : 1.5);
    fill(b.color);
    rect(b.x, b.y, b.w, b.h, 8); // Rounded corners

    // Box number
    fill(255);
    noStroke();
    textSize(fontSize * 0.9);
    textAlign(LEFT, TOP);
    text(i + 1, b.x + 6, b.y + 6);

    // Box label
    fill(b.tcolor);
    textSize(fontSize);
    textAlign(CENTER, CENTER);
    text(b.label, b.x + b.w / 2, b.y + b.h/2 + 5);
  }
}

function drawWorkflowArrows() {
  strokeWeight(lineStrokeWeight);
  stroke("#34495e");

  for (let i = 0; i < boxes.length - 1; i++) {
    const from = boxes[i];
    const to = boxes[i + 1];

    const startX = from.x + from.w;
    const startY = from.y + from.h / 2;
    const endX = to.x;
    const endY = to.y + to.h / 2;

    drawArrow(startX, startY, endX, endY);
  }
}

function drawArrow(x1, y1, x2, y2) {
  line(x1, y1, x2, y2);

  // Arrowhead
  const angle = atan2(y2 - y1, x2 - x1);
  push();
  translate(x2, y2);
  rotate(angle);
  fill("#34495e");
  noStroke();
  triangle(
    -arrowSize * 1.5,
    -arrowSize * 0.6,
    -arrowSize * 1.5,
    arrowSize * 0.6,
    0,
    0
  );
  pop();
}

function renderDescriptionBox() {
  
  // InfoBox vertical spacing parameters
  // this is the vertical height down from the top edge of the infoBox
  const descriptionY = workflowY + boxHeight + 20;
  const descriptionHeight = 120;

  // Panel background
  fill(250);
  stroke(220);
  strokeWeight(1);
  rect(20, descriptionY, containerWidth - 40, descriptionHeight, 5);

  // Text content
  noStroke();
  fill(0);
  const fontSize = constrain(containerWidth * 0.02, 12, 16);
  textSize(fontSize);
  textAlign(LEFT, TOP);

  if (currentHover !== -1 && currentHover < boxes.length) {
    const step = boxes[currentHover];

    // Step title
    fill(step.color);
    textSize(fontSize * 1.2);
    text(
      `Step ${currentHover + 1}: ${step.label.replace("\n", " ")}`,
      30,
      descriptionY + 15
    );

    // Step description
    fill(0);
    textSize(fontSize);
    text(
      step.description,
      30,
      descriptionY + 45,
      containerWidth - 60,
      descriptionHeight - 60
    );
  } else {
    textAlign(CENTER, CENTER);
    fill("#7f8c8d");
    textSize(fontSize * 1.1);
    text(
      "Hover over any step to see detailed information about that step.",
      containerWidth / 2,
      descriptionY + descriptionHeight / 2
    );
  }
}

function mouseMoved() {
  currentHover = -1;

  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i];
    if (
      mouseX >= b.x &&
      mouseX <= b.x + b.w &&
      mouseY >= b.y &&
      mouseY <= b.y + b.h
    ) {
      currentHover = i;
      cursor("pointer");
      return;
    }
  }
  cursor("default");
}

function windowResized() {
  updateCanvasSize();
  updateLayout();
  resizeCanvas(containerWidth, containerHeight);
}

function updateCanvasSize() {
  const rect = document.querySelector("main").getBoundingClientRect();
  containerWidth = Math.floor(rect.width);
  canvasWidth = containerWidth;
}
