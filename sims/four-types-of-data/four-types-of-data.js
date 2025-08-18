// Four Types of Data in Data Science MicroSim
// Canvas dimensions
// rewritten in the resize
let canvasWidth = 550;
let drawHeight = 500;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;
let defaultTextSize = 16;

// Global variables for responsive design
let containerWidth; // calculated by container upon resize
let containerHeight = canvasHeight; // fixed height on page

// Variables for the visualization
let currentHover = -1;
let quadrants = [];
let graphNodes = [];
let graphEdges = [];
let imageColors = []; // Fixed colors for image grid

function setup() {
  // Create a canvas to match the parent container's size
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  textSize(defaultTextSize);
  
  // Initialize quadrants
  initializeQuadrants();
  
  // Generate fixed colors for image grid
  generateImageColors();
  
  // Generate graph data
  generateGraphData();
  
  describe('Four Types of Data in Data Science infographic showing Images, Sequences, Tabular data, and Graph data with visual representations.', LABEL);
}

function initializeQuadrants() {
  let quadWidth = (canvasWidth - 3 * margin) / 2;
  let quadHeight = (drawHeight - 70) / 2; // Leave space for title
  
  quadrants = [
    {
      name: "Images",
      x: margin,
      y: 50,
      w: quadWidth,
      h: quadHeight,
      color: "#FFE0E0",
      description: "Images: Visual data represented as pixel arrays with RGB color values. Each pixel contains red, green, and blue color components. Common in computer vision, medical imaging, satellite imagery, and photo recognition. Neural networks like CNNs are specifically designed to process this type of spatial data structure."
    },
    {
      name: "Sequences",
      x: margin * 2 + quadWidth,
      y: 50,
      w: quadWidth,
      h: quadHeight,
      color: "#E0FFE0",
      description: "Sequences: Ordered data where position and timing matter critically. Examples include time series data, natural language text, DNA sequences, audio signals, and stock prices. RNNs, LSTMs, and Transformers are designed to capture temporal dependencies and patterns in sequential data."
    },
    {
      name: "Tabular",
      x: margin,
      y: 50 + quadHeight + margin,
      w: quadWidth,
      h: quadHeight,
      color: "#E0E0FF",
      description: "Tabular: Structured data organized in rows and columns, similar to spreadsheets or databases. Each row represents an observation and each column represents a feature or variable. This is the most common data type in traditional machine learning, handled well by algorithms like random forests, SVM, and gradient boosting.  We can use Python data frames to manipulate tabular data."
    },
    {
      name: "Graph",
      x: margin * 2 + quadWidth,
      y: 50 + quadHeight + margin,
      w: quadWidth,
      h: quadHeight,
      color: "#FFFFE0",
      description: "Graph: Network data representing relationships and connections between entities. Nodes represent objects (people, websites, molecules) while edges represent relationships (friendships, links, bonds). Used in social network analysis, recommendation systems, knowledge graphs, and molecular modeling. Graph Neural Networks (GNNs) are specialized for this data type."
    }
  ];
}

function generateImageColors() {
  // Generate fixed colors for the 5x5 image grid to prevent flickering
  imageColors = [];
  for (let i = 0; i < 25; i++) {
    imageColors.push(color(150 + (i * 7) % 105, 150 + (i * 11) % 105, 150 + (i * 13) % 105));
  }
}

function generateGraphData() {
  // Generate 25 nodes with random positions and colors
  graphNodes = [];
  let quad = quadrants[3]; // Graph quadrant
  let nodeMargin = 20;
  
  for (let i = 0; i < 25; i++) {
    graphNodes.push({
      x: quad.x + nodeMargin + random(quad.w - 2 * nodeMargin),
      y: quad.y + nodeMargin + 10 + random(quad.h - nodeMargin) * .9,
      color: color(random(100, 255), random(100, 255), random(100, 255)),
      id: i
    });
  }
  
  // Generate random edges
  graphEdges = [];
  for (let i = 0; i < 35; i++) {
    let from = floor(random(25));
    let to = floor(random(25));
    if (from !== to) {
      graphEdges.push({ from: from, to: to });
    }
  }
}

function draw() {
  // Draw area
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  
  // Controls area
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(0, drawHeight, canvasWidth, controlHeight);
  
  // Title
  fill('black');
  noStroke();
  textSize(24);
  textAlign(CENTER, TOP);
  text("Four Types of Data in Data Science", canvasWidth/2, margin/2);
  
  // Check for hover
  checkHover();
  
  // Draw quadrants
  drawQuadrants();
  
  // Draw description
  drawDescription();
}

function checkHover() {
  currentHover = -1;
  
  // Check if mouse is over any quadrant
  for (let i = 0; i < quadrants.length; i++) {
    let quad = quadrants[i];
    if (mouseX >= quad.x && mouseX <= quad.x + quad.w && 
        mouseY >= quad.y && mouseY <= quad.y + quad.h) {
      currentHover = i;
      break;
    }
  }
}

function drawQuadrants() {
  for (let i = 0; i < quadrants.length; i++) {
    let quad = quadrants[i];
    
    // Highlight if hovered
    if (currentHover === i) {
      stroke('black');
      strokeWeight(3);
    } else {
      stroke('gray');
      strokeWeight(1);
    }
    
    // Draw quadrant background
    fill(quad.color);
    rect(quad.x, quad.y, quad.w, quad.h);
    
    // Draw quadrant title
    fill('black');
    noStroke();
    textSize(20);
    textAlign(CENTER, TOP);
    text(quad.name, quad.x + quad.w/2, quad.y + 10);
    
    // Draw quadrant-specific content
    drawQuadrantContent(i);
  }
}

function drawQuadrantContent(quadIndex) {
  let quad = quadrants[quadIndex];
  let contentY = quad.y + 40; // Start below title
  let contentHeight = quad.h - 50;
  
  switch(quadIndex) {
    case 0: // Images
      drawImageGrid(quad.x, contentY, quad.w, contentHeight);
      break;
    case 1: // Sequences
      drawSequences(quad.x, contentY, quad.w, contentHeight);
      break;
    case 2: // Tabular
      drawTable(quad.x, contentY, quad.w, contentHeight);
      break;
    case 3: // Graph
      drawGraph(quad.x, contentY, quad.w, contentHeight);
      break;
  }
}

function drawImageGrid(x, y, w, h) {
  let cellSize = min(w, h) / 6;
  let startX = x + (w - cellSize * 5) / 2;
  let startY = y + (h - cellSize * 5) / 2;
  
  let colorIndex = 0;
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      let cellX = startX + col * cellSize;
      let cellY = startY + row * cellSize;
      
      // Use fixed color instead of random to prevent flickering
      fill(imageColors[colorIndex]);
      stroke('black');
      strokeWeight(1);
      circle(cellX + cellSize/2, cellY + cellSize/2, cellSize * 0.8);
      
      // Draw RGB text
      fill('black');
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(8);
      text("RGB", cellX + cellSize/2, cellY + cellSize/2);
      
      colorIndex++;
    }
  }
}

function drawSequences(x, y, w, h) {
  let circleSize = 20;
  let spacing = 36;
  
  // Center the sequence grid in the quadrant
  let gridWidth = 5 * spacing - (spacing - circleSize);
  let gridHeight = 5 * spacing - (spacing - circleSize);
  let startX = x + (w - gridWidth) / 2;
  let startY = y + (h - gridHeight) / 2 + 5;
  
  let circleCount = 0;
  
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      let circleX = startX + col * spacing;
      let circleY = startY + row * spacing;
      
      // Draw circle for this node
      // Use this for color that changes with item
      // fill(100 + circleCount * 5, 150, 200);
      // blue circles with black border
      fill('blue');
      stroke('black');
      strokeWeight(1);
      circle(circleX, circleY, circleSize);
      
      // Draw number
      fill('white');
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(12);
      text(circleCount + 1, circleX, circleY);
      
      // Draw arrow to next circle (except for last circle)
      if (circleCount < 24) {
        let nextCol = (col + 1) % 5;
        let nextRow = row + (col === 4 ? 1 : 0);
        
        if (nextRow < 5) {
          let nextX = startX + nextCol * spacing;
          let nextY = startY + nextRow * spacing;
          
          // Draw arrow
          stroke('red');
          strokeWeight(2);
          
          if (col === 4) {
            // Arrow going down to next row
            drawArrow(circleX, circleY + circleSize/2, nextX, nextY - circleSize/2);
          } else {
            // Arrow going right
            drawArrow(circleX + circleSize/2, circleY, nextX - circleSize/2, nextY);
          }
        }
      }
      
      circleCount++;
    }
  }
}

function drawTable(x, y, w, h) {
  let cellW = (w - 40) / 5;
  let cellH = (h - 20) / 5;
  let tableX = x + 20;
  let tableY = y + 10;
  
  let columnColors = ['red', 'orange', 'yellow', 'green', 'blue'];
  
  for (let row = 0; row < 5; row++) {
    for (let col = 0; col < 5; col++) {
      let cellX = tableX + col * cellW;
      let cellY = tableY + row * cellH;
      
      // Header row (gray background)
      if (row === 0) {
        fill('lightgray');
      } else {
        fill('white');
      }
      
      stroke('black');
      strokeWeight(1);
      rect(cellX, cellY, cellW, cellH);
      
      // Column header colors
      if (row === 0) {
        fill(columnColors[col]);
        noStroke();
        rect(cellX + 2, cellY + 2, cellW - 4, cellH - 4);
        fill('gray');
        textAlign(CENTER, CENTER);
        textSize(14);
        text("Col " + col, cellX + 32, cellY + 15);
      } else {
        // Data cells
        fill('black');
        noStroke();
        textAlign(CENTER, CENTER);
        textSize(12);
        text(`${col},${row-1}`, cellX + cellW/2, cellY + cellH/2);
      }
    }
  }
}

function drawGraph(x, y, w, h) {
  // Draw edges first
  stroke('gray');
  strokeWeight(1);
  for (let edge of graphEdges) {
    let fromNode = graphNodes[edge.from];
    let toNode = graphNodes[edge.to];
    line(fromNode.x, fromNode.y, toNode.x, toNode.y);
  }
  
  // Draw nodes
  for (let node of graphNodes) {
    fill(node.color);
    stroke('black');
    strokeWeight(1);
    circle(node.x, node.y, 12);
  }
}

function drawArrow(x1, y1, x2, y2) {
  // Draw line
  line(x1, y1, x2, y2);
  
  // Draw arrowhead
  let angle = atan2(y2 - y1, x2 - x1);
  let arrowSize = 6;
  
  push();
  translate(x2, y2);
  rotate(angle);
  fill('red');
  noStroke();
  triangle(0, 0, -arrowSize, -arrowSize/2, -arrowSize, arrowSize/2);
  pop();
}

function drawDescription() {
  let descriptionY = drawHeight + 20;
  let descriptionHeight = controlHeight - 20;
  
  // Display description for hovered quadrant
  if (currentHover !== -1) {
    fill('black');
    noStroke();
    textSize(constrain(containerWidth * 0.03, 10, 15));
    textAlign(LEFT, TOP);
    
    // Draw description text with more space
    let descWidth = canvasWidth - 40;
    let description = quadrants[currentHover].description;
    text(description, 20, descriptionY, descWidth, descriptionHeight);
  } else {
    // Display instruction when no quadrant is hovered
    fill('#666666');
    noStroke();
    textSize(constrain(containerWidth * 0.025, 12, 16));
    textAlign(CENTER, CENTER);
    text("Hover over each quadrant to learn about different types of data in data science", 
         canvasWidth / 2, descriptionY + descriptionHeight / 4);
  }
}

function windowResized() {
  // Update canvas size when the container resizes
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  
  // Reinitialize quadrants and regenerate fixed colors
  initializeQuadrants();
  generateImageColors();
  generateGraphData();
  
  redraw();
}

function updateCanvasSize() {
  // Get the exact dimensions of the container
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);  // Avoid fractional pixels
  canvasWidth = containerWidth;
}