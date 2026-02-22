// Python Data Structures Comparison MicroSim
// Quiz game: given a scenario, choose the right data structure
// Bloom Level: Apply (L3) - decision-making with feedback
// MicroSim template version 2026.02

let canvasWidth = 620;
let drawHeight = 460;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 20;
let defaultTextSize = 14;

let scenarios = [
  {
    prompt: "You want to store a list of student names that you'll add to and remove from over time.",
    correct: 'List',
    explanation: 'List is ideal — ordered, mutable, and supports append/remove operations easily.'
  },
  {
    prompt: "You need to store each student's GPA by their student ID number for fast lookup.",
    correct: 'Dictionary',
    explanation: 'Dictionary maps keys (student ID) to values (GPA) for O(1) average-case lookup.'
  },
  {
    prompt: "You need to store GPS coordinates (latitude, longitude) that should never change.",
    correct: 'Tuple',
    explanation: 'Tuple is perfect for immutable fixed-size data like coordinates — it prevents accidental changes.'
  },
  {
    prompt: "You loaded a CSV with 10,000 rows of sales data that needs filtering and grouping.",
    correct: 'DataFrame',
    explanation: 'DataFrame (pandas) is built for large tabular data — it handles filtering, grouping, and aggregation efficiently.'
  },
  {
    prompt: "You need a single column of 1,000 temperature readings with a date index.",
    correct: 'Series',
    explanation: 'Series is a 1D labeled array — perfect for a single time-indexed data column.'
  },
  {
    prompt: "You want to store three subject scores (math, science, english) in a fixed record.",
    correct: 'Tuple',
    explanation: 'Tuple works well for a small, fixed record where order is meaningful and data won\'t change.'
  },
  {
    prompt: "You need to count how many times each word appears in a text document.",
    correct: 'Dictionary',
    explanation: 'Dictionary maps word → count. Python even has Counter(dict subclass) for exactly this task.'
  },
  {
    prompt: "You have a dataset with 50 columns and want to select rows where age > 30.",
    correct: 'DataFrame',
    explanation: 'DataFrame supports powerful boolean indexing: df[df["age"] > 30] is clean and fast.'
  },
  {
    prompt: "You're collecting items from a loop and want to process them all at the end.",
    correct: 'List',
    explanation: 'List.append() in a loop, then process — simple, readable, and memory-efficient for most cases.'
  },
  {
    prompt: "You have daily closing stock prices for one stock over a year and want to plot them.",
    correct: 'Series',
    explanation: 'Series with a DatetimeIndex is perfect for a single time-series — works seamlessly with matplotlib.'
  }
];

let choices = ['List', 'Dictionary', 'Tuple', 'Series', 'DataFrame'];
let choiceColors = [
  [70, 140, 220],
  [60, 160, 90],
  [200, 130, 40],
  [180, 80, 160],
  [160, 60, 60]
];

let currentQ = 0;
let score = 0;
let answered = false;
let selectedChoice = -1;
let showExplanation = false;
let gameOver = false;
let nextButton, restartButton;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  nextButton = createButton('Next Question →');
  nextButton.position(margin, drawHeight + 10);
  nextButton.mousePressed(nextQuestion);
  nextButton.hide();

  restartButton = createButton('Play Again');
  restartButton.position(canvasWidth / 2 - 50, drawHeight + 10);
  restartButton.mousePressed(restartGame);
  restartButton.hide();

  describe('Python data structures quiz. Read each scenario and click the correct data structure: List, Dictionary, Tuple, Series, or DataFrame. Get instant feedback and explanations.', LABEL);
}

function nextQuestion() {
  if (currentQ < scenarios.length - 1) {
    currentQ++;
    answered = false;
    selectedChoice = -1;
    showExplanation = false;
    nextButton.hide();
  } else {
    gameOver = true;
    nextButton.hide();
    restartButton.show();
  }
}

function restartGame() {
  currentQ = 0;
  score = 0;
  answered = false;
  selectedChoice = -1;
  showExplanation = false;
  gameOver = false;
  restartButton.hide();
}

function mousePressed() {
  if (answered || gameOver) return;

  // Check if clicked a choice button
  for (let i = 0; i < choices.length; i++) {
    let bx = choiceX(i);
    let by = choiceY(i);
    let bw = choiceW();
    let bh = 44;
    if (mouseX >= bx && mouseX <= bx + bw && mouseY >= by && mouseY <= by + bh) {
      selectedChoice = i;
      answered = true;
      if (choices[i] === scenarios[currentQ].correct) {
        score++;
      }
      showExplanation = true;
      nextButton.show();
      return;
    }
  }
}

function choiceW() {
  return min(160, (canvasWidth - margin * 2 - 16) / 5);
}

function choiceX(i) {
  let bw = choiceW();
  let totalW = bw * 5 + 16;
  let startX = (canvasWidth - totalW) / 2;
  return startX + i * (bw + 4);
}

function choiceY(i) {
  return drawHeight - 70;
}

function draw() {
  updateCanvasSize();
  background(248);

  fill(248);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill(255);
  rect(0, drawHeight, canvasWidth, controlHeight);

  if (gameOver) {
    drawGameOver();
    return;
  }

  // Progress bar
  let progressW = (canvasWidth - margin * 2) * (currentQ / scenarios.length);
  fill(70, 140, 220, 80);
  noStroke();
  rect(margin, margin, canvasWidth - margin * 2, 8, 4);
  fill(70, 140, 220);
  rect(margin, margin, progressW, 8, 4);
  fill(80);
  textAlign(RIGHT, TOP);
  textSize(11);
  text('Question ' + (currentQ + 1) + ' / ' + scenarios.length + '  |  Score: ' + score, canvasWidth - margin, margin - 1);

  // Scenario card
  let cardTop = margin + 20;
  let cardH = drawHeight - cardTop - 100;
  fill(255);
  stroke(180);
  strokeWeight(1);
  rect(margin, cardTop, canvasWidth - margin * 2, cardH, 8);

  noStroke();
  fill(50, 80, 160);
  textAlign(LEFT, TOP);
  textSize(11);
  text('SCENARIO', margin + 14, cardTop + 10);

  fill(30);
  textAlign(LEFT, TOP);
  textSize(15);
  textWrap(WORD);
  text(scenarios[currentQ].prompt, margin + 14, cardTop + 28, canvasWidth - margin * 2 - 28, cardH - 80);

  // Explanation
  if (showExplanation) {
    let isCorrect = choices[selectedChoice] === scenarios[currentQ].correct;
    fill(isCorrect ? color(30, 120, 60) : color(160, 40, 40));
    textSize(12);
    textAlign(LEFT, TOP);
    let resultMsg = isCorrect ? '✓ Correct! ' : '✗ Not quite. The answer is ' + scenarios[currentQ].correct + '. ';
    textWrap(WORD);
    text(resultMsg + scenarios[currentQ].explanation, margin + 14, cardTop + cardH - 58, canvasWidth - margin * 2 - 28, 56);
  } else {
    fill(120);
    textSize(12);
    textAlign(CENTER, BOTTOM);
    text('Which data structure is best for this scenario?', canvasWidth / 2, cardTop + cardH - 8);
  }

  // Choice buttons
  let bw = choiceW();
  let bh = 44;
  for (let i = 0; i < choices.length; i++) {
    let bx = choiceX(i);
    let by = choiceY(i);
    let c = choiceColors[i];
    let isSelected = (i === selectedChoice);
    let isCorrectChoice = answered && choices[i] === scenarios[currentQ].correct;
    let isWrong = answered && isSelected && !isCorrectChoice;

    if (isCorrectChoice && answered) {
      fill(30, 160, 80);
      stroke(10, 100, 40);
      strokeWeight(2.5);
    } else if (isWrong) {
      fill(200, 60, 60);
      stroke(140, 20, 20);
      strokeWeight(2.5);
    } else if (isSelected) {
      fill(c[0] + 30, c[1] + 30, c[2] + 30);
      stroke(255, 220, 50);
      strokeWeight(2);
    } else {
      fill(c[0], c[1], c[2]);
      stroke(c[0] - 30, c[1] - 30, c[2] - 30);
      strokeWeight(1);
    }

    rect(bx, by, bw, bh, 6);
    noStroke();
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(12);
    text(choices[i], bx + bw / 2, by + bh / 2);
  }

  // Score in control area
  noStroke();
  fill(80);
  textAlign(CENTER, CENTER);
  textSize(12);
  text(answered ? (nextButton ? '' : '') : 'Click the best data structure for this scenario', canvasWidth / 2, drawHeight + controlHeight / 2);
}

function drawGameOver() {
  noStroke();
  fill(30);
  textAlign(CENTER, CENTER);
  textSize(28);
  text('Quiz Complete!', canvasWidth / 2, drawHeight / 2 - 60);
  textSize(18);
  let pct = Math.round(score / scenarios.length * 100);
  text('Score: ' + score + ' / ' + scenarios.length + ' (' + pct + '%)', canvasWidth / 2, drawHeight / 2 - 20);

  textSize(14);
  fill(80);
  let msg = pct >= 80 ? 'Excellent! You have a strong grasp of Python data structures.' :
            pct >= 60 ? 'Good effort! Review the structures you missed and try again.' :
                        'Keep practicing — review Lists, Dicts, Tuples, Series, and DataFrames.';
  textWrap(WORD);
  text(msg, canvasWidth / 2 - 200, drawHeight / 2 + 20, 400, 80);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  nextButton.position(margin, drawHeight + 10);
  restartButton.position(canvasWidth / 2 - 50, drawHeight + 10);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
