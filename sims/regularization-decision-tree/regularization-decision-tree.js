// Regularization Decision Tree
// CANVAS_HEIGHT: 625
// Bloom L5 and L3 (Evaluate, Apply): students read the facts of a modeling problem, walk the
// decision tree, and click the regularization method they judge to be the right one. Feedback
// explains the choice. Clicking a question shows what it asks and why it matters.
//
// Model: the path for each case is computed from its facts, not stored.
//   overfitting     training R² − validation R² is greater than 0.05
//   strength of λ   p / n above 0.5 is strong, 0.05 to 0.5 is moderate, below 0.05 is light
//                   (a rule of thumb for where to start; cross-validation chooses the actual λ)
//   method          no overfitting: no penalty. Keep every feature: Ridge.
//                   Drop features, no correlated groups: Lasso. Drop features, correlated groups: Elastic Net.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 580;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// fill and outline for each kind of box
const KINDS = { ask: ['lightblue', 'royalblue'], step: ['khaki', 'goldenrod'], none: ['gainsboro', 'gray'],
  ridge: ['lightsteelblue', 'steelblue'], lasso: ['navajowhite', 'darkorange'], enet: ['plum', 'purple'] };
// The questions run down the left side. side: the box reached by the sideways answer.
const SPINE = [
  { text: 'Is the relationship a straight line?', side: 'poly', sideLabel: 'No', down: 'Yes, or once added',
    about: 'A penalty cannot repair the wrong shape. If a scatter plot or a residual plot bends, add polynomial features or transform a variable first, then come back to the tree.' },
  { text: 'Does the model overfit? Is training R² well above validation R²?', side: 'none', sideLabel: 'No', down: 'Yes',
    about: 'Overfitting shows as a gap between the two scores. This tree calls a gap above 0.05 overfitting. With no gap there is nothing for a penalty to fix, and it would only pull good coefficients toward zero.' },
  { text: 'How many features p for n samples?', side: 'strength', sideLabel: '', down: '',
    about: 'More features per sample give the model more ways to fit noise, so the penalty must be stronger. Rule of thumb used here: p / n above 0.5 strong, 0.05 to 0.5 moderate, below 0.05 light. Cross-validation sets the actual λ.' },
  { text: 'Should the model drop some features?', side: 'ridge', sideLabel: 'No', down: 'Yes',
    about: 'Lasso and Elastic Net can set coefficients to exactly 0. That removes features and leaves a shorter model that is easier to explain. Ridge keeps every feature and often predicts a little better when all of them matter.' },
  { text: 'Are some features highly correlated?', side: 'lasso', sideLabel: 'No', down: 'Yes',
    about: 'Among highly correlated features, Lasso tends to keep one and drop the rest, and its pick can change when the data change. Elastic Net adds an L2 term that keeps correlated features together.' }
];
// The four results a student can choose. wrong: shown when it is picked for a case it does not fit.
const RESULTS = {
  none: { top: 'No penalty needed', label: 'LinearRegression()', wrong: 'Without a penalty, nothing closes the gap between the training and validation scores.' },
  ridge: { top: 'Keep every feature', label: 'Ridge(alpha=λ)', wrong: 'Ridge shrinks coefficients but never sets one to 0, so every feature stays in the model.' },
  lasso: { top: 'A few key features', label: 'Lasso(alpha=λ)', wrong: 'Lasso sets some coefficients to exactly 0 and, among correlated features, keeps one and drops the rest.' },
  enet: { top: 'Groups of features', label: 'ElasticNet(alpha=λ)', wrong: 'Elastic Net removes features while keeping correlated ones together, and it has a second setting (l1_ratio) to tune.' }
};

// Facts of each problem. select: the goal calls for dropping features. grouped: some features are highly correlated.
const CASES = [
  { name: 'A. Exam scores', n: 400, p: 3, train: 0.72, val: 0.71, curved: false, select: false, grouped: false,
    story: 'Predict an exam score from 3 study habits. Each habit plotted against the score looks like a straight line.',
    why: 'Training and validation R² are almost equal, so the model is not overfitting. A penalty would only pull good coefficients toward zero.',
    hint: 'Compare the training and validation scores first.' },
  { name: 'B. House prices', n: 1200, p: 30, train: 0.91, val: 0.84, curved: false, select: false, grouped: true,
    story: 'Predict sale price from 30 features of a house with a straight-line model. Size, rooms, and lot area rise together. Every feature is thought to matter, and the goal is the most accurate price.',
    why: 'The gap shows overfitting, every feature is wanted, and Ridge handles correlated features well by sharing the weight among them.',
    hint: 'Does anyone want features removed here?' },
  { name: 'C. Customer survey', n: 250, p: 60, train: 0.88, val: 0.61, curved: false, select: true, grouped: false,
    story: 'Explain satisfaction from 60 unrelated survey questions with straight-line effects. Managers want a short list of the questions that matter.',
    why: 'The gap is large and the goal is a short list. The questions are not strongly correlated, so Lasso can safely set the unhelpful coefficients to exactly 0.',
    hint: 'What does the goal say about the number of features, and are the questions correlated?' },
  { name: 'D. Gene activity', n: 80, p: 2000, train: 1.00, val: 0.35, curved: false, select: true, grouped: true,
    story: 'Predict drug response from the activity of 2,000 genes. Genes in the same pathway are highly correlated, and the lab wants to know which genes to study.',
    why: 'With far more genes than patients the model memorizes the training data. The lab wants a subset, but Lasso alone would keep one gene per pathway almost at random. Elastic Net keeps a pathway together.',
    hint: 'The lab wants a subset of genes, and the genes come in correlated groups.' },
  { name: 'E. Sensor calibration', n: 40, p: 10, train: 0.97, val: 0.55, curved: true, select: false, grouped: true,
    story: 'Calibrate a sensor from one input. The scatter plot bends, so powers of the input up to degree 10 were added. The numbers below are for that model. All ten powers are kept, and they are highly correlated.',
    why: 'After polynomial features are added the model overfits. Nothing needs to be dropped, and powers of one input are highly correlated, which is where Ridge works well.',
    hint: 'All ten powers are kept, and the goal is a smooth prediction.' },
  { name: 'F. Wine chemistry', n: 150, p: 12, train: 0.58, val: 0.41, curved: false, select: true, grouped: true,
    story: 'Explain a wine quality score from 12 lab measurements with straight-line effects. Fixed acidity, citric acid, and pH move together. The winemaker wants to know which measurements to watch.',
    why: 'The model overfits and the winemaker wants a subset. The acidity measurements are highly correlated, so Elastic Net is the safer way to drop features.',
    hint: 'The winemaker wants a subset, and the three acidity measurements move together.' }
];

let caseSelect, pathCheckbox;
let caseIndex = 0;
let pick = null;                  // the result the student clicked
let info = null;                  // index of the question whose explanation is open
let resultBoxes = [], questionBoxes = [];

// Walk the tree for one case. Returns the result, how many questions lie on the path, and the λ strength.
function decide(cs) {
  const gap = cs.train - cs.val, ratio = cs.p / cs.n;
  const strength = ratio > 0.5 ? 'strong' : ratio >= 0.05 ? 'moderate' : 'light';
  const best = gap <= 0.05 ? 'none' : !cs.select ? 'ridge' : cs.grouped ? 'enet' : 'lasso';
  return { gap, ratio, strength, best, depth: best === 'none' ? 1 : best === 'ridge' ? 3 : 4 };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  caseSelect = createSelect();
  caseSelect.parent(mainElement);
  caseSelect.position(10, drawHeight + 10);
  CASES.forEach(cs => caseSelect.option(cs.name));
  caseSelect.style('font-size', '15px');
  // a new case is a new problem: the answer is hidden again
  caseSelect.changed(() => {
    caseIndex = CASES.findIndex(cs => cs.name === caseSelect.value());
    pick = null;
    info = null;
    pathCheckbox.checked(false);
  });

  // case A opens with its path shown, as a worked example
  pathCheckbox = createCheckbox(' Show the path', true);
  pathCheckbox.parent(mainElement);
  pathCheckbox.position(195, drawHeight + 11);
  pathCheckbox.style('font-size', '16px');

  describe('A decision tree for choosing a regularization method. Five questions about the shape of the relationship, ' +
    'overfitting, the number of features, feature selection, and correlated features lead to four results: no penalty, ' +
    'ridge, lasso, and elastic net. A menu chooses one of six example problems. Clicking a result gives feedback, ' +
    'clicking a question explains it, and a checkbox highlights the path that fits the problem.', LABEL);
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600;
  const cs = CASES[caseIndex], d = decide(cs);
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Regularization Decision Tree', canvasWidth / 2, 8);

  // tree on the left and the case on the right, or stacked when narrow
  const top = narrow ? 32 : 44, fullW = canvasWidth - 2 * margin, bottom = drawHeight - 8;
  if (narrow) {
    const caseH = 236;
    drawTree(margin, top, fullW, bottom - top - caseH - 6, narrow, cs, d);
    drawCase(margin, bottom - caseH, fullW, caseH, narrow, cs, d);
  } else {
    const treeW = fullW * 0.56;
    drawTree(margin, top, treeW, bottom - top, narrow, cs, d);
    drawCase(margin + treeW + 12, top, fullW - treeW - 12, bottom - top, narrow, cs, d);
  }
  cursor(resultBoxes.concat(questionBoxes).some(mouseOver) ? HAND : ARROW);
}

// A line with an arrowhead at its end and an optional answer label. Lines are vertical or horizontal.
function connector(x1, y1, x2, y2, on, label, ts) {
  const vertical = x1 === x2, a = 5;
  stroke(on ? 'darkgreen' : 'gray');
  strokeWeight(on ? 3.5 : 1.5);
  line(x1, y1, x2, y2);
  noStroke();
  fill(on ? 'darkgreen' : 'gray');
  if (vertical) triangle(x2, y2, x2 - a, y2 - 8, x2 + a, y2 - 8);
  else triangle(x2, y2, x2 - 8, y2 - a, x2 - 8, y2 + a);
  if (label) {
    fill('black');
    textStyle(BOLD);
    textSize(ts);
    textAlign(vertical ? LEFT : CENTER, vertical ? CENTER : BOTTOM);
    text(label, vertical ? x1 + 8 : (x1 + x2) / 2 - 4, vertical ? (y1 + y2) / 2 - 3 : y1 - 5);
  }
}

// A box with one wrapped line of text, or a plain top line over a bold bottom line
function drawBox(x, y, w, h, kind, emphasis, top, label, ts) {
  fill(KINDS[kind][0]);
  stroke(emphasis || KINDS[kind][1]);
  strokeWeight(emphasis ? 3.5 : 1.3);
  rect(x, y, w, h, 8);
  noStroke();
  fill('black');
  textAlign(CENTER, CENTER);
  textSize(ts);
  textLeading(ts + 2);
  textStyle(NORMAL);
  if (label === undefined) {
    text(top, x + 5, y, w - 10, h);
    return;
  }
  text(top, x + w / 2, y + h * 0.3);
  textStyle(BOLD);
  text(label, x + w / 2, y + h * 0.7);
}

function drawTree(x, y, w, h, narrow, cs, d) {
  const nodeH = narrow ? 33 : 52, noteH = narrow ? 16 : 22, rows = SPINE.length + 1;
  const gap = (h - noteH - rows * nodeH) / (rows - 1);
  const spineW = w * 0.52, sideW = w * 0.37, sideX = x + w - sideW, mid = x + spineW / 2;
  const ts = narrow ? 11 : 14, showPath = pathCheckbox.checked();
  const depth = showPath ? d.depth : -1;                     // questions 0..depth lie on the path
  const nodeY = i => y + i * (nodeH + gap);
  const ring = id => pick === id ? 'black' : showPath && d.best === id ? 'darkgreen' : null;
  resultBoxes = [];
  questionBoxes = [];

  SPINE.forEach((node, i) => {
    const cy = nodeY(i) + nodeH / 2, sideOn = showPath && (node.side === d.best || node.side === 'poly' && cs.curved || node.side === 'strength' && depth >= 2);
    connector(mid, nodeY(i) + nodeH, mid, nodeY(i + 1), i < depth || showPath && i === 4 && d.best === 'enet', node.down, ts);
    connector(x + spineW, cy, sideX, cy, sideOn, node.sideLabel, ts);
    drawBox(x, nodeY(i), spineW, nodeH, 'ask', info === i ? 'black' : i <= depth ? 'darkgreen' : null, node.text, undefined, ts);
    questionBoxes.push({ x, y: nodeY(i), w: spineW, h: nodeH, id: i });
    if (node.side === 'poly') {
      drawBox(sideX, nodeY(i), sideW, nodeH, 'step', sideOn ? 'darkgreen' : null, 'Add polynomial features first', undefined, ts);
    } else if (node.side === 'strength') {
      drawBox(sideX, nodeY(i), sideW, nodeH, 'step', sideOn ? 'darkgreen' : null, 'p / n = ' + Number(d.ratio.toPrecision(2)), d.strength + ' λ', ts);
    } else {
      drawBox(sideX, nodeY(i), sideW, nodeH, node.side, ring(node.side), RESULTS[node.side].top, RESULTS[node.side].label, narrow ? 11 : 13);
      resultBoxes.push({ x: sideX, y: nodeY(i), w: sideW, h: nodeH, id: node.side });
    }
  });
  // the last Yes leads to Elastic Net at the foot of the tree
  drawBox(x, nodeY(rows - 1), spineW, nodeH, 'enet', ring('enet'), RESULTS.enet.top, RESULTS.enet.label, narrow ? 11 : 13);
  resultBoxes.push({ x, y: nodeY(rows - 1), w: spineW, h: nodeH, id: 'enet' });

  noStroke();
  fill('dimgray');
  textStyle(NORMAL);
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, BOTTOM);
  text(narrow ? 'Click a question to explain it, or a result to choose it.' : 'Click a blue question to explain it. Click a result to choose it.', x, y + h + 2);
}

// The case: its story, its numbers, and either feedback on the choice or the explanation of a question
function drawCase(x, y, w, h, narrow, cs, d) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 11 : 14, lh = ts + (narrow ? 3 : 5), iw = w - 2 * pad, ix = x + pad;
  const label = (str, ly, lines, col, style) => {
    noStroke();
    fill(col);
    textStyle(style);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, ix, ly, iw, lines * lh + 3);
    return ly + lines * lh;
  };
  let ty = label(cs.name, y + pad, 1, 'black', BOLD) + 2;
  textStyle(NORMAL);                  // room for the story: an estimate of its wrapped lines
  const storyLines = Math.ceil(textWidth(cs.story) / (iw * 0.88));
  ty = label(cs.story, ty, storyLines, 'black', NORMAL) + (narrow ? 2 : 8);
  // the numbers, with the two quantities the tree asks about worked out
  ty = label('Samples n = ' + cs.n.toLocaleString('en-US') + ',  features p = ' + cs.p.toLocaleString('en-US') + (narrow ? ',  ' : '\n') +
    'p / n = ' + Number(d.ratio.toPrecision(2)), ty, narrow ? 1 : 2, 'midnightblue', NORMAL);
  ty = label('Training R² = ' + cs.train.toFixed(2) + ',  validation R² = ' + cs.val.toFixed(2) + (narrow ? ',  ' : '\n') +
    'gap = ' + d.gap.toFixed(2), ty, narrow ? 1 : 2, 'midnightblue', NORMAL) + (narrow ? 4 : 12);
  stroke('gainsboro');
  line(ix, ty - (narrow ? 2 : 6), ix + iw, ty - (narrow ? 2 : 6));

  const room = Math.floor((y + h - ty - 4) / lh);
  if (info !== null) {
    ty = label('About question ' + (info + 1), ty, 1, 'royalblue', BOLD);
    label(SPINE[info].about, ty, room - 1, 'black', NORMAL);
    return;
  }
  const shown = pick || (pathCheckbox.checked() ? d.best : null);
  if (!shown) {
    label('Walk the tree with the facts above, one question at a time, then click the result you would choose.', ty, room, 'dimgray', ITALIC);
    return;
  }
  const result = RESULTS[shown], lambdaNote = d.best === 'none' ? '' : ' Start with a ' + d.strength + ' λ and tune it by cross-validation.';
  ty = label((pick ? 'Your choice: ' : 'The path leads to: ') + result.label, ty, 1, 'black', BOLD);
  if (!pick) label('Why this path: ' + cs.why + lambdaNote, ty, room - 1, 'darkgreen', NORMAL);
  else if (pick === d.best) label('Good choice. ' + cs.why + lambdaNote, ty, room - 1, 'darkgreen', NORMAL);
  else label('Not the best fit for this problem. ' + result.wrong + ' ' + cs.hint, ty, room - 1, 'firebrick', NORMAL);
}

// Clicking a result chooses it and clicking a question explains it. A second click clears either one.
const mouseOver = b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
function mousePressed() {
  const result = resultBoxes.find(mouseOver), question = questionBoxes.find(mouseOver);
  if (result) { pick = pick === result.id ? null : result.id; info = null; }
  if (question) info = info === question.id ? null : question.id;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
