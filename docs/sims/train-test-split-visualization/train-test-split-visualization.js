// Train-Test Split Visualization
// CANVAS_HEIGHT: 580
// Bloom L2 (Understand): students move a slider to change the split ratio and see which of 100
// rows become training data and which are held back for testing. Nothing is animated: the picture
// changes only when the slider or the New Shuffle button is used.
//
// Model: 100 synthetic houses, price = 50,000 + 150 * sqft + noise, from a seeded generator. The
// rows are shuffled with a seeded Fisher-Yates shuffle (the seed plays the role of random_state)
// and the shuffled list is cut once: the first n_train rows are the training set, the rest the
// test set. A least-squares line is fit on the training rows only. R^2 = 1 - SS_res / SS_tot is
// then computed separately on the training rows and on the test rows (the r2_score definition).

let containerWidth;
let canvasWidth = 400;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 180;
let defaultTextSize = 16;

const N = 100, DATA_SEED = 2040, GAP = 12;   // GAP: width of the wall between the two sets
const TRAIN_COLOR = 'seagreen', TEST_COLOR = 'royalblue';
let rngState = 1;
let houses = [];            // { sqft, price } for each row, in file order
let order = [];             // shuffled row numbers: the first nTrain are the training set
let where = [];             // where[row] = position of that row in the shuffled order
let shuffleSeed = 42;
let splitSlider, shuffleButton;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }

function makeData() {
  rngState = DATA_SEED;
  houses = [];
  for (let i = 0; i < N; i++) {
    const sqft = Math.round((800 + 2200 * uniform()) / 10) * 10;
    const price = Math.round((50000 + 150 * sqft + 60000 * stdNormal()) / 1000) * 1000;
    houses.push({ sqft, price });
  }
}

function shuffleRows() {
  rngState = shuffleSeed;
  order = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  order.forEach((row, k) => { where[row] = k; });
}

// Least-squares line through the given rows: returns [intercept, slope]
function fitLine(rows) {
  const n = rows.length;
  const mx = rows.reduce((s, r) => s + houses[r].sqft, 0) / n;
  const my = rows.reduce((s, r) => s + houses[r].price, 0) / n;
  let sxy = 0, sxx = 0;
  for (const r of rows) {
    sxy += (houses[r].sqft - mx) * (houses[r].price - my);
    sxx += (houses[r].sqft - mx) ** 2;
  }
  return [my - sxy / sxx * mx, sxy / sxx];
}

// R^2 of the line price = b0 + b1 * sqft, measured on the given rows
function rSquared(rows, b0, b1) {
  const my = rows.reduce((s, r) => s + houses[r].price, 0) / rows.length;
  let ssRes = 0, ssTot = 0;
  for (const r of rows) {
    ssRes += (houses[r].price - (b0 + b1 * houses[r].sqft)) ** 2;
    ssTot += (houses[r].price - my) ** 2;
  }
  return 1 - ssRes / ssTot;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  splitSlider = createSlider(50, 95, 80, 5);
  splitSlider.parent(mainElement);
  splitSlider.position(sliderLeftMargin, drawHeight + 8);
  splitSlider.size(canvasWidth - sliderLeftMargin - margin);

  shuffleButton = createButton('New Shuffle');
  shuffleButton.parent(mainElement);
  shuffleButton.position(10, drawHeight + 45);
  shuffleButton.mousePressed(() => { shuffleSeed++; shuffleRows(); });

  makeData();
  shuffleRows();

  describe('One hundred squares stand for the rows of a housing data set. A slider sets the share used for training. ' +
    'The rows are shuffled and cut once into a green training set and a blue test set with a wall between them. ' +
    'Panels show R squared for a line fit on the training rows, measured on each set, and warn when the split is too extreme.', LABEL);
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
  const ts = narrow ? 12 : 15;
  const nTrain = splitSlider.value(), nTest = N - nTrain;
  const trainRows = order.slice(0, nTrain), testRows = order.slice(nTrain);
  const [b0, b1] = fitLine(trainRows);
  const r2Train = rSquared(trainRows, b0, b1), r2Test = rSquared(testRows, b0, b1);

  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Train-Test Split Visualization', canvasWidth / 2, 8);
  fill('dimgray');
  textSize(narrow ? 12 : 15);
  text('train_test_split(X, y, test_size=' + nf(nTest / N, 1, 2) + ', random_state=' + shuffleSeed + ')', canvasWidth / 2, 38);

  // geometry of the file-order strip (2 x 50) and of the shuffled grid (5 rows, filled column by column)
  const sc = min(15, (canvasWidth - 2 * margin) / 50);
  const sx = (canvasWidth - 50 * sc) / 2, sy = 84;
  const cell = min(30, (canvasWidth - 2 * margin - GAP) / 20);
  const gx = (canvasWidth - 20 * cell - GAP) / 2, gy = sy + 2 * sc + 62;
  const gridRight = gx + 20 * cell + GAP, gridBottom = gy + 5 * cell;
  const wallX = gx + nTrain / 5 * cell;
  const cellX = k => gx + floor(k / 5) * cell + (k >= nTrain ? GAP : 0);

  // which row is the mouse over, in either picture?
  let hover = -1;
  if (mouseX >= sx && mouseX < sx + 50 * sc && mouseY >= sy && mouseY < sy + 2 * sc) {
    hover = floor((mouseY - sy) / sc) * 50 + floor((mouseX - sx) / sc);
  } else if (mouseY >= gy && mouseY < gridBottom) {
    for (let k = 0; k < N; k++) {
      if (mouseX >= cellX(k) && mouseX < cellX(k) + cell && floor((mouseY - gy) / cell) === k % 5) hover = order[k];
    }
  }

  // 1. the full data set in file order, colored by where each row will go
  noStroke();
  fill('black');
  textAlign(CENTER, BOTTOM);
  textSize(ts);
  text(narrow ? 'Full data set: 100 houses in file order' :
    'Full data set: 100 houses in file order. The color shows where the shuffle sends each row.', canvasWidth / 2, sy - 4);
  for (let row = 0; row < N; row++) {
    fill(where[row] < nTrain ? TRAIN_COLOR : TEST_COLOR);
    stroke('white');
    strokeWeight(1);
    rect(sx + (row % 50) * sc, sy + floor(row / 50) * sc, sc, sc);
  }
  if (hover >= 0) {                      // outline the hovered row last so its neighbors do not cover it
    noFill();
    stroke('black');
    strokeWeight(2);
    rect(sx + (hover % 50) * sc, sy + floor(hover / 50) * sc, sc, sc);
  }

  // 2. the step between the two pictures, or the details of the row under the mouse
  noStroke();
  fill(hover >= 0 ? 'black' : 'dimgray');
  textAlign(CENTER, TOP);
  textSize(ts);
  text(hover >= 0 ? 'Row ' + hover + ': ' + houses[hover].sqft.toLocaleString('en-US') + ' sq ft, $' +
    houses[hover].price.toLocaleString('en-US') + ' → ' + (where[hover] < nTrain ? 'training set' : 'test set') :
    '▼  shuffle the rows, then cut once  ▼', canvasWidth / 2, sy + 2 * sc + 8);

  // 3. the two sets with the wall between them
  textStyle(BOLD);
  textSize(narrow ? 13 : 16);
  textAlign(LEFT, CENTER);
  fill('darkgreen');
  text('Training: ' + nTrain + ' samples', gx + 28, gy - 14);
  textAlign(RIGHT, CENTER);
  fill('mediumblue');
  text('Testing: ' + nTest + ' samples', gridRight - 30, gy - 14);
  textStyle(NORMAL);
  drawEye(gx + 11, gy - 14, 20, 'darkgreen', false);
  drawEye(gridRight - 11, gy - 14, 20, 'mediumblue', true);

  textAlign(CENTER, CENTER);
  textSize(11);
  for (let k = 0; k < N; k++) {
    const x = cellX(k), y = gy + (k % 5) * cell;
    fill(k < nTrain ? TRAIN_COLOR : TEST_COLOR);
    stroke('white');
    strokeWeight(1);
    rect(x, y, cell, cell, 3);
    noStroke();
    fill('white');
    text(order[k], x + cell / 2, y + cell / 2 + 1);
  }
  if (hover >= 0) {
    noFill();
    stroke('black');
    strokeWeight(2.5);
    rect(cellX(where[hover]), gy + (where[hover] % 5) * cell, cell, cell, 3);
  }
  // wall
  fill('saddlebrown');
  stroke('black');
  strokeWeight(1);
  rect(wallX + 2, gy - 6, GAP - 4, 5 * cell + 12, 2);
  stroke('peru');
  for (let y = gy + 4; y < gridBottom; y += 12) line(wallX + 3, y, wallX + GAP - 3, y);

  // brackets and captions under the two sets
  strokeWeight(3);
  stroke(TRAIN_COLOR);
  line(gx, gridBottom + 7, wallX - 2, gridBottom + 7);
  stroke(TEST_COLOR);
  line(wallX + GAP + 2, gridBottom + 7, gridRight, gridBottom + 7);
  noStroke();
  textSize(ts);
  textAlign(LEFT, TOP);
  fill('darkgreen');
  text('The model studies these rows\nmodel.fit(X_train, y_train)', gx, gridBottom + 13);
  textAlign(RIGHT, TOP);
  fill('mediumblue');
  text('Hidden until the final check\nmodel.score(X_test, y_test)', gridRight, gridBottom + 13);

  // 4. scores and advice
  const py = gridBottom + 2 * (ts + 5) + 20;
  const lh = ts + 6;
  const pw = narrow ? canvasWidth - 2 * margin : (canvasWidth - 2 * margin - 10) / 2;
  const scoresBottom = drawPanel(margin, py, pw, 'A line fit on the training rows only', 'white', ts, lh, [
    ['R² on the ' + nTrain + ' training rows (already seen): ' + nf(r2Train, 1, 2), 'darkgreen'],
    ['R² on the ' + nTest + ' test rows (never seen): ' + nf(r2Test, 1, 2), 'mediumblue'],
    ['Only the test R² is a fair estimate for new houses.', 'dimgray']
  ]);
  const ax = narrow ? margin : margin + pw + 10, ay = narrow ? scoresBottom + 7 : py;
  drawAdvice(ax, ay, pw, narrow ? drawHeight - 8 - ay : scoresBottom - py, nTrain, nTest, ts);

  // control labels
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Training share: ' + nTrain + '%', 10, drawHeight + 18);
  text('Shuffle seed: ' + shuffleSeed, 115, drawHeight + 57);
}

// The advice panel: is this split reasonable?
function drawAdvice(x, y, w, h, nTrain, nTest, ts) {
  let title = 'A typical split', bg = 'honeydew', titleColor = 'darkgreen';
  let msg = 'Using 70 to 80% of the rows for training is the usual choice. The model has plenty to learn from, ' +
    'and enough rows are held back to test it.';
  if (nTrain < 60) {
    title = 'Warning: too little training data'; bg = 'moccasin'; titleColor = 'chocolate';
    msg = nTest + ' of the 100 rows are locked away, so the model learns from only ' + nTrain + '. ' +
      'Rows used for testing cannot also teach the model.';
  } else if (nTrain > 90) {
    title = 'Warning: too little test data'; bg = 'mistyrose'; titleColor = 'firebrick';
    msg = 'A score from only ' + nTest + ' test rows depends on which rows were drawn. ' +
      'Press New Shuffle a few times and watch the test R² jump.';
  } else if (nTrain < 70) {
    title = 'Usable, with less to learn from'; bg = 'lightyellow'; titleColor = 'black';
    msg = 'The test score rests on ' + nTest + ' rows, but the model learns from only ' + nTrain + '. Most projects train on 70 to 80%.';
  } else if (nTrain > 80) {
    title = 'Usable, with a small test set'; bg = 'lightyellow'; titleColor = 'black';
    msg = 'The model learns from ' + nTrain + ' rows, but only ' + nTest + ' are left to test it. ' +
      'Press New Shuffle to see how much the test R² moves.';
  }
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill(titleColor);
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  fill('black');
  textSize(ts);
  textWrap(WORD);
  text(msg, x + 10, y + ts + 15, w - 20, h - ts - 18);
}

// An eye (the model can see these rows) or a crossed-out eye (these rows are hidden)
function drawEye(cx, cy, s, col, hidden) {
  stroke(col);
  strokeWeight(1.5);
  fill('white');
  ellipse(cx, cy, s, s * 0.6);
  noStroke();
  fill(col);
  circle(cx, cy, s * 0.34);
  if (hidden) {
    stroke('crimson');
    strokeWeight(2.5);
    line(cx - s * 0.5, cy + s * 0.42, cx + s * 0.5, cy - s * 0.42);
  }
}

// A titled panel holding lines of text. Returns the y coordinate of its bottom edge.
function drawPanel(x, y, w, title, bg, ts, lh, lines) {
  const h = 14 + lh * (lines.length + 1);
  fill(bg);
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text(title, x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
  for (let i = 0; i < lines.length; i++) {
    fill(lines[i][1]);
    text(lines[i][0], x + 10, y + 9 + lh * (i + 1));
  }
  return y + h;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  splitSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
