// NumPy Ecosystem Map
// CANVAS_HEIGHT: 565
// Bloom L2 (Understand): a hub-and-spoke map with NumPy at the center. Students click a
// library, or step with Next and Previous, to read what it does, how it uses NumPy arrays,
// and the code that moves data between the two.
//
// Two kinds of link are drawn. "Built on NumPy" (solid): NumPy is a required dependency and
// the library computes with arrays inside (pandas, SciPy, scikit-learn, Matplotlib).
// "Works with NumPy" (dashed): the library has its own engine or tensor type and accepts
// arrays or converts to and from them (Plotly, PyTorch, TensorFlow). An arrowhead at the
// NumPy end means data also comes back as an array.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 520;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

const KINDS = {
  hub: 'The foundation',
  built: 'Built on NumPy',
  works: 'Works with NumPy'
};
// The first entry is the hub. back: data also returns to NumPy as an array.
const LIBS = [
  { name: 'NumPy', kind: 'hub', ink: 'royalblue',
    what: 'N-dimensional arrays (ndarray) and fast math on whole arrays at once. Click a library to see how it connects.',
    how: 'The array is the shared data format of scientific Python. The libraries around it either compute with ' +
      'arrays directly or convert their own data to and from them.',
    code: [['Make an array', 'arr = np.array([[1, 2], [3, 4]])'], ['Compute with it', 'arr.mean(axis=0)   # array([2., 3.])']],
    note: 'Solid line: built on NumPy. Dashed line: own engine, works with arrays. Two arrowheads: results come back as arrays.' },
  { name: 'pandas', kind: 'built', ink: 'slateblue', back: true,
    what: 'Labeled tables for data analysis: the DataFrame and the Series.',
    how: 'A DataFrame keeps its numeric columns in NumPy arrays by default, so column math is vectorized NumPy math.',
    code: [['Array → DataFrame', 'df = pd.DataFrame(arr, columns=["a", "b"])'], ['DataFrame → array', 'arr = df.to_numpy()']],
    note: 'to_numpy() is the recommended call. The older df.values gives the same array.' },
  { name: 'SciPy', kind: 'built', ink: 'steelblue', back: true,
    what: 'Scientific algorithms: optimization, integration, interpolation, statistics, sparse matrices.',
    how: 'SciPy functions take NumPy arrays as input and hand NumPy arrays back.',
    code: [['Arrays in', 'x = scipy.linalg.solve(A, b)'], ['Array out', 'type(x)   # numpy.ndarray']],
    note: 'NumPy has the basics, such as np.linalg. SciPy adds the larger toolbox on the same arrays.' },
  { name: 'scikit-learn', kind: 'built', ink: 'darkorange', back: true,
    what: 'Machine learning: regression, classification, clustering, model evaluation.',
    how: 'Estimators turn the data you pass (lists, DataFrames) into NumPy arrays, and predict() returns a NumPy array.',
    code: [['Arrays in', 'model = LinearRegression().fit(X, y)'], ['Array out', 'y_pred = model.predict(X_new)']],
    note: 'X must be 2-D, shape (samples, features). That is why you see X.reshape(-1, 1).' },
  { name: 'Matplotlib', kind: 'built', ink: 'seagreen',
    what: 'Static charts: line plots, scatter plots, histograms and many more.',
    how: 'Plotting functions convert their inputs to NumPy arrays before drawing.',
    code: [['Make the data', 'x = np.linspace(0, 10, 100)'], ['Arrays in, picture out', 'plt.plot(x, np.sin(x))']],
    note: 'Data moves one way here: arrays go in and a figure comes out.' },
  { name: 'Plotly', kind: 'works', ink: 'mediumpurple',
    what: 'Interactive charts that run in the browser.',
    how: 'Chart functions accept NumPy arrays, as well as lists and DataFrame columns, for x, y and other values.',
    code: [['Arrays in', 'fig = px.scatter(x=x, y=y)'], ['Interactive chart out', 'fig.show()']],
    note: 'The drawing is done by the plotly.js JavaScript library, so the array values are sent to it.' },
  { name: 'PyTorch', kind: 'works', ink: 'orangered', back: true,
    what: 'Deep learning: tensors that can run on a GPU and track gradients.',
    how: 'A tensor is PyTorch\'s own array type. It converts to and from a NumPy array in one call.',
    code: [['Array → tensor', 't = torch.from_numpy(arr)'], ['Tensor → array', 'arr = t.numpy()']],
    note: 'On the CPU both calls share memory: change the array and the tensor changes too.' },
  { name: 'TensorFlow', kind: 'works', ink: 'goldenrod', back: true,
    what: 'Deep learning: its own tensors, plus the Keras API for building models.',
    how: 'Tensors convert to and from NumPy arrays, and Keras model.fit(X, y) accepts arrays as training data.',
    code: [['Array → tensor', 't = tf.convert_to_tensor(arr)'], ['Tensor → array', 'arr = t.numpy()']],
    note: 'Both deep learning libraries keep NumPy\'s ideas: shape, dtype, slicing and broadcasting.' }
];

let selected = 0;
let prevButton, nextButton, kindSelect;
let nodeBoxes = [];               // clickable nodes: { x, y, w, h, index }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => step(-1));

  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 10);
  nextButton.mousePressed(() => step(1));

  kindSelect = createSelect();
  kindSelect.parent(mainElement);
  kindSelect.position(142, drawHeight + 10);
  kindSelect.option('Show all libraries', 'all');
  kindSelect.option('Only: built on NumPy', 'built');
  kindSelect.option('Only: works with NumPy', 'works');
  kindSelect.changed(() => { if (!visible(selected)) selected = 0; });

  describe('A hub and spoke map with NumPy in the center and seven libraries around it: pandas, SciPy, scikit-learn, ' +
    'Matplotlib, Plotly, PyTorch and TensorFlow. Solid lines mark libraries built on NumPy and dashed lines mark ' +
    'libraries that work with NumPy arrays. Selecting a library shows what it does, how it uses arrays, and two ' +
    'lines of code that move data between it and NumPy.', LABEL);
}

// The hub always shows. A library shows when its kind matches the menu.
function visible(i) { return i === 0 || kindSelect.value() === 'all' || LIBS[i].kind === kindSelect.value(); }
function step(direction) {
  do { selected = (selected + direction + LIBS.length) % LIBS.length; } while (!visible(selected));
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
  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('NumPy Ecosystem Map', canvasWidth / 2, 8);

  const fullW = canvasWidth - 2 * margin, top = narrow ? 34 : 44;
  nodeBoxes = [];
  if (narrow) {
    drawMap(margin, top, fullW, 228, narrow);
    drawDetail(margin, top + 234, fullW, drawHeight - top - 242, narrow);
  } else {
    const leftW = fullW * 0.54;
    drawMap(margin, top, leftW, drawHeight - top - 8, narrow);
    drawDetail(margin + leftW + 10, top, fullW - leftW - 10, drawHeight - top - 8, narrow);
  }
  cursor(nodeBoxes.some(mouseOver) ? HAND : ARROW);
}

// A filled arrowhead with its tip at (x, y), pointing along the angle
function arrowHead(x, y, angle, ink) {
  push();
  translate(x, y);
  rotate(angle);
  noStroke();
  fill(ink);
  triangle(0, 0, -10, -5, -10, 5);
  pop();
}

// The hub, the seven libraries on an ellipse around it, and the links between them
function drawMap(x, y, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 12 : 15, nw = narrow ? 88 : 116, nh = narrow ? 26 : 36, hubR = narrow ? 33 : 48;
  const legendH = narrow ? 20 : 30;
  const cx = x + w / 2, cy = y + (h - legendH) / 2;
  const rx = w / 2 - nw / 2 - 8, ry = (h - legendH) / 2 - nh / 2 - 8;
  const count = LIBS.length - 1;

  for (let i = 1; i < LIBS.length; i++) {
    const lib = LIBS[i], angle = -HALF_PI + (i - 1) * TWO_PI / count;
    const px = cx + rx * cos(angle), py = cy + ry * sin(angle);
    const shown = visible(i), on = i === selected;
    const ink = color(lib.ink);
    if (!shown) ink.setAlpha(50);
    // link: stops at the edge of the hub circle and at the edge of the node box
    const dx = px - cx, dy = py - cy, len = sqrt(dx * dx + dy * dy), ux = dx / len, uy = dy / len;
    const reach = len - min(abs(dx) > 0.01 ? (nw / 2) / abs(ux) : Infinity, abs(dy) > 0.01 ? (nh / 2) / abs(uy) : Infinity) - 3;
    stroke(ink);
    strokeWeight(on ? 4 : 2);
    if (lib.kind === 'works') drawingContext.setLineDash([7, 5]);
    line(cx + ux * (hubR + 12), cy + uy * (hubR + 12), cx + ux * (reach - 9), cy + uy * (reach - 9));
    drawingContext.setLineDash([]);
    arrowHead(cx + ux * reach, cy + uy * reach, atan2(uy, ux), ink);
    if (lib.back) arrowHead(cx + ux * (hubR + 3), cy + uy * (hubR + 3), atan2(-uy, -ux), ink);

    const box = { x: px - nw / 2, y: py - nh / 2, w: nw, h: nh, index: i };
    const bg = color(lib.ink);
    bg.setAlpha(!shown ? 12 : on ? 90 : mouseOver(box) ? 55 : 30);
    fill('white');
    noStroke();
    rect(box.x, box.y, nw, nh, 8);
    fill(bg);
    stroke(ink);
    strokeWeight(on ? 3 : 1.5);
    rect(box.x, box.y, nw, nh, 8);
    noStroke();
    fill(shown ? 'black' : 'silver');
    textStyle(on ? BOLD : NORMAL);
    textSize(ts);
    textAlign(CENTER, CENTER);
    text(lib.name, px, py + 1);
    if (shown) nodeBoxes.push(box);
  }

  // hub
  const hubBox = { x: cx - hubR, y: cy - hubR, w: 2 * hubR, h: 2 * hubR, index: 0 };
  fill(selected === 0 ? 'royalblue' : mouseOver(hubBox) ? 'dodgerblue' : 'cornflowerblue');
  stroke('midnightblue');
  strokeWeight(selected === 0 ? 4 : 2);
  circle(cx, cy, 2 * hubR);
  // array-grid icon
  const g = narrow ? 6 : 8;
  stroke('white');
  strokeWeight(1);
  noFill();
  for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) rect(cx - 1.5 * g + c * g, cy - hubR * 0.62 + r * g, g, g);
  noStroke();
  fill('white');
  textStyle(BOLD);
  textSize(ts + 2);
  textAlign(CENTER, CENTER);
  text('NumPy', cx, cy + (narrow ? 5 : 8));
  textStyle(NORMAL);
  textSize(narrow ? 11 : 12);
  text('ndarray', cx, cy + (narrow ? 19 : 26));
  nodeBoxes.push(hubBox);

  // legend
  const ly = y + h - legendH / 2 - 4, half = w / 2;
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, CENTER);
  [['built', false], ['works', true]].forEach(([kind, dashed], k) => {
    const lx = x + 12 + k * half;
    stroke('dimgray');
    strokeWeight(2);
    if (dashed) drawingContext.setLineDash([7, 5]);
    line(lx, ly, lx + 30, ly);
    drawingContext.setLineDash([]);
    noStroke();
    fill('dimgray');
    text(KINDS[kind], lx + 36, ly);
  });
}

// What the selected library is, how it uses NumPy, and the code that moves the data
function drawDetail(x, y, w, h, narrow) {
  const lib = LIBS[selected];
  fill('white');
  stroke(lib.ink);
  strokeWeight(2);
  rect(x, y, w, h, 10);
  const pad = narrow ? 10 : 14, iw = w - 2 * pad, ts = narrow ? 12 : 16, lh = ts + (narrow ? 3 : 5);
  let cy = y + (narrow ? 7 : 12);
  const say = (str, lines, ink, style) => {
    noStroke();
    fill(ink);
    textStyle(style || NORMAL);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, x + pad, cy, iw, lines * lh + 3);
    cy += lines * lh + (narrow ? 3 : 10);
    textStyle(NORMAL);
  };

  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(narrow ? 15 : 20);
  textAlign(LEFT, TOP);
  text(lib.name, x + pad, cy);
  textStyle(NORMAL);
  textSize(ts);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text(KINDS[lib.kind], x + w - pad, cy + (narrow ? 2 : 4));
  cy += narrow ? 22 : 32;

  say(lib.what, narrow ? 2 : 3, 'black');
  say(selected === 0 ? 'Why it is in the middle' : 'How it uses NumPy', 1, 'black', BOLD);
  cy -= narrow ? 2 : 6;
  say(lib.how, narrow ? 3 : 4, 'black');

  // two lines of code, each with a label saying which way the data moves
  const boxH = 2 * (2 * lh + 2) + 10;
  fill('whitesmoke');
  stroke('silver');
  strokeWeight(1);
  rect(x + pad - 4, cy, iw + 8, boxH, 6);
  noStroke();
  textAlign(LEFT, TOP);
  lib.code.forEach(([label, code], k) => {
    const ly = cy + 6 + k * (2 * lh + 2);
    fill('dimgray');
    textSize(narrow ? 11 : 13);
    text(label, x + pad + 2, ly);
    fill('black');
    textSize(narrow ? 12 : 15);
    text(code, x + pad + 2, ly + lh);
  });
  cy += boxH + (narrow ? 6 : 12);
  say(lib.note, 3, 'darkgreen');
}

// Clicking the hub or a library selects it.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = nodeBoxes.find(mouseOver);
  if (box) selected = box.index;
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
