// Transformation Gallery
// CANVAS_HEIGHT: 640
// Bloom L2-L3 (Understand, Apply): students compare six common transformations on synthetic data,
// click one to study its before and after plots, tune the Box-Cox λ, and add zeros and negative
// values to see where each transformation is undefined.
//
// Data (seeded generator, 200 values each): lognormal incomes (log), Poisson counts at five shops
// with means 5, 10, 20, 35, 50 (square root), driving time = 1 / (speed / 120 + noise) (reciprocal),
// pendulum period 2π√(L / 9.81) + noise (square), gamma(shape 2) waiting times (Box-Cox), and
// income in dollars with age in years (standardize).
// Statistics: skewness = mean((y − mean)³) / SD³ and z = (x − μ) / σ, both with the population SD
// (ddof = 0, as StandardScaler uses). R² of a straight line is the squared correlation.
// Box-Cox: y(λ) = (y^λ − 1) / λ, and ln y at λ = 0. The maximum-likelihood λ maximizes
// −(n / 2) ln var(y(λ)) + (λ − 1) Σ ln y, the function scipy.stats.boxcox maximizes.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 560;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

const N = 200, N_ADDED = 12, SHOP_MEANS = [5, 10, 20, 35, 50];
const CLASS_COLORS = ['royalblue', 'darkorange', 'crimson'];   // original, added and defined, undefined
// f: the transformation of one value. ok: where it is defined. kind: how the data are plotted.
const TRANSFORMS = [
  { id: 'log', name: 'Log', formula: 'y′ = ln(y)', kind: 'hist', f: v => Math.log(v), ok: v => v > 0,
    before: 'income ($1000s)', after: 'ln(income)',
    use: 'right-skewed values that span orders of magnitude, exponential growth, and multiplicative effects.',
    data: '200 household incomes (lognormal).',
    domain: 'y > 0 only. np.log(0) is -inf and the log of a negative number is nan. When the data hold zeros, np.log1p(y) is a common substitute.' },
  { id: 'sqrt', name: 'Square root', formula: 'y′ = √y', kind: 'scatter', f: v => Math.sqrt(v), ok: v => v >= 0,
    before: 'customers per day', after: '√(customers per day)', xl: 'shop, smallest to largest',
    use: 'counts, where the spread grows with the mean (Poisson-like data).',
    data: 'daily customer counts at 5 shops, 40 days each (Poisson, means 5 to 50).',
    domain: 'y ≥ 0. Zero is fine because √0 = 0, but np.sqrt of a negative number is nan.' },
  { id: 'recip', name: 'Reciprocal', formula: 'y′ = 1 / y', kind: 'scatter', line: true, f: v => 1 / v, ok: v => v !== 0,
    before: 'hours to drive 120 km', after: '1 / hours', xl: 'speed (km/h)',
    use: 'inverse relationships, where y falls quickly and then levels off as x grows.',
    data: 'driving time for a 120 km trip against speed. 1 / time is a rate, and the rate is proportional to speed.',
    domain: 'every y except 0 (NumPy gives inf for 1 / 0). Negative values are allowed but land on a separate branch.' },
  { id: 'square', name: 'Square', formula: 'y′ = y²', kind: 'scatter', line: true, f: v => v * v, ok: () => true,
    before: 'period T (s)', after: 'T² (s²)', xl: 'pendulum length L (m)',
    use: 'a curve that rises quickly and then flattens, like a square root. To straighten a curve that keeps getting steeper, square the feature x instead.',
    data: 'period of a pendulum against its length, T = 2π√(L / g) plus timing error.',
    domain: 'every number. But −3 and 3 both become 9, so the order is scrambled when y has both signs.' },
  { id: 'boxcox', name: 'Box-Cox', formula: 'y′ = (y^λ − 1) / λ', kind: 'hist', f: null, ok: v => v > 0,
    before: 'waiting time (min)', after: 'Box-Cox value',
    use: 'skewed positive data when you want the power that makes it most nearly normal. λ = 1 keeps the shape, 0.5 acts like √y, 0 is ln(y), and −1 acts like 1 / y.',
    data: '200 waiting times (gamma, shape 2).',
    domain: 'y > 0 only (scipy.stats.boxcox raises an error otherwise). Yeo-Johnson, the default of PowerTransformer, also accepts zeros and negatives.' },
  { id: 'std', name: 'Standardize', formula: 'z = (x − μ) / σ', kind: 'hist', f: null, ok: () => true,
    before: 'income ($), age (yr)', after: 'z-scores of both',
    use: 'features on different scales that must be compared or penalized equally, as in regularization.',
    data: 'income in dollars (blue) and age in years (green) for 200 people. On a shared axis every age falls in the first bar.',
    domain: 'every number, as long as σ > 0. It shifts and rescales but never changes the shape.' }
];

let rngState = 1, seed = 7;
let data = {};
let selected = 0;
let tileBoxes = [];
let lambdaSlider, dataButton, addedCheckbox;

function uniform() {                    // mulberry32, shifted so that 0 is never returned
  let t = (rngState = (rngState + 0x6D2B79F5) | 0);
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
}
function stdNormal() { return Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform()); }
function poisson(m) {                   // Knuth's method
  const limit = Math.exp(-m);
  let k = 0, p = uniform();
  while (p > limit) { k++; p *= uniform(); }
  return k;
}

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const variance = a => { const m = mean(a); return mean(a.map(v => (v - m) ** 2)); };
const sd = a => Math.sqrt(variance(a));
const skewness = a => { const m = mean(a); return mean(a.map(v => (v - m) ** 3)) / sd(a) ** 3; };
// slope and intercept of the least-squares line, and its R²
function lineFit(x, y) {
  const mx = mean(x), my = mean(y);
  const sxy = mean(x.map((v, i) => (v - mx) * (y[i] - my)));
  const slope = sxy / variance(x);
  return { slope, intercept: my - slope * mx, r2: sxy * sxy / (variance(x) * variance(y)) };
}
function boxcox(v, lam) { return Math.abs(lam) < 1e-9 ? Math.log(v) : (Math.pow(v, lam) - 1) / lam; }
function boxcoxBestLambda(y) {          // grid search on the log-likelihood, λ from −2 to 2
  const sumLog = y.reduce((s, v) => s + Math.log(v), 0);
  let best = 0, bestLL = -Infinity;
  for (let k = -200; k <= 200; k++) {
    const ll = -y.length / 2 * Math.log(variance(y.map(v => boxcox(v, k / 100)))) + (k / 100 - 1) * sumLog;
    if (ll > bestLL) { bestLL = ll; best = k / 100; }
  }
  return best;
}
const fx = (v, d) => (Math.abs(v) < 0.5 * Math.pow(10, -d) ? 0 : v).toFixed(d).replace('-', '−');     // never prints −0.00

function makeData() {
  rngState = seed;
  const gen = fn => Array.from({ length: N }, (_, i) => fn(i));
  data = {
    log: { y: gen(() => 50 * Math.exp(0.6 * stdNormal())) },
    sqrt: { x: gen(i => i % 5 + 1 + 0.5 * (uniform() - 0.5)), y: gen(i => poisson(SHOP_MEANS[i % 5])) },
    boxcox: { y: gen(() => -6 * Math.log(uniform() * uniform())) },
    std: { y: gen(() => Math.round(42000 * Math.exp(0.55 * stdNormal()))),
      y2: gen(() => Math.round(Math.min(80, Math.max(18, 42 + 12 * stdNormal())))) }
  };
  const speed = gen(() => 20 + 100 * uniform()), len = gen(() => 0.05 + 1.95 * uniform());
  data.recip = { x: speed, y: speed.map(s => 1 / (s / 120 + 0.03 * stdNormal())) };
  data.square = { x: len, y: len.map(L => 2 * Math.PI * Math.sqrt(L / 9.81) + 0.04 * stdNormal()) };
  data.boxcox.best = boxcoxBestLambda(data.boxcox.y);
  // 12 problem values for each data set: 6 zeros and 6 negatives, used when the checkbox is on
  for (const d of Object.values(data)) {
    const med = d.y.slice().sort((a, b) => a - b)[N / 2];
    const lo = d.x ? Math.min(...d.x) : 0, hi = d.x ? Math.max(...d.x) : 0;
    d.addedY = Array.from({ length: N_ADDED }, (_, i) => i < 6 ? 0 : -med * (0.3 + 0.7 * uniform()));
    d.addedX = d.addedY.map(() => lo + (hi - lo) * uniform());
  }
}

// Everything the plots and the text need for one transformation
function build(t) {
  const d = data[t.id], lam = lambdaSlider.value();
  const x = d.x ? d.x.slice() : null, y = d.y.slice(), cls = y.map(() => 0);
  if (addedCheckbox.checked()) {
    d.addedY.forEach((v, i) => { y.push(v); if (x) x.push(d.addedX[i]); cls.push(t.ok(v) ? 1 : 2); });
  }
  let f = t.f;
  if (t.id === 'boxcox') f = v => boxcox(v, lam);
  if (t.id === 'std') { const m = mean(y), s = sd(y); f = v => (v - m) / s; }
  const keep = i => cls[i] < 2;
  const before = { x, y, cls }, yb = d.y, tb = yb.map(f);
  const after = { x: x && x.filter((v, i) => keep(i)), y: y.filter((v, i) => keep(i)).map(f), cls: cls.filter(c => c < 2) };
  let stat;
  if (t.id === 'sqrt') {
    const shop = g => yb.filter((v, i) => i % 5 === g), s = a => fx(sd(a), 2);
    stat = 'Spread (SD) within the smallest and the largest shop: ' + s(shop(0)) + ' and ' + s(shop(4)) + ' before, ' +
      s(shop(0).map(f)) + ' and ' + s(shop(4).map(f)) + ' after. The spread is now about the same in every shop.';
  } else if (t.line) {
    before.fit = lineFit(d.x, yb);
    after.fit = lineFit(d.x, tb);
    stat = 'R² of a straight line (dashed): ' + fx(before.fit.r2, 3) + ' before, ' + fx(after.fit.r2, 3) + ' after.';
  } else if (t.id === 'std') {
    const m2 = mean(d.y2), s2 = sd(d.y2), z2 = d.y2.map(v => (v - m2) / s2), z = y.map(f);
    before.y2 = d.y2;
    after.y2 = z2;
    stat = 'Income: mean ' + Math.round(mean(y)).toLocaleString('en-US') + ', SD ' + Math.round(sd(y)).toLocaleString('en-US') +
      ' before; mean ' + fx(mean(z), 2) + ', SD ' + fx(sd(z), 2) + ' after. Age: mean ' + fx(m2, 1) + ', SD ' + fx(s2, 1) +
      ' before; ' + fx(mean(z2), 2) + ' and ' + fx(sd(z2), 2) + ' after. Skewness of income: ' + fx(skewness(y), 2) +
      ' before and ' + fx(skewness(z), 2) + ' after.';
  } else {
    stat = 'Skewness: ' + fx(skewness(yb), 2) + ' before, ' + fx(skewness(tb), 2) + ' after (0 means symmetric).';
    if (t.id === 'boxcox') stat = 'At λ = ' + fx(lam, 1) + ', s' + stat.slice(1) + ' Maximum-likelihood λ for this data: ' + fx(d.best, 2) + '.';
  }
  const nBad = cls.filter(c => c === 2).length;
  let note = '';
  if (addedCheckbox.checked()) {
    note = nBad === 0 ? ' All 12 added values can be transformed (orange).'
      : ' Here ' + nBad + ' of the 12 added values are undefined (red) and are left out of the After plot.';
  }
  return { before, after, stat, nBad, note, afterLabel: t.id === 'boxcox' ? 'Box-Cox, λ = ' + fx(lam, 1) : t.after };
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  lambdaSlider = createSlider(-2, 2, 0.3, 0.1);
  lambdaSlider.parent(mainElement);
  lambdaSlider.position(sliderLeftMargin, drawHeight + 8);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);
  lambdaSlider.input(() => { selected = 4; });       // moving λ opens the Box-Cox panel

  dataButton = createButton('New Data');
  dataButton.parent(mainElement);
  dataButton.position(10, drawHeight + 45);
  dataButton.mousePressed(() => { seed++; makeData(); });

  addedCheckbox = createCheckbox(' Add zeros and negative values', false);
  addedCheckbox.parent(mainElement);
  addedCheckbox.position(100, drawHeight + 46);
  addedCheckbox.style('font-size', '16px');

  makeData();

  describe('A gallery of six transformations: log, square root, reciprocal, square, Box-Cox, and standardization. ' +
    'Each tile shows a small before and after plot of synthetic data. Clicking a tile opens large before and after ' +
    'plots with the formula, when to use it, statistics computed from the data, and where it is undefined. ' +
    'A slider sets the Box-Cox lambda and a checkbox adds zeros and negative values.', LABEL);
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
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Transformation Gallery', canvasWidth / 2, 8);

  // tiles: a 2 x 3 grid on the left, or 3 x 2 across the top when narrow
  const top = narrow ? 34 : 42, gap = 8, cols = narrow ? 3 : 2, rows = 6 / cols;
  const galleryW = narrow ? canvasWidth - 2 * margin : 308;
  const tileW = (galleryW - (cols - 1) * gap) / cols;
  const tileH = narrow ? 76 : (drawHeight - top - 8 - (rows - 1) * gap) / rows;
  const views = TRANSFORMS.map(build);
  tileBoxes = TRANSFORMS.map((t, i) => ({ x: margin + (i % cols) * (tileW + gap), y: top + Math.floor(i / cols) * (tileH + gap), w: tileW, h: tileH }));
  TRANSFORMS.forEach((t, i) => drawTile(t, views[i], tileBoxes[i], i === selected, narrow));

  const t = TRANSFORMS[selected], v = views[selected];
  if (narrow) {
    const y = top + rows * (tileH + gap);
    drawDetail(t, v, margin, y, canvasWidth - 2 * margin, drawHeight - 8 - y, narrow);
  } else {
    drawDetail(t, v, margin + galleryW + 10, top, canvasWidth - 2 * margin - galleryW - 10, drawHeight - top - 8, narrow);
  }
  cursor(tileBoxes.some(mouseOver) ? HAND : ARROW);

  // control label
  noStroke();
  fill('black');
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Box-Cox λ: ' + fx(lambdaSlider.value(), 1), 10, drawHeight + 18);
}

// One gallery tile: name, formula, and a small before and after plot
function drawTile(t, v, b, on, narrow) {
  fill(on ? 'lightyellow' : 'white');
  stroke(on ? 'royalblue' : 'silver');
  strokeWeight(on ? 3 : 1);
  rect(b.x, b.y, b.w, b.h, 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 11 : 14);
  text(t.name, b.x + 6, b.y + 6);
  textStyle(NORMAL);
  if (!narrow) {
    fill('dimgray');
    textSize(13);
    text(t.formula, b.x + 6, b.y + 25);
  }
  if (v.nBad) {                         // how many of the added values this transformation cannot take
    fill('crimson');
    textAlign(RIGHT, TOP);
    textSize(narrow ? 11 : 12);
    text(v.nBad + ' undef.', b.x + b.w - 6, b.y + (narrow ? 6 : 8));
  }
  const my = b.y + (narrow ? 22 : 46), mh = b.y + b.h - 6 - my, mw = (b.w - 30) / 2;
  drawPlot(t, v.before, b.x + 6, my, mw, mh, true);
  drawPlot(t, v.after, b.x + b.w - 6 - mw, my, mw, mh, true);
  const ax = b.x + b.w / 2, ay = my + mh / 2;
  stroke('gray');
  strokeWeight(1.5);
  line(ax - 6, ay, ax + 3, ay);
  noStroke();
  fill('gray');
  triangle(ax + 7, ay, ax + 1, ay - 4, ax + 1, ay + 4);
}

// A histogram or scatter plot of one view { x, y, cls, y2, fit }. mini: no axes or labels.
function drawPlot(t, v, bx, by, bw, bh, mini, title, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(bx, by, bw, bh, 4);
  const hist = t.kind === 'hist', ts = narrow ? 11 : 12;
  const px = bx + (mini ? 3 : hist ? 12 : 38), py = by + (mini ? 3 : 24);
  const pw = bx + bw - (mini ? 3 : 12) - px, ph = by + bh - (mini ? 3 : hist ? 20 : 34) - py;
  const all = v.y.concat(v.y2 || []), span = Math.max(...all) - Math.min(...all);
  const lo = Math.min(...all) - (hist ? 0 : 0.06 * span), hi = Math.max(...all) + (hist ? 0 : 0.06 * span);
  const xlo = hist ? lo : Math.min(...v.x), xhi = hist ? hi : Math.max(...v.x);
  const gx = a => px + (hist ? 0 : 5) + (a - xlo) / (xhi - xlo) * (pw - (hist ? 0 : 10));
  const gy = a => py + ph - (a - lo) / (hi - lo) * ph;
  if (!mini) {
    noStroke();
    fill('black');
    textSize(ts + 1);
    textAlign(LEFT, TOP);
    text(title, bx + 8, by + 6);
    textSize(ts);
    for (const tick of ticks(xlo, xhi, 4)) {
      stroke('gainsboro');
      line(gx(tick), py, gx(tick), py + ph);
      noStroke();
      fill('dimgray');
      textAlign(CENTER, TOP);
      text(tickLabel(tick), gx(tick), py + ph + 3);
    }
    for (const tick of hist ? [] : ticks(lo, hi, 4)) {
      stroke('gainsboro');
      line(px, gy(tick), px + pw, gy(tick));
      noStroke();
      fill('dimgray');
      textAlign(RIGHT, CENTER);
      text(tickLabel(tick), px - 4, gy(tick));
    }
    if (t.xl) {
      fill('black');
      textAlign(CENTER, TOP);
      text(t.xl, px + pw / 2, py + ph + 17);
    }
    stroke('gray');
    strokeWeight(1.5);
    line(px, py + ph, px + pw, py + ph);
    if (!hist) line(px, py, px, py + ph);
  }
  if (hist) {
    // stacked bars by class. A second feature (age) is drawn in green on the same axis,
    // and each feature is scaled to its own tallest bar.
    const bins = mini ? 10 : 24, barW = pw / bins;
    noStroke();
    [[v.y, v.cls, [65, 105, 225, 190]], [v.y2, null, [46, 139, 87, 150]]].forEach(([vals, cls, rgba]) => {
      if (!vals) return;
      const counts = [0, 1, 2].map(() => new Array(bins).fill(0));
      vals.forEach((val, i) => { counts[cls ? cls[i] : 0][Math.min(bins - 1, Math.floor((val - lo) / (hi - lo) * bins))]++; });
      const peak = Math.max(...counts[0].map((c, b) => c + counts[1][b] + counts[2][b]));
      for (let b = 0; b < bins; b++) {
        let base = py + ph;
        for (let c = 0; c < 3; c++) {
          const h = counts[c][b] / peak * (ph - 2);
          if (c === 0) fill(...rgba); else fill(CLASS_COLORS[c]);
          if (h > 0) rect(px + b * barW, base - h, barW - (mini ? 0 : 1), h);
          base -= h;
        }
      }
    });
  } else {
    if (v.fit) {                          // least-squares line of the original points, clipped to the plot
      push();
      drawingContext.beginPath();
      drawingContext.rect(px, py, pw, ph);
      drawingContext.clip();
      stroke('dimgray');
      strokeWeight(mini ? 1 : 1.5);
      drawingContext.setLineDash([5, 4]);
      line(gx(xlo), gy(v.fit.intercept + v.fit.slope * xlo), gx(xhi), gy(v.fit.intercept + v.fit.slope * xhi));
      pop();
    }
    noStroke();
    for (let c = 0; c < 3; c++) {         // problem values are drawn last, on top
      fill(CLASS_COLORS[c]);
      v.y.forEach((val, i) => { if (v.cls[i] === c) circle(gx(v.x[i]), gy(val), mini ? 2.5 : c ? 7 : 5); });
    }
  }
}

// Round tick values (1, 2, or 5 times a power of ten) covering lo to hi
function ticks(lo, hi, n) {
  const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map(m => m * p).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}
const tickLabel = v => (Math.abs(v) >= 1000 ? v / 1000 + 'k' : String(+v.toFixed(2))).replace('-', '−');

// The selected transformation: large before and after plots, then what it is for and where it fails
function drawDetail(t, v, x, y, w, h, narrow) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const ts = narrow ? 11 : 14, lh = ts + 4, pad = narrow ? 8 : 12, iw = w - 2 * pad;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 2);
  text(t.name + ':  ' + t.formula, x + pad, y + 8);
  textStyle(NORMAL);
  const plotY = y + (narrow ? 27 : 34), plotH = narrow ? 140 : 235, plotW = (iw - 10) / 2;
  drawPlot(t, v.before, x + pad, plotY, plotW, plotH, false, 'Before: ' + t.before, narrow);
  drawPlot(t, v.after, x + pad + plotW + 10, plotY, plotW, plotH, false, 'After: ' + v.afterLabel, narrow);

  // paragraphs are stacked using an estimate of the number of wrapped lines
  const paragraphs = [['Use it for ' + t.use, 'black'], ['This data: ' + t.data, 'dimgray'], [v.stat, 'darkgreen'],
    ['Defined for ' + t.domain + v.note, v.nBad ? 'crimson' : 'black']];
  let ty = plotY + plotH + (narrow ? 6 : 10);
  textWrap(WORD);
  textSize(ts);
  textLeading(lh);
  textAlign(LEFT, TOP);
  noStroke();
  for (const [str, col] of paragraphs) {
    const lines = Math.ceil(textWidth(str) / (iw * 0.9));
    fill(col);
    text(str, x + pad, ty, iw, (lines + 1) * lh);
    ty += lines * lh + (narrow ? 3 : 7);
  }
}

const mouseOver = b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
function mousePressed() {
  const i = tileBoxes.findIndex(mouseOver);
  if (i >= 0) selected = i;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
