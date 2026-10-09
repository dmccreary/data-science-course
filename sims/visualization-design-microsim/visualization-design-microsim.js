// Chart Design Playground
// CANVAS_HEIGHT: 570
// Bloom L6 (Create): students design their own chart of a restaurant tips dataset. They choose
// the chart type, the x and y columns, an optional color column, and write a title. The preview,
// the Plotly Express code, and a design check all update as they work.
//
// The preview follows what Plotly Express does with the same arguments:
//   scatter    one marker per row              line  joins the rows in table order
//   bar        rows that share an x value are stacked, so a bar is the SUM of y
//   histogram  counts rows per bin (or per category) and ignores y
//   box        quartiles by linear interpolation, whiskers to the last point within
//              1.5 IQR of the box, and points beyond that drawn as outliers
// Data: 60 seeded rows shaped like px.data.tips(), with tip = 0.9 + 0.105 * bill + noise, where
// the noise grows with the bill.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 490;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// ---- seeded data (plain JavaScript so it is ready before p5 starts) ----
let lcg = 2025;
function rnd() { lcg = (lcg * 1664525 + 1013904223) % 4294967296; return lcg / 4294967296; }
function gauss(mean, sd) { return mean + sd * Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }
const pick = list => list[Math.floor(rnd() * list.length)];
const round2 = v => Math.round(v * 100) / 100;
const ROWS = Array.from({ length: 60 }, () => {
  const day = pick(['Thur', 'Thur', 'Fri', 'Sat', 'Sat', 'Sat', 'Sun', 'Sun']);
  const time = day === 'Thur' ? (rnd() < 0.9 ? 'Lunch' : 'Dinner') : day === 'Fri' ? (rnd() < 0.4 ? 'Lunch' : 'Dinner') : 'Dinner';
  const size = pick([1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6]);
  const total_bill = round2(Math.max(5, 5 + 5.5 * size + gauss(0, 6)));
  return { total_bill, tip: round2(Math.max(1, 0.9 + 0.105 * total_bill + gauss(0, 0.25 + 0.03 * total_bill))), size, day, time, smoker: rnd() < 0.38 ? 'Yes' : 'No' };
});
// Column facts. A discrete column has a few distinct values, listed in cats.
const VARS = {};
['total_bill', 'tip', 'size', 'day', 'time', 'smoker'].forEach(name => {
  const values = ROWS.map(r => r[name]), numeric = typeof values[0] === 'number';
  const cats = [...new Set(values)];
  if (numeric) cats.sort((a, b) => a - b);
  VARS[name] = { discrete: !numeric || name === 'size', cats, max: numeric ? Math.max(...values) : 0 };
});
const PALETTE = ['royalblue', 'tomato', 'mediumseagreen', 'mediumpurple'];
const PX_NAME = { Scatter: 'scatter', Line: 'line', Bar: 'bar', Histogram: 'histogram', Box: 'box' };

let typeSelect, xSelect, ySelect, colorSelect, titleInput;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));

  typeSelect = makeSelect(Object.keys(PX_NAME), 58, 0, 96);
  xSelect = makeSelect(['total_bill', 'tip', 'size', 'day', 'time', 'smoker'], 180, 0, 100);
  ySelect = makeSelect(['tip', 'total_bill', 'size'], 306, 0, 100);
  colorSelect = makeSelect(['none', 'day', 'time', 'smoker'], 58, 1, 96);
  colorSelect.selected('time');
  titleInput = createInput('Bigger bills, bigger tips');
  titleInput.parent(document.querySelector('main'));
  titleInput.position(206, drawHeight + 45);
  titleInput.attribute('maxlength', 30);
  resizeControls();

  describe('A chart design tool. Drop-down lists choose the chart type, the x column, the y column, and a color ' +
    'column of a restaurant tips dataset, and a text box sets the title. The canvas shows a live preview of the ' +
    'chart, the Plotly Express code that would make it, and a short design check with advice.', LABEL);
}

function makeSelect(options, x, row, w) {
  const sel = createSelect();
  sel.parent(document.querySelector('main'));
  sel.position(x, drawHeight + 10 + row * 35);
  sel.size(w);
  options.forEach(o => sel.option(o));
  return sel;
}

function resizeControls() {
  titleInput.size(constrain(canvasWidth - 206 - 20, 120, 300));
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, fullW = canvasWidth - 2 * margin;
  const d = { type: typeSelect.value(), xv: xSelect.value(), yv: ySelect.value(), cv: colorSelect.value(), title: titleInput.value().trim() };
  if (d.type === 'Histogram') ySelect.attribute('disabled', ''); else ySelect.removeAttribute('disabled');
  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Chart Design Playground', canvasWidth / 2, 8);
  // control labels
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Chart', 10, drawHeight + 22);
  text('x', 164, drawHeight + 22);
  text('y', 290, drawHeight + 22);
  text('Color', 10, drawHeight + 57);
  text('Title', 164, drawHeight + 57);

  // preview on the left (or on top when narrow), code and design check beside (or below) it
  const top = narrow ? 32 : 42, bottom = drawHeight - 8;
  if (narrow) {
    drawChart(margin, top, fullW, 218, true, d);
    drawCode(margin, top + 224, fullW, 100, true, d);
    drawCheck(margin, top + 330, fullW, bottom - top - 330, true, d);
  } else {
    const chartW = fullW * 0.62, sideX = margin + chartW + 10, sideW = fullW - chartW - 10;
    drawChart(margin, top, chartW, bottom - top, false, d);
    drawCode(sideX, top, sideW, 150, false, d);
    drawCheck(sideX, top + 158, sideW, bottom - top - 158, false, d);
  }
}

// A round tick spacing that gives about five ticks, and the first tick at or above a value
function tickStep(range) {
  const raw = range / 5, pow = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / pow;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * pow;
}
const niceMax = v => Math.ceil(v / tickStep(v) - 1e-9) * tickStep(v);
const fmt = v => String(Number(v.toFixed(2)));

// The live preview of the chart
function drawChart(x, y, w, h, narrow, d) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 8);
  const ts = narrow ? 11 : 13, X = VARS[d.xv], groups = d.cv === 'none' ? [''] : VARS[d.cv].cats;
  const colorIndex = r => d.cv === 'none' ? 0 : groups.indexOf(r[d.cv]);

  // title row, then the legend row
  noStroke();
  fill(d.title ? 'black' : 'gray');
  textStyle(d.title ? BOLD : ITALIC);
  textSize(narrow ? 13 : 16);
  textAlign(LEFT, TOP);
  text(d.title || 'No title yet', x + 10, y + 7);
  textStyle(NORMAL);
  textSize(ts);
  fill('dimgray');
  textAlign(RIGHT, TOP);
  text(ROWS.length + ' sample rows', x + w - 10, y + 9);
  textAlign(LEFT, CENTER);
  let lx = x + 48;
  const ly = y + (narrow ? 33 : 38);
  if (d.cv !== 'none') {
    text(d.cv + ':', lx, ly);
    lx += textWidth(d.cv + ':') + 10;
    groups.forEach((g, i) => {
      fill(PALETTE[i]);
      rect(lx, ly - 5, 10, 10, 2);
      fill('black');
      text(g, lx + 14, ly);
      lx += textWidth(g) + 28;
    });
  }

  // x positions: a discrete x gets one slot per value, a histogram of a continuous x one slot per bin
  const l = x + 48, t = y + (narrow ? 44 : 52), pw = w - 62, ph = y + h - 36 - t;
  const stacked = d.type === 'Histogram' || (d.type === 'Bar' && X.discrete);
  const binW = tickStep(X.max) / 2;
  const slots = X.discrete ? X.cats.length : Math.ceil(X.max / binW + 1e-9), slotW = pw / slots;
  const slotOf = r => X.discrete ? X.cats.indexOf(r[d.xv]) : Math.min(slots - 1, Math.floor(r[d.xv] / binW));
  const xMax = X.discrete ? slots : d.type === 'Histogram' ? slots * binW : niceMax(X.max);
  const xp = r => X.discrete ? l + (slotOf(r) + 0.5) * slotW : l + r[d.xv] / xMax * pw;
  // stacked bars: the total (or the row count) for every slot and color group
  const sums = Array.from({ length: slots }, () => groups.map(() => 0));
  if (stacked) ROWS.forEach(r => { sums[slotOf(r)][colorIndex(r)] += d.type === 'Histogram' ? 1 : r[d.yv]; });
  const yMax = niceMax(stacked ? Math.max(...sums.map(s => s.reduce((a, b) => a + b, 0))) : VARS[d.yv].max);
  const yp = v => t + ph - v / yMax * ph;

  // grid, ticks, and axis titles
  stroke('gray');
  noFill();
  rect(l, t, pw, ph);
  for (let v = 0; v <= yMax + 1e-9; v += tickStep(yMax)) {
    stroke('gainsboro');
    line(l + 1, yp(v), l + pw - 1, yp(v));
    noStroke();
    fill('dimgray');
    textAlign(RIGHT, CENTER);
    text(fmt(v), l - 6, yp(v));
  }
  textAlign(CENTER, TOP);
  if (X.discrete) X.cats.forEach((c, i) => text(c, l + (i + 0.5) * slotW, t + ph + 5));
  else for (let v = 0; v <= xMax + 1e-9; v += tickStep(xMax)) {
    stroke('gainsboro');
    line(l + v / xMax * pw, t + 1, l + v / xMax * pw, t + ph - 1);
    noStroke();
    text(fmt(v), l + v / xMax * pw, t + ph + 5);
  }
  fill('black');
  text(d.xv, l + pw / 2, t + ph + 19);
  push();
  translate(x + 11, t + ph / 2);
  rotate(-HALF_PI);
  textAlign(CENTER, CENTER);
  text(d.type === 'Histogram' ? 'count' : d.yv, 0, 0);
  pop();

  // the marks
  if (stacked) {
    const pad = X.discrete ? slotW * 0.15 : 0.5;
    sums.forEach((s, i) => {
      let base = 0;
      s.forEach((v, c) => {
        fill(PALETTE[c]);
        rect(l + i * slotW + pad, yp(base + v), slotW - 2 * pad, yp(base) - yp(base + v));
        base += v;
      });
    });
  } else if (d.type === 'Bar') {
    ROWS.forEach(r => { fill(PALETTE[colorIndex(r)]); rect(xp(r) - 1.5, yp(r[d.yv]), 3, yp(0) - yp(r[d.yv])); });
  } else if (d.type === 'Box' && X.discrete) {
    const bw = slotW * 0.7 / groups.length;
    X.cats.forEach((cat, i) => groups.forEach((g, c) => {
      const vals = ROWS.filter(r => r[d.xv] === cat && colorIndex(r) === c).map(r => r[d.yv]).sort((a, b) => a - b);
      if (vals.length > 0) drawBox(vals, l + (i + 0.15) * slotW + (c + 0.5) * bw, bw * 0.8, yp, PALETTE[c]);
    }));
  } else if (d.type === 'Box') {
    // a box that holds one row collapses to its median line
    strokeWeight(2);
    ROWS.forEach(r => { stroke(PALETTE[colorIndex(r)]); line(xp(r) - 4, yp(r[d.yv]), xp(r) + 4, yp(r[d.yv])); });
  } else if (d.type === 'Line') {
    noFill();
    strokeWeight(1.5);
    groups.forEach((g, c) => {
      stroke(PALETTE[c]);
      beginShape();
      ROWS.forEach(r => { if (colorIndex(r) === c) vertex(xp(r), yp(r[d.yv])); });
      endShape();
    });
  } else {
    stroke('white');
    ROWS.forEach(r => { fill(PALETTE[colorIndex(r)]); circle(xp(r), yp(r[d.yv]), narrow ? 7 : 9); });
  }
  strokeWeight(1);
}

// One box: quartiles, whiskers to the last value within 1.5 IQR, and outliers as dots
function drawBox(vals, cx, bw, yp, col) {
  const q = p => { const i = (vals.length - 1) * p, lo = Math.floor(i); return vals[lo] + (vals[Math.min(lo + 1, vals.length - 1)] - vals[lo]) * (i - lo); };
  const q1 = q(0.25), q3 = q(0.75), reach = 1.5 * (q3 - q1);
  const inside = vals.filter(v => v >= q1 - reach && v <= q3 + reach), c = color(col);
  stroke(col);
  strokeWeight(1.5);
  line(cx, yp(inside[0]), cx, yp(q1));
  line(cx, yp(q3), cx, yp(inside[inside.length - 1]));
  c.setAlpha(90);
  fill(c);
  rect(cx - bw / 2, yp(q3), bw, yp(q1) - yp(q3));
  line(cx - bw / 2, yp(q(0.5)), cx + bw / 2, yp(q(0.5)));
  fill(col);
  noStroke();
  vals.filter(v => !inside.includes(v)).forEach(v => circle(cx, yp(v), 5));
}

// The Plotly Express code for the current design, wrapped to fit the panel
function drawCode(x, y, w, h, narrow, d) {
  const args = ["x='" + d.xv + "'"];
  if (d.type !== 'Histogram') args.push("y='" + d.yv + "'");
  if (d.cv !== 'none') args.push("color='" + d.cv + "'");
  if (d.title) args.push("title='" + d.title.replace(/'/g, "\\'") + "'");
  const call = ['fig = px.' + PX_NAME[d.type] + '(tips,'], maxChars = narrow ? 56 : 40;
  args.forEach((a, i) => {
    const piece = a + (i < args.length - 1 ? ',' : ')');
    if ((call[call.length - 1] + ' ' + piece).length > maxChars) call.push('    ' + piece);
    else call[call.length - 1] += ' ' + piece;
  });
  const lines = ['import plotly.express as px', 'tips = px.data.tips()', ...call, 'fig.show()'];
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(narrow ? 12 : 14);
  text('Plotly Express code', x + 10, y + 6);
  textStyle(NORMAL);
  textSize(12);
  fill('darkslateblue');
  lines.forEach((ln, i) => text(ln, x + 10, y + (narrow ? 23 : 28) + i * (narrow ? 15 : 18)));
}

// Advice on the current design: [is it fine?, text]
function designNotes(d) {
  const disc = VARS[d.xv].discrete, notes = [];
  if (d.type === 'Scatter') notes.push(disc ? [false, d.xv + ' has only a few values, so the dots stack in columns. A box plot shows each group better.']
    : [true, 'Two numeric columns: a scatter plot shows how they relate.']);
  else if (d.type === 'Line') notes.push([false, 'A line chart joins rows in table order. These rows have no time order, so the zigzag means nothing.']);
  else if (d.type === 'Histogram') notes.push([true, 'Counts the rows in each ' + (disc ? 'value' : 'bin') + ' of ' + d.xv + '. A histogram does not use y.']);
  else if (!disc) notes.push([false, 'Nearly every ' + d.xv + ' differs, so each row gets its own ' + (d.type === 'Bar' ? 'thin bar' : 'box') +
    '. Use a category like day for x.']);
  else notes.push([true, d.type === 'Bar' ? 'Each bar is the total ' + d.yv + ' for one ' + d.xv + ' value.'
    : 'One box per ' + d.xv + ' value: the median, quartiles, and spread of ' + d.yv + '.']);
  if (d.type !== 'Histogram' && d.xv === d.yv) notes.push([false, 'x and y are the same column, so the chart cannot show a relationship.']);
  else if (d.cv === d.xv) notes.push([false, 'Color repeats ' + d.xv + ', which the x-axis already shows.']);
  else if (d.cv !== 'none') notes.push([true, 'Color splits the rows by ' + d.cv + ' and adds a legend.']);
  notes.push(d.title ? [true, 'It has a title. Does it say what the reader should notice?'] : [false, 'Add a title that says what the chart shows.']);
  return notes;
}

function drawCheck(x, y, w, h, narrow, d) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 8);
  const ts = narrow ? 12 : 13, lh = ts + 3, slot = (narrow ? 2 : 3) * lh + (narrow ? 2 : 8);
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textStyle(BOLD);
  textSize(ts + 1);
  text('Design check', x + 10, y + 6);
  designNotes(d).forEach(([ok, msg], i) => {
    const ny = y + (narrow ? 24 : 30) + i * slot;
    noStroke();
    fill(ok ? 'mediumseagreen' : 'darkorange');
    circle(x + 18, ny + 8, 16);
    fill('white');
    textStyle(BOLD);
    textSize(12);
    textAlign(CENTER, CENTER);
    text(ok ? '✓' : '!', x + 18, ny + 9);
    fill(ok ? 'black' : 'saddlebrown');
    textStyle(NORMAL);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(msg, x + 32, ny, w - 42, slot);
  });
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  resizeControls();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
