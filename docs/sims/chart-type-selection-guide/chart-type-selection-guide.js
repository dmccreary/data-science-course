// Chart Type Selection Guide
// CANVAS_HEIGHT: 593
// Bloom L3 (Apply): students use a decision tree to choose a chart. They pick what they want to
// show (the goal), then the description that matches their data, and read the chart that fits,
// its Matplotlib, seaborn, or Plotly call, and any warning. "New scenario" poses a data question
// that the student answers by clicking a goal and a chart, with immediate feedback.
//
// Every thumbnail is drawn from data: the chapter's own examples (language popularity, monthly
// budget) or samples from a seeded generator. The histogram and the KDE plot use the same 200
// scores. The KDE is a real Gaussian kernel density estimate with bandwidth 4.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 548;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// ---- seeded data (plain JavaScript so it is ready before p5 starts) ----
let lcg = 2024;
function rnd() { lcg = (lcg * 1664525 + 1013904223) % 4294967296; return lcg / 4294967296; }
function gauss(mean, sd) { return mean + sd * Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }
const sample = (n, mean, sd) => Array.from({ length: n }, () => gauss(mean, sd));
const quantile = (a, p) => { const i = (a.length - 1) * p, lo = Math.floor(i); return a[lo] + (a[Math.min(lo + 1, a.length - 1)] - a[lo]) * (i - lo); };

const LANG = [28, 25, 18, 12, 8];                 // the chapter's bar chart example
const BUDGET = [1200, 400, 200, 150, 250];        // the chapter's pie chart example
const SCORES = sample(200, 75, 10);               // test scores, as in the chapter's histogram
const BINS = Array(10).fill(0);                   // ten bins of width 6 from 45 to 105
SCORES.forEach(s => { const b = Math.floor((s - 45) / 6); if (b >= 0 && b < 10) BINS[b]++; });
const KDE = Array.from({ length: 41 }, (_, i) => SCORES.reduce((sum, s) => sum + Math.exp(-0.5 * ((45 + 1.5 * i - s) / 4) ** 2), 0));
const CLASSES = [[75, 8], [70, 12], [80, 5]].map(([m, s]) => sample(30, m, s).sort((a, b) => a - b));
const PEOPLE = Array.from({ length: 30 }, () => { const h = 155 + 40 * rnd(); return [h, 0.9 * (h - 100) + gauss(0, 5)]; });
const BUBBLES = Array.from({ length: 12 }, () => { const u = 0.1 + 0.8 * rnd(); return [u, 0.15 + 0.65 * u + gauss(0, 0.07), 0.3 + rnd()]; });
const TRIO = Array.from({ length: 24 }, () => { const a = rnd(); return [a, Math.min(1, Math.max(0, 0.15 + 0.7 * a + gauss(0, 0.08))), rnd()]; });
const LINES = [[0.2, 0.035], [0.55, 0.005], [0.78, -0.02]].map(([b, m]) => Array.from({ length: 16 }, (_, i) => b + m * i + gauss(0, 0.03)));
const STOCK = (() => { let v = 0; const a = Array.from({ length: 30 }, () => (v += gauss(0, 2))); const lo = Math.min(...a), hi = Math.max(...a);
  return a.map(x => 0.08 + 0.84 * (x - lo) / (hi - lo)); })();
const FORECAST = Array.from({ length: 12 }, (_, i) => 0.3 + 0.035 * i + gauss(0, 0.02));
const AREA = Array.from({ length: 9 }, (_, i) => [0.18 + 0.01 * i, 0.12 + 0.02 * i, 0.08 + 0.045 * i]);
const TREE = [35, 20, 15, 12, 8, 6, 4];
const PAL = ['royalblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson', 'goldenrod', 'teal'];

const GOALS = [
  { title: 'Comparison', color: 'royalblue', ink: 'mediumblue' },
  { title: 'Distribution', color: 'seagreen', ink: 'darkgreen' },
  { title: 'Relationship', color: 'mediumpurple', ink: 'indigo' },
  { title: 'Composition', color: 'darkorange', ink: 'saddlebrown' },
  { title: 'Trend', color: 'crimson', ink: 'firebrick' }
];

// Three charts per goal, in goal order (chart index = goal * 3 + position)
const CHARTS = [
  { id: 'bar', when: 'Few categories', name: 'Vertical bar chart', code: ['plt.bar(languages, popularity)'],
    use: 'Compare a handful of categories. Bar heights are easy to rank at a glance.',
    ex: 'Popularity of five programming languages', warn: '3D charts: avoid them. They distort perception.' },
  { id: 'barh', when: 'Many categories', name: 'Horizontal bar chart', code: ['plt.barh(countries, population)'],
    use: 'With many categories or long names, horizontal bars leave room for every label. Sort them by value.',
    ex: 'Ten countries ranked by population' },
  { id: 'lines', when: 'Over time', name: 'Multiple-line chart',
    code: ["plt.plot(month, sales_a, label='A')", "plt.plot(month, sales_b, label='B')"],
    use: 'Compare how several groups change over the same period, with one line per group.',
    ex: 'Monthly sales of three products', warn: 'Dual y-axes: use carefully. Two scales on one chart can mislead.' },
  { id: 'hist', when: 'Single variable', name: 'Histogram', code: ['plt.hist(scores, bins=20)'],
    use: 'See the shape of one numeric variable: where values pile up, how spread out they are, and any outliers.',
    ex: 'Test scores, counted in bins' },
  { id: 'box', when: 'Compare groups', name: 'Box plot', code: ['plt.boxplot([class_a, class_b, class_c])'],
    use: 'Compare the median, quartiles, and range of several distributions side by side.',
    ex: 'Scores in classes A, B, and C' },
  { id: 'kde', when: 'Density estimate', name: 'KDE plot', code: ["sns.kdeplot(data=df, x='score', fill=True)"],
    use: 'A smooth curve that estimates the same shape as a histogram, without choosing bins.',
    ex: 'The same test scores as a smooth density curve' },
  { id: 'scatter', when: 'Two variables', name: 'Scatter plot', code: ['plt.scatter(height, weight)'],
    use: 'See whether two numeric variables move together, and spot clusters and outliers.',
    ex: 'Height and weight, one dot per person' },
  { id: 'bubble', when: 'Three variables', name: 'Bubble chart', code: ['plt.scatter(income, life_exp, s=population)'],
    use: 'A scatter plot where the size of each dot shows a third numeric variable.',
    ex: 'Income, life expectancy, and population (dot size)' },
  { id: 'pair', when: 'Many variables', name: 'Pair plot', code: ['sns.pairplot(df)'],
    use: 'A grid with one scatter plot for every pair of numeric columns. A fast first look at a new dataset.',
    ex: 'Three numeric columns, every pair plotted' },
  { id: 'pie', when: 'One snapshot', name: 'Pie chart', code: ['plt.pie(amounts, labels=categories)'],
    use: 'Show how a few parts make up one whole at a single point in time.',
    ex: 'A monthly budget in five categories', warn: 'Pie charts: only use with 2 to 5 slices. With more, use a bar chart.' },
  { id: 'area', when: 'Over time', name: 'Stacked area chart', code: ['plt.stackplot(year, phone, tablet, desktop)'],
    use: 'Show how the parts of a total change over time. The top edge is the total.',
    ex: 'Web visits from three kinds of device, by year' },
  { id: 'treemap', when: 'Many parts', name: 'Treemap', code: ["px.treemap(df, path=['folder'], values='size')"],
    use: 'Rectangles sized by value. It works when there are too many parts for a pie chart.',
    ex: 'Disk space used by seven folders' },
  { id: 'line', when: 'Over time', name: 'Line chart', code: ['plt.plot(days, stock_price)'],
    use: 'Follow one quantity through time. The slope shows how fast it rises or falls.',
    ex: 'A stock price over 30 days' },
  { id: 'band', when: 'With uncertainty', name: 'Line + confidence band',
    code: ['plt.plot(day, forecast)', 'plt.fill_between(day, low, high, alpha=0.3)'],
    use: 'Shade a band around the line to show the range the true value is likely to fall in.',
    ex: 'A forecast that is less certain further ahead' },
  { id: 'legend', when: 'Multiple series', name: 'Multiple lines + legend',
    code: ["plt.plot(day, a, label='Product A')", "plt.plot(day, b, label='Product B')", 'plt.legend()'],
    use: 'Follow several series at once. Label each line and add a legend so readers can tell them apart.',
    ex: 'Three products with a legend' }
];

const SCENARIOS = [
  { q: 'A survey asked 200 students to pick their favorite of four lunch options. Which option won, and by how much?',
    ok: ['bar'], why: 'Four categories with one value each: a vertical bar chart.' },
  { q: 'You have 1,000 exam scores and want to see where most scores fall and whether any are unusually low.',
    ok: ['hist', 'kde'], why: 'The shape of one numeric variable: a histogram (a KDE plot also works).' },
  { q: 'For each student you know the hours studied and the exam score. Does more study go with a higher score?',
    ok: ['scatter'], why: 'Two numeric variables for every student: a scatter plot.' },
  { q: 'A club spends its budget on four things. What share of the whole goes to each one?',
    ok: ['pie'], why: 'A few parts of one whole at one moment: a pie chart.' },
  { q: 'You have exam scores for three classes and want to compare their medians and spreads.',
    ok: ['box'], why: 'Several distributions side by side: a box plot.' },
  { q: 'You need to rank 25 countries with long names by population.',
    ok: ['barh'], why: 'Many categories with long labels: a horizontal bar chart.' },
  { q: 'You have the average temperature of one city for every month of the last ten years. How has it changed?',
    ok: ['line'], why: 'One quantity followed through time: a line chart.' },
  { q: 'Monthly sales for three products over two years: which product is ahead, and when did that change?',
    ok: ['lines', 'legend'], why: 'Several series over the same period: one line per product, with a legend.' },
  { q: 'A weather model predicts the temperature for the next ten days, with a low and a high estimate for each day.',
    ok: ['band'], why: 'A trend with uncertainty: a line with a confidence band.' },
  { q: 'How has the share of web visits from phones, tablets, and desktops changed over ten years?',
    ok: ['area'], why: 'Parts of a whole that change over time: a stacked area chart.' }
];

let selChart = 0;                 // index into CHARTS; its goal is floor(selChart / 3)
let scenario = -1;                // index into SCENARIOS, or -1 before practice starts
let picked = false;               // true once a chart has been clicked for the current scenario
let scenarioButton;
let hitBoxes = [];                // clickable goals and chart cards: { x, y, w, h, goal } or { ..., chart }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  scenarioButton = createButton('New scenario');
  scenarioButton.parent(mainElement);
  scenarioButton.position(10, drawHeight + 10);
  scenarioButton.mousePressed(() => { scenario = (scenario + 1) % SCENARIOS.length; picked = false; });

  describe('A decision tree for choosing a chart. The question What do you want to show leads to five goals: ' +
    'comparison, distribution, relationship, composition, and trend. Each goal leads to three chart types with small ' +
    'example charts. Clicking a chart shows a larger example, when to use it, the Python call, and any warning. ' +
    'A practice strip poses data scenarios and gives feedback on the chart the student picks.', LABEL);
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
  const fullW = canvasWidth - 2 * margin, cx = canvasWidth / 2, selGoal = floor(selChart / 3);
  textWrap(WORD);
  textStyle(NORMAL);
  noStroke();
  fill('black');
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Which Chart Should I Use?', cx, 8);

  // vertical layout: start question, goals, chart cards, detail panel, practice strip
  const qY = narrow ? 34 : 40, qH = narrow ? 22 : 26, chipY = narrow ? 68 : 80, chipH = narrow ? 24 : 28;
  const cardY = narrow ? 106 : 124, cardH = narrow ? 96 : 112;
  const detailY = cardY + cardH + 8, stripH = narrow ? 96 : 74, stripY = drawHeight - 8 - stripH;
  hitBoxes = [];

  // start question
  const qW = narrow ? 220 : 290;
  pen('lightyellow', 'goldenrod', 1.5);
  rect(cx - qW / 2, qY, qW, qH, qH / 2);
  noStroke();
  fill('black');
  textStyle(BOLD);
  textSize(narrow ? 13 : 16);
  textAlign(CENTER, CENTER);
  text('What do you want to show?', cx, qY + qH / 2 + 1);

  // goals
  const gap = narrow ? 4 : 10, chipW = (fullW - 4 * gap) / 5;
  GOALS.forEach((goal, g) => {
    const x = margin + g * (chipW + gap), on = g === selGoal;
    const box = { x, y: chipY, w: chipW, h: chipH, goal: g };
    pen(null, on ? goal.color : 'silver', on ? 2.5 : 1);
    line(cx, qY + qH, x + chipW / 2, chipY);
    pen(on ? goal.color : mouseOver(box) ? 'lavender' : 'white', goal.color, 1.5);
    rect(x, chipY, chipW, chipH, 6);
    noStroke();
    fill(on ? 'white' : goal.ink);
    textSize(narrow ? 11 : 15);
    text(goal.title, x + chipW / 2, chipY + chipH / 2 + 1);
    hitBoxes.push(box);
  });

  // the three charts of the selected goal
  const goal = GOALS[selGoal], cgap = narrow ? 6 : 14, cardW = (fullW - 2 * cgap) / 3;
  const fromX = margin + selGoal * (chipW + gap) + chipW / 2;
  for (let k = 0; k < 3; k++) {
    const i = selGoal * 3 + k, c = CHARTS[i], x = margin + k * (cardW + cgap), on = i === selChart;
    const box = { x, y: cardY, w: cardW, h: cardH, chart: i };
    pen(null, goal.color, on ? 2.5 : 1);
    line(fromX, chipY + chipH, x + cardW / 2, cardY);
    pen(mouseOver(box) && !on ? 'ghostwhite' : 'white', goal.color, on ? 3 : 1);
    rect(x, cardY, cardW, cardH, 8);
    noStroke();
    fill(goal.ink);
    textStyle(BOLD);
    textSize(narrow ? 11 : 14);
    textAlign(CENTER, TOP);
    text(c.when, x + cardW / 2, cardY + 6);
    const thumbW = min(cardW - 16, 150);
    drawChart(c.id, x + (cardW - thumbW) / 2, cardY + (narrow ? 22 : 26), thumbW, cardH - (narrow ? 44 : 52), goal.color);
    noStroke();
    fill('black');
    textStyle(NORMAL);
    textSize(narrow ? 11 : 14);
    textAlign(CENTER, BOTTOM);
    text(c.name, x + cardW / 2, cardY + cardH - 5);
    hitBoxes.push(box);
  }
  cursor(hitBoxes.some(mouseOver) ? HAND : ARROW);

  drawDetail(margin, detailY, fullW, stripY - 8 - detailY, narrow);
  drawPractice(margin, stripY, fullW, stripH, narrow);
}

// Fill, stroke, and stroke weight of a shape in one call (null means none)
function pen(fillColor, strokeColor, weight) {
  if (fillColor) fill(fillColor); else noFill();
  if (strokeColor) stroke(strokeColor); else noStroke();
  strokeWeight(weight || 1);
}

// The selected chart: a larger example, when to use it, the code, and a warning if there is one
function drawDetail(x, y, w, h, narrow) {
  const c = CHARTS[selChart], goal = GOALS[floor(selChart / 3)];
  pen('white', goal.color, 2);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 12 : 15, lh = ts + 4;
  const chartW = narrow ? 130 : 250, chartH = narrow ? 84 : h - 2 * pad;
  drawChart(c.id, x + pad, y + pad, chartW, chartH, goal.color);
  const say = (str, sx, sy, sw, lines, col, style, size) => {
    noStroke();
    fill(col);
    textStyle(style);
    textSize(size);
    textLeading(size + 4);
    textAlign(LEFT, TOP);
    text(str, sx, sy, sw, lines * (size + 4) + 3);
  };
  // heading beside the chart
  const tx = x + pad + chartW + (narrow ? 10 : 16), tw = x + w - pad - tx;
  say(c.name, tx, y + pad, tw, 1, goal.ink, BOLD, narrow ? 14 : 18);
  say(goal.title + '  →  ' + c.when, tx, y + pad + (narrow ? 20 : 25), tw, 1, 'dimgray', NORMAL, narrow ? 11 : 13);
  say('Example: ' + c.ex, tx, y + pad + (narrow ? 38 : 44), tw, narrow ? 3 : 1, 'dimgray', ITALIC, narrow ? 11 : 13);

  // body: beside the chart when wide, under it when narrow
  const bx = narrow ? x + pad : tx, bw = narrow ? w - 2 * pad : tw;
  let by = y + pad + (narrow ? chartH + 6 : 70);
  say(c.use, bx, by, bw, 2, 'black', NORMAL, ts);
  by += 2 * lh + 4;
  const codeSize = narrow ? 12 : 14, codeH = c.code.length * (codeSize + 4) + 8;
  pen('whitesmoke', 'silver');
  rect(bx, by, bw, codeH, 5);
  c.code.forEach((ln, i) => say(ln, bx + 8, by + 5 + i * (codeSize + 4), bw - 16, 1, 'darkslateblue', BOLD, codeSize));
  if (c.warn) say('Warning.  ' + c.warn, bx, by + codeH + 6, bw, 2, 'firebrick', BOLD, narrow ? 12 : 14);
  textStyle(NORMAL);
}

// The practice strip: a scenario and feedback on the chart the student picked
function drawPractice(x, y, w, h, narrow) {
  pen('lightyellow', 'goldenrod');
  rect(x, y, w, h, 8);
  const ts = narrow ? 12 : 14, lh = ts + 4, pad = narrow ? 8 : 12;
  const leftW = narrow ? w - 2 * pad : (w - 3 * pad) * 0.56;
  noStroke();
  fill('black');
  textAlign(LEFT, TOP);
  textLeading(lh);
  textStyle(BOLD);
  textSize(ts);
  text(scenario < 0 ? 'Practice' : 'Practice scenario ' + (scenario + 1) + ' of ' + SCENARIOS.length, x + pad, y + 6);
  textStyle(NORMAL);
  text(scenario < 0 ? 'Press New scenario to get a question about some data. Then click the goal and the chart that fit it.'
    : SCENARIOS[scenario].q, x + pad, y + 6 + lh, leftW, 3 * lh);
  if (scenario < 0) return;

  // feedback: right chart, right goal only, or wrong goal
  const sc = SCENARIOS[scenario], goalOf = id => floor(CHARTS.findIndex(ch => ch.id === id) / 3);
  const myGoal = floor(selChart / 3);
  let msg = 'Click a goal, then click the chart that fits the data.', col = 'dimgray';
  if (picked && sc.ok.includes(CHARTS[selChart].id)) { msg = 'Correct.  ' + sc.why; col = 'darkgreen'; }
  else if (picked && sc.ok.some(id => goalOf(id) === myGoal)) { msg = 'Right goal, but another ' + GOALS[myGoal].title.toLowerCase() +
    ' chart fits this data better. Read the three descriptions again.'; col = 'chocolate'; }
  else if (picked) { msg = 'Not yet. Decide the goal first: comparison, distribution, relationship, composition, ' +
    'or trend?'; col = 'firebrick'; }
  fill(col);
  textStyle(BOLD);
  if (narrow) text(msg, x + pad, y + 6 + 3.2 * lh, leftW, 2 * lh + 3);
  else text(msg, x + 2 * pad + leftW, y + 6, w - 3 * pad - leftW, h - 10);
  textStyle(NORMAL);
}

// One example chart drawn inside the frame (x, y, w, h). Data are in 0..1 units of the plot area.
function drawChart(kind, x, y, w, h, col) {
  pen('white', 'silver');
  rect(x, y, w, h, 4);
  const big = w > 160, p = big ? 14 : 6, thick = big ? 2.5 : 1.5;
  const L = x + p, B = y + h - p, W = w - 2 * p, H = h - 2 * p;
  const X = u => L + u * W, Y = v => B - v * H;
  const soft = alpha => { const c = color(col); c.setAlpha(alpha); return c; };
  const poly = (vals, c) => {
    pen(null, c, thick);
    beginShape();
    vals.forEach((v, i) => vertex(X(i / (vals.length - 1)), Y(v)));
    endShape();
  };
  if (!['pie', 'treemap', 'pair'].includes(kind)) {
    pen(null, 'gray');
    line(L, y + p, L, B);
    line(L, B, L + W, B);
  }
  pen(col, null);
  if (kind === 'bar') {
    LANG.forEach((v, i) => rect(X((i + 0.15) / 5), Y(v / 30), 0.7 * W / 5, v / 30 * H));
  } else if (kind === 'barh') {
    for (let i = 0; i < 10; i++) rect(L, y + p + (i + 0.15) * H / 10, 0.95 * W * Math.pow(0.84, i), 0.7 * H / 10);
  } else if (kind === 'hist') {
    const top = Math.max(...BINS);
    BINS.forEach((n, i) => rect(X(i / 10), Y(0.95 * n / top), W / 10 - 1, 0.95 * n / top * H));
  } else if (kind === 'kde') {
    const top = Math.max(...KDE);
    pen(soft(90), col, thick);
    beginShape();
    vertex(X(0), Y(0));
    KDE.forEach((d, i) => vertex(X(i / 40), Y(0.95 * d / top)));
    vertex(X(1), Y(0));
    endShape();
  } else if (kind === 'box') {
    // five-number summary of each class on a score axis from 40 to 110
    const S = v => Y((v - 40) / 70), bw = W * 0.2;
    CLASSES.forEach((a, i) => {
      const mid = X((i + 0.5) / 3), q1 = S(quantile(a, 0.25)), q2 = S(quantile(a, 0.5)), q3 = S(quantile(a, 0.75));
      pen(PAL[i], 'black');
      line(mid, S(a[0]), mid, S(a[a.length - 1]));
      rect(mid - bw / 2, q3, bw, q1 - q3);
      strokeWeight(thick);
      line(mid - bw / 2, q2, mid + bw / 2, q2);
    });
  } else if (kind === 'scatter') {
    PEOPLE.forEach(([ht, wt]) => circle(X((ht - 150) / 50), Y((wt - 35) / 65), big ? 8 : 4));
  } else if (kind === 'bubble') {
    fill(soft(130));
    BUBBLES.forEach(([u, v, s]) => circle(X(u), Y(v), Math.sqrt(s) * (big ? 30 : 13)));
  } else if (kind === 'pair') {
    // 3 x 3 grid: a histogram on the diagonal, a scatter plot for every pair elsewhere
    const cw = W / 3, ch = H / 3;
    for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) {
      const gx = L + q * cw, gy = y + p + r * ch;
      pen(null, 'silver');
      rect(gx + 1, gy + 1, cw - 2, ch - 2);
      pen(col, null);
      if (r !== q) TRIO.forEach(t => circle(gx + 4 + t[q] * (cw - 8), gy + ch - 4 - t[r] * (ch - 8), big ? 4 : 2));
      else for (let b = 0; b < 4; b++) {
        const n = TRIO.filter(t => Math.min(3, Math.floor(t[r] * 4)) === b).length / 12 * (ch - 6);
        rect(gx + 3 + b * (cw - 6) / 4, gy + ch - 2 - n, (cw - 6) / 4 - 1, n);
      }
    }
  } else if (kind === 'pie') {
    const total = BUDGET.reduce((a, b) => a + b, 0), d = min(W, H);
    let a0 = -HALF_PI;
    BUDGET.forEach((v, i) => {
      pen(PAL[i], 'white');
      arc(x + w / 2, y + h / 2, d, d, a0, a0 + TWO_PI * v / total, PIE);
      a0 += TWO_PI * v / total;
    });
  } else if (kind === 'area') {
    for (let s = 2; s >= 0; s--) {
      fill(PAL[s]);
      beginShape();
      vertex(X(0), Y(0));
      AREA.forEach((row, i) => vertex(X(i / 8), Y(row.slice(0, s + 1).reduce((a, b) => a + b, 0) / 1.2)));
      vertex(X(1), Y(0));
      endShape(CLOSE);
    }
  } else if (kind === 'treemap') {
    // slice the remaining rectangle along its longer side, one value at a time
    let rx = L, ry = y + p, rw = W, rh = H, left = TREE.reduce((a, b) => a + b, 0);
    TREE.forEach((v, i) => {
      pen(PAL[i], 'white', big ? 2 : 1);
      if (rw >= rh) { const t = rw * v / left; rect(rx, ry, t, rh); rx += t; rw -= t; }
      else { const t = rh * v / left; rect(rx, ry, rw, t); ry += t; rh -= t; }
      left -= v;
    });
  } else if (kind === 'line') {
    poly(STOCK, col);
  } else if (kind === 'band') {
    const n = FORECAST.length - 1;
    fill(soft(60));
    beginShape();
    FORECAST.forEach((v, i) => vertex(X(i / n), Y(v + 0.04 + 0.022 * i)));
    for (let i = n; i >= 0; i--) vertex(X(i / n), Y(FORECAST[i] - 0.04 - 0.022 * i));
    endShape(CLOSE);
    poly(FORECAST, col);
  } else {
    // 'lines' and 'legend': three series on one scale. The legend version adds a key.
    LINES.forEach((s, i) => poly(s, PAL[i]));
    if (kind !== 'legend') return;
    const lw = big ? 96 : 22, rowH = big ? 16 : 6, lx = L + W - lw - 2, ly = B - 3 * rowH - (big ? 10 : 6);
    pen('white', 'silver');
    rect(lx, ly, lw, 3 * rowH + 4, 3);
    LINES.forEach((s, i) => {
      const yy = ly + 2 + (i + 0.5) * rowH;
      pen(null, PAL[i], big ? 3 : 2);
      line(lx + 4, yy, lx + (big ? 22 : 18), yy);
      if (!big) return;
      noStroke();
      fill('black');
      textStyle(NORMAL);
      textSize(12);
      textAlign(LEFT, CENTER);
      text('Product ' + 'ABC'[i], lx + 28, yy + 1);
    });
  }
}

// Clicking a goal opens its three charts. Clicking a chart selects it and answers the scenario.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = hitBoxes.find(mouseOver);
  if (!box) return;
  if (box.chart !== undefined) { selChart = box.chart; picked = scenario >= 0; }
  else if (box.goal !== floor(selChart / 3)) { selChart = box.goal * 3; picked = false; }
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
