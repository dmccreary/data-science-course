// Plotly Express Quick Reference Card
// CANVAS_HEIGHT: 645
// Bloom L1 (Remember): a four-part reference card for Plotly Express (chart functions, essential
// parameters, layout settings, and saving) plus a strip of six templates. Students click an
// entry, or step with Next and Previous, to recall what it does and see it in a line of code.
// "Quiz me" hides the meanings so they can recall each one before pressing Reveal.
//
// The template swatches use the real colors of each Plotly template (plot background, grid
// color, and the first two colors of its color sequence), which is why they are hex values.

let containerWidth;
let canvasWidth = 400;
let drawHeight = 600;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// nameW is the width of the name column when the meanings are shown beside the names
const SECTIONS = [
  { title: 'Common Chart Functions', hint: 'import plotly.express as px', color: 'indigo', nameW: 122 },
  { title: 'Essential Parameters', hint: 'px.scatter(df, ...)', color: 'rebeccapurple', nameW: 128 },
  { title: 'Layout Customization', hint: 'fig.update_layout(...)', color: 'slateblue', nameW: 104 },
  { title: 'Saving Options', hint: '', color: 'darkorchid', nameW: 205 },
  { title: 'Templates', hint: '', color: 'purple', nameW: 0 }
];

// [section, name, meaning, example code, note]
const ITEMS = [
  [0, 'px.line()', 'Line charts', ["fig = px.line(df, x='Month', y='Sales', markers=True)"],
    'Joins the rows in the order they appear in df. Use it for trends over time. markers=True also draws a dot at each point.'],
  [0, 'px.scatter()', 'Scatter plots', ["fig = px.scatter(df, x='total_bill', y='tip')"],
    'One marker for every row. Use it to see how two numeric columns relate.'],
  [0, 'px.bar()', 'Bar charts', ["fig = px.bar(df, x='day', y='total_bill')"],
    'Draws one rectangle per row and stacks the rows that share an x value, so each bar shows the total for that day.'],
  [0, 'px.histogram()', 'Histograms', ["fig = px.histogram(df, x='total_bill', nbins=20)"],
    'Sorts one numeric column into bins and counts the rows in each bin. nbins sets the most bins Plotly may use.'],
  [0, 'px.box()', 'Box plots', ["fig = px.box(df, x='day', y='total_bill')"],
    'One box per day: the median, the quartiles, the whiskers, and any outliers as separate points.'],
  [0, 'px.pie()', 'Pie charts', ["fig = px.pie(df, names='day', values='tip')"],
    'Takes names and values instead of x and y, and adds up the values for each name. Keep it to 2 to 5 slices.'],
  [0, 'px.area()', 'Area charts', ["fig = px.area(df, x='year', y='pop', color='continent')"],
    'A line chart filled down to the axis. With color, the areas are stacked on top of each other.'],
  [0, 'px.violin()', 'Violin plots', ["fig = px.violin(df, x='day', y='total_bill', box=True)"],
    'Shows the shape of each distribution as a mirrored density curve. box=True draws a small box plot inside.'],
  [1, 'x, y', 'Data columns', ["px.scatter(df, x='total_bill', y='tip')"],
    'The names of the df columns to put on the horizontal and vertical axes.'],
  [1, 'color', 'Color by category', ["px.scatter(df, x='total_bill', y='tip', color='day')"],
    'A text column gives one color per category and a legend. A numeric column gives a continuous color scale.'],
  [1, 'size', 'Size by value', ["px.scatter(df, x='gdpPercap', y='lifeExp', size='pop')"],
    'Makes each marker bigger for larger values of a numeric column. It turns a scatter plot into a bubble chart.'],
  [1, 'hover_data', 'Extra tooltip info', ["px.scatter(df, x='total_bill', y='tip',", "    hover_data=['size', 'time'])"],
    'A list of extra columns to show in the tooltip when the mouse is over a point.'],
  [1, 'title', 'Chart title', ["px.line(df, x='Month', y='Sales',", "    title='Monthly Sales Trend')"],
    'The text shown at the top of the figure.'],
  [1, 'labels', 'Rename axis labels', ["px.bar(df, x='day', y='tip', labels={'tip': 'Tip ($)'})"],
    'A dictionary from column name to display name. It renames axis titles, legend titles, and hover text.'],
  [1, 'facet_col', 'Small multiples (columns)', ["px.scatter(df, x='total_bill', y='tip', facet_col='time')"],
    'Splits the chart into side-by-side panels, one for each value of the column.'],
  [1, 'facet_row', 'Small multiples (rows)', ["px.scatter(df, x='total_bill', y='tip', facet_row='sex')"],
    'Splits the chart into panels stacked from top to bottom, one for each value of the column.'],
  [1, 'animation_frame', 'Animate over values', ["px.scatter(df, x='gdpPercap', y='lifeExp',", "    animation_frame='year')"],
    'Adds a play button and a slider. Each value of the column becomes one frame of the animation.'],
  [2, 'title', 'Title text', ["fig.update_layout(title='My Title')"], 'Sets or replaces the title after the figure has been made.'],
  [2, 'xaxis_title', 'Label under the x-axis', ["fig.update_layout(xaxis_title='X Label')"],
    'The text under the x-axis. It is short for xaxis=dict(title=...).'],
  [2, 'yaxis_title', 'Label beside the y-axis', ["fig.update_layout(yaxis_title='Y Label')"],
    'The text beside the y-axis. Include the units, as in Tip ($).'],
  [2, 'legend_title', 'Heading of the legend', ["fig.update_layout(legend_title='Legend')"], 'The heading above the legend entries.'],
  [2, 'template', 'Theme for the whole figure', ["fig.update_layout(template='plotly_white')"],
    'Applies a whole theme at once: background, grid lines, fonts, and the color sequence. See the templates strip.'],
  [2, 'height', 'Height in pixels', ['fig.update_layout(height=500)'], 'The height of the figure in pixels.'],
  [2, 'width', 'Width in pixels', ['fig.update_layout(width=800)'],
    'The width of the figure in pixels. Without it, the figure stretches to fit the page.'],
  [3, "fig.write_html('chart.html')", 'Interactive HTML file', ["fig.write_html('chart.html')"],
    'Saves one HTML file that opens in any browser and keeps hover, zoom, and legend clicks.'],
  [3, "fig.write_image('chart.png')", 'Static PNG image', ["fig.write_image('chart.png', scale=2)"],
    'A picture for slides and documents. Static export needs the kaleido package. scale=2 doubles the resolution.'],
  [3, "fig.write_image('chart.pdf')", 'PDF for print', ["fig.write_image('chart.pdf')"],
    'The file extension chooses the format. PDF is a vector format that stays sharp when printed.'],
  [3, "fig.write_image('chart.svg')", 'SVG vector image', ["fig.write_image('chart.svg')"],
    'A vector image that stays sharp at any size and can be edited in a drawing program.'],
  [3, 'fig.show()', 'Display the figure', ['fig.show()'],
    'Displays the interactive figure in a Jupyter notebook. From a plain script it opens in the web browser.'],
  [4, 'plotly', 'Default theme', 0, 'The default. A light blue-gray plot area with white grid lines.'],
  [4, 'plotly_white', 'White background', 0, 'A white background with pale grid lines. A clean choice for reports.'],
  [4, 'plotly_dark', 'Dark background', 0, 'A dark background with light text, for dark slides and dashboards.'],
  [4, 'ggplot2', 'ggplot2 style', 0, 'A gray plot area and colors in the style of the R package ggplot2.'],
  [4, 'seaborn', 'seaborn style', 0, 'A soft gray plot area and muted colors in the style of seaborn.'],
  [4, 'simple_white', 'Minimal, no grid', 0, 'A white background, no grid lines, and visible axis lines. It suits printed figures.']
];
const TEMPLATE_STYLE = {
  plotly: { bg: '#E5ECF6', grid: 'white', a: '#636EFA', b: '#EF553B' },
  plotly_white: { bg: 'white', grid: '#EBF0F8', a: '#636EFA', b: '#EF553B' },
  plotly_dark: { bg: '#111111', grid: '#283442', a: '#636EFA', b: '#EF553B' },
  ggplot2: { bg: '#EDEDED', grid: 'white', a: '#F8766D', b: '#A3A500' },
  seaborn: { bg: '#EAEAF2', grid: 'white', a: '#4C72B0', b: '#DD8452' },
  simple_white: { bg: 'white', grid: null, a: '#1F77B4', b: '#FF7F0E', axis: '#242424' }
};
const SWATCH_LINES = [[0.25, 0.4, 0.35, 0.6, 0.55, 0.8], [0.7, 0.55, 0.6, 0.35, 0.4, 0.2]];

let selected = 0;
let revealed = false;             // in quiz mode: has the answer for the selected entry been shown?
let prevButton, nextButton, revealButton, quizCheckbox;
let rowBoxes = [];                // clickable entries: { x, y, w, h, index }

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);

  prevButton = createButton('Previous');
  prevButton.parent(mainElement);
  prevButton.position(10, drawHeight + 10);
  prevButton.mousePressed(() => choose((selected + ITEMS.length - 1) % ITEMS.length));

  nextButton = createButton('Next');
  nextButton.parent(mainElement);
  nextButton.position(86, drawHeight + 10);
  nextButton.mousePressed(() => choose((selected + 1) % ITEMS.length));

  quizCheckbox = createCheckbox(' Quiz me', false);
  quizCheckbox.parent(mainElement);
  quizCheckbox.position(140, drawHeight + 11);
  quizCheckbox.style('font-size', '16px');
  quizCheckbox.changed(() => { revealed = false; });

  revealButton = createButton('Reveal');
  revealButton.parent(mainElement);
  revealButton.position(245, drawHeight + 10);
  revealButton.mousePressed(() => { revealed = true; });

  describe('A reference card for Plotly Express with four panels: common chart functions, essential parameters, ' +
    'layout customization, and saving options, plus a strip of six template swatches. Clicking an entry shows its ' +
    'meaning, an example line of code, and a note. A quiz checkbox hides the meanings until Reveal is pressed.', LABEL);
}

function choose(i) {
  selected = i;
  revealed = false;
}

function draw() {
  updateCanvasSize();

  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  fill('white');
  rect(0, drawHeight, canvasWidth, controlHeight);

  const narrow = canvasWidth < 600, quiz = quizCheckbox.checked();
  if (quiz) revealButton.removeAttribute('disabled'); else revealButton.attribute('disabled', '');
  textWrap(WORD);
  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Plotly Express Quick Reference', canvasWidth / 2, 8);

  // four cards in a 2 x 2 grid, then the templates strip, then the detail panel
  const gap = narrow ? 5 : 8, fullW = canvasWidth - 2 * margin, cardW = (fullW - gap) / 2;
  const headH = narrow ? 20 : 24, rowH = narrow ? 15 : 17;
  const count = s => ITEMS.filter(it => it[0] === s).length;
  let y = narrow ? 32 : 40;
  rowBoxes = [];
  for (let s = 0; s < 4; s += 2) {
    const cardH = headH + max(count(s), count(s + 1)) * rowH + 6;
    drawCard(s, margin, y, cardW, cardH, headH, rowH, narrow, quiz);
    drawCard(s + 1, margin + cardW + gap, y, cardW, cardH, headH, rowH, narrow, quiz);
    y += cardH + gap;
  }
  const stripH = narrow ? 64 : 56;
  drawTemplates(margin, y, fullW, stripH, narrow);
  y += stripH + gap;
  cursor(rowBoxes.some(mouseOver) ? HAND : ARROW);
  drawDetail(margin, y, fullW, drawHeight - 8 - y, narrow, quiz);
}

// A frame with a colored header bar of height headH
function drawFrame(s, x, y, w, h, headH, narrow) {
  const sec = SECTIONS[s];
  fill('white');
  stroke(sec.color);
  strokeWeight(1.5);
  rect(x, y, w, h, 8);
  noStroke();
  fill(sec.color);
  if (headH === 0) return;                 // the wide templates strip draws its own label
  rect(x, y, w, headH, 8, 8, 0, 0);
  fill('white');
  textStyle(BOLD);
  textSize(narrow ? 12 : 15);
  textAlign(LEFT, CENTER);
  text(sec.title, x + 9, y + headH / 2 + 1);
  if (!narrow) {
    textStyle(NORMAL);
    textSize(12);
    textAlign(RIGHT, CENTER);
    text(sec.hint, x + w - 9, y + headH / 2 + 1);
  }
}

// Highlight behind the selected or hovered entry
function drawHighlight(box, s) {
  if (box.index !== selected && !mouseOver(box)) return;
  const c = color(SECTIONS[s].color);
  c.setAlpha(box.index === selected ? 60 : 22);
  noStroke();
  fill(c);
  rect(box.x, box.y, box.w, box.h, 4);
}

// One of the four cards: its entries, with the meaning beside each name when there is room
function drawCard(s, x, y, w, h, headH, rowH, narrow, quiz) {
  drawFrame(s, x, y, w, h, headH, narrow);
  let k = 0;
  ITEMS.forEach((it, i) => {
    if (it[0] !== s) return;
    const box = { x: x + 3, y: y + headH + 3 + k * rowH, w: w - 6, h: rowH - 1, index: i };
    drawHighlight(box, s);
    noStroke();
    fill(SECTIONS[s].color);
    textStyle(BOLD);
    textSize(narrow ? 11 : 13);
    textAlign(LEFT, CENTER);
    text(it[1], x + 9, box.y + rowH / 2);
    if (!narrow) {
      fill('dimgray');
      textStyle(NORMAL);
      text(quiz ? '?' : it[2], x + 9 + SECTIONS[s].nameW, box.y + rowH / 2);
    }
    rowBoxes.push(box);
    k++;
  });
  textStyle(NORMAL);
}

// The templates strip: one swatch per template
function drawTemplates(x, y, w, h, narrow) {
  const headH = narrow ? 18 : 0, labelW = narrow ? 0 : 96;
  if (narrow) drawFrame(4, x, y, w, h, headH, true);
  else {
    drawFrame(4, x, y, w, h, 0, false);
    fill(SECTIONS[4].color);
    rect(x, y, labelW, h, 8, 0, 0, 8);
    fill('white');
    textStyle(BOLD);
    textSize(15);
    textAlign(CENTER, CENTER);
    text('Templates', x + labelW / 2, y + h / 2);
  }
  const names = ITEMS.filter(it => it[0] === 4), cellW = (w - labelW - 6) / names.length;
  names.forEach((it, k) => {
    const box = { x: x + labelW + 3 + k * cellW, y: y + headH + 3, w: cellW - 2, h: h - headH - 6, index: ITEMS.indexOf(it) };
    drawHighlight(box, 4);
    const sw = min(cellW - 14, 66), sh = box.h - (narrow ? 18 : 20);
    drawSwatch(it[1], box.x + (box.w - sw) / 2, box.y + 3, sw, sh);
    noStroke();
    fill('black');
    textStyle(NORMAL);
    textSize(narrow ? 11 : 13);
    textAlign(CENTER, BOTTOM);
    text(it[1], box.x + box.w / 2, box.y + box.h - 1);
    rowBoxes.push(box);
  });
}

// A small two-line chart drawn with one template's background, grid, and colors
function drawSwatch(name, x, y, w, h) {
  const t = TEMPLATE_STYLE[name];
  fill(t.bg);
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h);
  if (t.grid) {
    stroke(t.grid);
    for (let i = 1; i < 4; i++) {
      line(x + i * w / 4, y + 1, x + i * w / 4, y + h - 1);
      if (i < 3) line(x + 1, y + i * h / 3, x + w - 1, y + i * h / 3);
    }
  } else {
    stroke(t.axis);
    line(x + 3, y + 2, x + 3, y + h - 3);
    line(x + 3, y + h - 3, x + w - 2, y + h - 3);
  }
  noFill();
  strokeWeight(w > 80 ? 3 : 2);
  SWATCH_LINES.forEach((vals, k) => {
    stroke(k === 0 ? t.a : t.b);
    beginShape();
    vals.forEach((v, i) => vertex(x + (0.08 + 0.84 * i / (vals.length - 1)) * w, y + h - v * h));
    endShape();
  });
}

// The selected entry: its meaning, a line of code, and a note
function drawDetail(x, y, w, h, narrow, quiz) {
  const it = ITEMS[selected], sec = SECTIONS[it[0]], isTemplate = it[0] === 4;
  const hidden = quiz && !revealed;
  fill('white');
  stroke(sec.color);
  strokeWeight(2);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 14, ts = narrow ? 12 : 15, lh = ts + 4;
  const swatchW = isTemplate ? (narrow ? 96 : 150) : 0;
  const ix = x + pad, iw = w - 2 * pad - (swatchW ? swatchW + 12 : 0);
  const say = (str, sy, lines, col, style, size) => {
    noStroke();
    fill(col);
    textStyle(style);
    textSize(size);
    textLeading(size + 4);
    textAlign(LEFT, TOP);
    text(str, ix, sy, iw, lines * (size + 4) + 3);
  };
  const inSection = ITEMS.filter(e => e[0] === it[0]);
  let cy = y + pad;
  say(it[1], cy, 1, sec.color, BOLD, narrow ? 15 : 18);
  // where the entry sits: under the name when narrow, at the right of the name line when wide
  const where = sec.title + '  ·  ' + (inSection.indexOf(it) + 1) + ' of ' + inSection.length;
  if (narrow) {
    say(where, cy + 20, 1, 'dimgray', NORMAL, 11);
    cy += 38;
  } else {
    fill('dimgray');
    textStyle(NORMAL);
    textSize(13);
    textAlign(RIGHT, TOP);
    text(where, ix + iw, cy + 4);
    cy += 27;
  }
  if (isTemplate) drawSwatch(it[1], x + w - pad - swatchW, y + pad, swatchW, narrow ? 60 : 84);
  if (hidden) {
    say('What does it do, and how would you write it? Say your answer, then press Reveal.', cy, 2, 'dimgray', ITALIC, ts);
    textStyle(NORMAL);
    return;
  }
  say(it[2], cy, 1, 'black', BOLD, ts);
  cy += lh + 2;
  const code = isTemplate ? ["fig.update_layout(template='" + it[1] + "')"] : it[3];
  const codeSize = narrow ? 11 : 14, codeH = code.length * (codeSize + 4) + 8;
  fill('whitesmoke');
  stroke('silver');
  strokeWeight(1);
  rect(ix, cy, iw, codeH, 5);
  code.forEach((ln, i) => {
    noStroke();
    fill('darkslateblue');
    textStyle(BOLD);
    textSize(codeSize);
    textAlign(LEFT, TOP);
    text(ln, ix + 8, cy + 5 + i * (codeSize + 4));
  });
  cy += codeH + 6;
  say(it[4], cy, 3, 'black', NORMAL, ts);
  textStyle(NORMAL);
}

// Clicking an entry selects it.
function mouseOver(b) {
  return mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
}
function mousePressed() {
  const box = rowBoxes.find(mouseOver);
  if (box && box.index !== selected) choose(box.index);
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
