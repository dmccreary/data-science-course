// Duplicate Handling Decision Tree
// CANVAS_HEIGHT: 625
// Bloom L4 (Analyze): students examine a small table that contains duplicates, work out which
// branch of the decision tree it belongs to, and click the strategy at the end of that branch.
// The strategy is applied to the table, so different strategies can be compared on the same rows.
//
// Model: each strategy is carried out on the case table the way pandas does it.
//   keep         nothing is removed
//   exact        df.drop_duplicates()                    drops a row equal to an earlier row
//   subset/first df.drop_duplicates(subset=keys)         keeps the first row for each key
//                                                        (keep="first" is the default)
//   last         df.drop_duplicates(subset=keys, keep="last")
//   merge        df.groupby(keys, as_index=False).first()  one row per key, sorted by key,
//                                                        with the first non-missing value of each column
//   review       df.duplicated(subset=keys, keep=False)  flags every copy and removes nothing

let containerWidth;
let canvasWidth = 400;
let drawHeight = 580;
let controlHeight = 45;
let canvasHeight = drawHeight + controlHeight;
let containerHeight = canvasHeight;
let margin = 15;
let defaultTextSize = 16;

// fill and outline for each kind of box
const KINDS = { ask: ['lightblue', 'royalblue'], check: ['khaki', 'goldenrod'],
  safe: ['palegreen', 'seagreen'], care: ['lightpink', 'crimson'] };
// The questions run down the left side. side: the strategy reached by the sideways answer.
const SPINE = [
  { kind: 'ask', text: 'Duplicates detected. Is the duplication intentional?', side: 'keep', sideLabel: 'Yes', down: 'No' },
  { kind: 'ask', text: 'Are the rows exactly identical?', side: 'exact', sideLabel: 'Yes', down: 'No' },
  { kind: 'check', text: 'Which columns identify a record? Choose the key columns.' },
  { kind: 'ask', text: 'Do the other columns differ in a way that matters?', side: 'subset', sideLabel: 'No', down: 'Yes' },
  { kind: 'ask', text: 'Which version is correct?' }
];
// depth: the last question on the way to the strategy
const LEAVES = {
  keep: { kind: 'care', top: 'Intentional', label: "Don't remove!", depth: 0 },
  exact: { kind: 'safe', top: 'Safe to drop', label: 'drop_duplicates()', depth: 1 },
  subset: { kind: 'safe', top: 'Drop by key', label: 'subset=keys', depth: 3 },
  first: { kind: 'care', top: 'First is right', label: "keep='first'", depth: 4 },
  last: { kind: 'care', top: 'Last is right', label: "keep='last'", depth: 4 },
  merge: { kind: 'check', top: 'Need to merge', label: 'groupby().agg()', depth: 4 },
  review: { kind: 'check', top: "Can't tell", label: 'Manual review', depth: 4 }
};
const BOTTOM = ['first', 'last', 'merge', 'review'];

// key: positions of the columns that identify a record. null is a missing value.
const CASES = [
  { name: 'A. File loaded twice', cols: ['email', 'name', 'age'], key: [0], best: 'exact',
    rows: [['a@x.com', 'Alice', 25], ['b@x.com', 'Bob', 30], ['a@x.com', 'Alice', 25]],
    story: 'A customer list. The same file was loaded two times by mistake.',
    hint: 'Compare rows 0 and 2 value by value.',
    why: 'Rows 0 and 2 match in every column, and a customer belongs in the list once. An exact copy adds nothing, so it is safe to drop.' },
  { name: 'B. Sales log', cols: ['customer', 'item', 'price'], key: [0, 1], best: 'keep',
    rows: [['Ana', 'pen', 2.5], ['Ana', 'pen', 2.5], ['Ben', 'ink', 9.5]],
    story: 'A store log. One row is written each time an item is scanned at the till.',
    hint: 'What does one row of this table record?',
    why: 'Each row is one item sold. Ana bought two pens, so both rows are real. Dropping one would lose a sale.' },
  { name: 'C. Two source files', cols: ['email', 'name', 'file'], key: [0], best: 'subset',
    rows: [['a@x.com', 'Alice', 'jan.csv'], ['b@x.com', 'Bob', 'jan.csv'], ['a@x.com', 'Alice', 'feb.csv']],
    story: 'Member lists from two months were stacked. The file column records where each row came from.',
    hint: 'Which column keeps rows 0 and 2 from matching? Does it describe the member?',
    why: 'Only the file column differs, and it says nothing about Alice. The email identifies a member, so drop on that key.' },
  { name: 'D. Updated profile', cols: ['email', 'city', 'updated'], key: [0], best: 'last',
    rows: [['a@x.com', 'Austin', '2023-01'], ['b@x.com', 'Boston', '2023-02'], ['a@x.com', 'Denver', '2024-06']],
    story: 'Profile records, sorted from oldest to newest. Members can change their city.',
    hint: 'The two rows for a@x.com disagree about the city. Which one is current?',
    why: 'The rows disagree about the city. The table is sorted by date, so the last row for each email is the current one.' },
  { name: 'E. Repeat sign-up', cols: ['email', 'name', 'joined'], key: [0], best: 'first',
    rows: [['a@x.com', 'Alice', '2024-01'], ['b@x.com', 'Bob', '2024-01'], ['a@x.com', 'Alice', '2024-03']],
    story: 'Sign-up records in time order. You need the month each member first joined.',
    hint: 'The joined values differ. Which row answers the question being asked?',
    why: 'The joined months differ, and the question asks when each member first joined. The earliest row comes first.' },
  { name: 'F. Two partial records', cols: ['email', 'phone', 'city'], key: [0], best: 'merge',
    rows: [['a@x.com', null, 'Austin'], ['a@x.com', '555-0101', null], ['b@x.com', '555-0102', 'Boston']],
    story: 'Two systems each stored part of a contact record.',
    hint: 'Look at what each row for a@x.com knows that the other does not.',
    why: 'Each row for a@x.com holds a value the other lacks. Keeping either row loses data, so combine them into one.' },
  { name: 'G. Conflicting ages', cols: ['id', 'name', 'age'], key: [0], best: 'review',
    rows: [[17, 'Alice', 25], [17, 'Alice', 52], [18, 'Bob', 30]],
    story: 'Two rows share an id but give different ages. Nothing records which was typed correctly.',
    hint: 'Is there anything in the table that says which age is right?',
    why: 'One age is a typing error, but the data cannot say which. Flag both rows and ask the source.' }
];

let caseSelect, pathCheckbox;
let caseIndex = 0;
let pick = null;                  // the strategy the student clicked
let leafBoxes = [];               // clickable strategies: { x, y, w, h, id }

// Carry out a strategy on a case table. Returns rows of { id, v, flag } where id is the index label.
function applyStrategy(op, cs) {
  const rows = cs.rows.map((v, i) => ({ id: i, v: v.slice(), flag: false }));
  const keyOf = (r, cols) => JSON.stringify(cols.map(c => r.v[c]));
  const dedupe = (cols, keepLast) => {
    const seen = new Set(), order = keepLast ? rows.slice().reverse() : rows;
    const kept = order.filter(r => !seen.has(keyOf(r, cols)) && seen.add(keyOf(r, cols)));
    return keepLast ? kept.reverse() : kept;
  };
  if (op === 'keep') return rows;
  if (op === 'exact') return dedupe(cs.cols.map((name, c) => c), false);
  if (op === 'subset' || op === 'first') return dedupe(cs.key, false);
  if (op === 'last') return dedupe(cs.key, true);
  const groups = {};
  rows.forEach(r => { const k = keyOf(r, cs.key); (groups[k] = groups[k] || []).push(r); });
  if (op === 'review') return rows.map(r => Object.assign(r, { flag: groups[keyOf(r, cs.key)].length > 1 }));
  // merge: groupby sorts the keys and gives the result a new index 0, 1, 2, ...
  return Object.keys(groups).sort().map((k, i) => ({ id: i, flag: false,
    v: cs.cols.map((name, c) => { const hit = groups[k].find(r => r.v[c] !== null); return hit ? hit.v[c] : null; }) }));
}

// The pandas code for a strategy, written with the key columns of the case
function codeFor(op, cs) {
  const keys = cs.key.map(c => '"' + cs.cols[c] + '"').join(', ');
  return { keep: 'df  (left as it is)', exact: 'df.drop_duplicates()',
    subset: 'df.drop_duplicates(subset=[' + keys + '])',
    first: 'df.drop_duplicates(subset=[' + keys + '], keep="first")',
    last: 'df.drop_duplicates(subset=[' + keys + '], keep="last")',
    merge: 'df.groupby(' + (cs.key.length > 1 ? '[' + keys + ']' : keys) + ', as_index=False).first()',
    review: 'df.duplicated(subset=[' + keys + '], keep=False)' }[op];
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
    pathCheckbox.checked(false);
  });

  // case A opens with its path shown, as a worked example
  pathCheckbox = createCheckbox(' Show the path', true);
  pathCheckbox.parent(mainElement);
  pathCheckbox.position(215, drawHeight + 11);
  pathCheckbox.style('font-size', '16px');

  describe('A decision tree for handling duplicate rows. Four questions and one step lead to seven strategies: do not remove, ' +
    'drop_duplicates, drop by key columns, keep first, keep last, merge with groupby, and manual review. A menu ' +
    'chooses one of seven small example tables. Clicking a strategy applies it to the table and shows the result ' +
    'with feedback. A checkbox highlights the path through the tree that fits the case.', LABEL);
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
  const cs = CASES[caseIndex];
  textWrap(WORD);

  noStroke();
  fill('black');
  textStyle(NORMAL);
  textAlign(CENTER, TOP);
  textSize(narrow ? 18 : 24);
  text('Duplicate Handling Decision Tree', canvasWidth / 2, 8);

  // tree on the left and the case on the right, or stacked when narrow
  const top = narrow ? 32 : 44, fullW = canvasWidth - 2 * margin, bottom = drawHeight - 8;
  if (narrow) {
    const caseH = 212;
    drawTree(margin, top, fullW, bottom - top - caseH - 6, narrow, cs);
    drawCase(margin, bottom - caseH, fullW, caseH, narrow, cs);
  } else {
    const treeW = fullW * 0.57;
    drawTree(margin, top, treeW, bottom - top, narrow, cs);
    drawCase(margin + treeW + 12, top, fullW - treeW - 12, bottom - top, narrow, cs);
  }
  cursor(leafBoxes.some(mouseOver) ? HAND : ARROW);
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

function drawBox(x, y, w, h, kind, emphasis) {
  fill(KINDS[kind][0]);
  stroke(emphasis || KINDS[kind][1]);
  strokeWeight(emphasis ? 3.5 : 1.3);
  rect(x, y, w, h, 8);
}

// A strategy box: plain-language answer on top, the pandas call below
function drawLeaf(id, x, y, w, h, narrow, cs) {
  const leaf = LEAVES[id];
  const best = pathCheckbox.checked() && cs.best === id;
  drawBox(x, y, w, h, leaf.kind, pick === id ? 'black' : best ? 'darkgreen' : null);
  noStroke();
  fill('black');
  textAlign(CENTER, CENTER);
  textStyle(NORMAL);
  textSize(narrow ? 11 : 13);
  text(leaf.top, x + w / 2, y + h * 0.3);
  textStyle(BOLD);
  textSize(narrow ? 11 : 13);
  text(leaf.label, x + w / 2, y + h * 0.7);
  leafBoxes.push({ x, y, w, h, id });
}

function drawTree(x, y, w, h, narrow, cs) {
  const nodeH = narrow ? 32 : 48, leafH = narrow ? 34 : 50, legendH = narrow ? 16 : 22;
  const gap = (h - legendH - leafH - SPINE.length * nodeH) / SPINE.length;
  const spineW = w * 0.54, leafW = w * 0.35, leafX = x + w - leafW, mid = x + spineW / 2;
  const ts = narrow ? 11 : 14;
  const showPath = pathCheckbox.checked();
  const depth = showPath ? LEAVES[cs.best].depth : -1;       // questions 0..depth lie on the path
  const nodeY = i => y + i * (nodeH + gap);
  leafBoxes = [];

  // connectors between questions and out to the side strategies
  SPINE.forEach((node, i) => {
    const cy = nodeY(i) + nodeH / 2;
    if (i < SPINE.length - 1) connector(mid, nodeY(i) + nodeH, mid, nodeY(i + 1), i < depth, node.down, ts);
    if (node.side) connector(x + spineW, cy, leafX, cy, showPath && cs.best === node.side, node.sideLabel, ts);
  });
  // the last question fans out to four strategies along the bottom
  const leafGap = narrow ? 6 : 10, bw = (w - 3 * leafGap) / 4;
  const lastBottom = nodeY(SPINE.length - 1) + nodeH, by = lastBottom + gap, busY = lastBottom + gap / 2;
  const centers = BOTTOM.map((id, k) => x + k * (bw + leafGap) + bw / 2);
  stroke('gray');
  strokeWeight(1.5);
  line(mid, lastBottom, mid, busY);
  line(centers[0], busY, centers[3], busY);
  BOTTOM.forEach((id, k) => {
    const on = showPath && cs.best === id;
    if (on) {
      stroke('darkgreen');
      strokeWeight(3.5);
      line(mid, lastBottom, mid, busY);
      line(mid, busY, centers[k], busY);
    }
    connector(centers[k], busY, centers[k], by, on, '', ts);
  });

  // question boxes
  SPINE.forEach((node, i) => {
    drawBox(x, nodeY(i), spineW, nodeH, node.kind, i <= depth ? 'darkgreen' : null);
    noStroke();
    fill('black');
    textStyle(NORMAL);
    textSize(ts);
    textLeading(ts + 2);
    textAlign(CENTER, CENTER);
    text(node.text, x + 6, nodeY(i), spineW - 12, nodeH);
    if (node.side) drawLeaf(node.side, leafX, nodeY(i), leafW, nodeH, narrow, cs);
  });
  BOTTOM.forEach((id, k) => drawLeaf(id, x + k * (bw + leafGap), by, bw, leafH, narrow, cs));

  // color key
  const items = [['ask', 'question'], ['safe', 'safe action'], ['check', 'investigate'], ['care', 'be careful']];
  let kx = x;
  const ky = y + h - legendH / 2 + 3, s = narrow ? 10 : 12;
  textStyle(NORMAL);
  textSize(narrow ? 11 : 13);
  textAlign(LEFT, CENTER);
  for (const [kind, label] of items) {
    fill(KINDS[kind][0]);
    stroke(KINDS[kind][1]);
    strokeWeight(1);
    rect(kx, ky - s / 2, s, s, 2);
    noStroke();
    fill('black');
    text(label, kx + s + 4, ky + 1);
    kx += s + 4 + textWidth(label) + (narrow ? 10 : 16);
  }
}

// A small table with index labels. Flagged rows are tinted.
function drawSmallTable(x, y, w, rowH, cols, rows, ts) {
  const idxW = ts + 8;
  textSize(ts);
  textStyle(BOLD);
  const show = v => v === null ? 'NaN' : String(v);
  const natural = cols.map((name, c) => max([textWidth(name)].concat(rows.map(r => textWidth(show(r.v[c]))))) + 12);
  const scale = (w - idxW) / natural.reduce((a, b) => a + b, 0);
  noStroke();
  fill('white');
  rect(x, y, w, rowH * (rows.length + 1));
  fill('gainsboro');
  rect(x, y, w, rowH);
  textAlign(LEFT, CENTER);
  for (let k = -1; k < rows.length; k++) {
    const cy = y + (k + 1.5) * rowH + 1;
    if (k >= 0 && rows[k].flag) {
      fill(255, 215, 0, 110);
      rect(x, y + (k + 1) * rowH, w, rowH);
    }
    let cx = x + idxW;
    fill('dimgray');
    textStyle(BOLD);
    if (k >= 0) text(rows[k].id, x + 5, cy);
    cols.forEach((name, c) => {
      const v = k < 0 ? name : rows[k].v[c];
      fill(v === null ? 'firebrick' : 'black');
      textStyle(k < 0 ? BOLD : NORMAL);
      text(k < 0 ? name : show(v), cx + 5, cy);
      cx += natural[c] * scale;
    });
  }
  noFill();
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, rowH * (rows.length + 1));
  textStyle(NORMAL);
}

// The case: its story, the table before, the table after the chosen strategy, and feedback
function drawCase(x, y, w, h, narrow, cs) {
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(x, y, w, h, 10);
  const pad = narrow ? 8 : 12, ts = narrow ? 11 : 14, lh = ts + (narrow ? 2 : 5), iw = w - 2 * pad;
  const rowH = narrow ? 15 : 23, tableH = rowH * (cs.rows.length + 1), ix = x + pad;
  const shown = pick || (pathCheckbox.checked() ? cs.best : null);
  const before = applyStrategy('keep', cs);
  const label = (str, lx, ly, lw, lines, col, style) => {
    noStroke();
    fill(col);
    textStyle(style);
    textSize(ts);
    textLeading(lh);
    textAlign(LEFT, TOP);
    text(str, lx, ly, lw, lines * lh + 3);
  };

  textSize(ts + 1);
  label(cs.name, ix, y + pad, iw, 1, 'black', BOLD);
  const storyLines = narrow ? 2 : 3;
  label(cs.story, ix, y + pad + lh + 2, iw, storyLines, 'black', NORMAL);

  // narrow: tables side by side, then the code. wide: before table, code, after table.
  const tw = narrow ? (iw - 10) / 2 : iw;
  const beforeY = y + pad + lh + storyLines * lh + (narrow ? 6 : 10);
  const afterX = narrow ? ix + tw + 10 : ix;
  const afterY = narrow ? beforeY : beforeY + lh + tableH + 16;
  label('Before: df', ix, beforeY, tw, 1, 'black', BOLD);
  drawSmallTable(ix, beforeY + lh + 2, tw, rowH, cs.cols, before, ts);
  if (!shown) {
    label('Always examine duplicates before removing them. Decide which branch of the tree fits these rows, then ' +
      'click the strategy at its end.', afterX, afterY + (narrow ? lh : 0), tw, 6, 'dimgray', ITALIC);
    return;
  }
  const after = applyStrategy(shown, cs), code = codeFor(shown, cs);
  const tableY = narrow ? afterY + lh + 2 : afterY + 3 * lh + 2;
  const codeY = narrow ? tableY + tableH + 4 : afterY + lh;
  label('After:', afterX, afterY, tw, 1, 'black', BOLD);
  label(code, ix, codeY, iw, 2, 'midnightblue', BOLD);
  drawSmallTable(afterX, tableY, tw, rowH, cs.cols, after, ts);

  // the result in numbers, then the judgment on the choice
  const flagged = after.filter(r => r.flag).length;
  const result = shown === 'review' ? flagged + ' rows flagged for review, none removed.'
    : before.length + ' rows before, ' + after.length + ' rows after.';
  let verdict = 'Why this path: ' + cs.why, verdictColor = 'darkgreen';
  if (pick === cs.best) {
    verdict = 'Good choice. ' + cs.why;
  } else if (pick) {
    const same = JSON.stringify(after) === JSON.stringify(applyStrategy(cs.best, cs));
    verdict = same ? 'Same table as the best answer here, but the tree reaches it by a different question. ' + cs.hint
      : 'Not the best fit for this case. ' + cs.hint;
    verdictColor = same ? 'chocolate' : 'firebrick';
  }
  const feedbackY = narrow ? codeY + 2 * lh + 2 : tableY + tableH + 12;
  if (narrow) {
    label(result + ' ' + verdict, ix, feedbackY, iw, 3, verdictColor, NORMAL);
  } else {
    label(result, ix, feedbackY, iw, 1, 'black', NORMAL);
    label(verdict, ix, feedbackY + lh + 4, iw, 6, verdictColor, NORMAL);
  }
}

// Clicking a strategy applies it. Clicking it again clears the choice.
const mouseOver = b => mouseX >= b.x && mouseX <= b.x + b.w && mouseY >= b.y && mouseY <= b.y + b.h;
function mousePressed() {
  const box = leafBoxes.find(mouseOver);
  if (box) pick = pick === box.id ? null : box.id;
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
