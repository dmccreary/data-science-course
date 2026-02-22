// Virtual Environment Isolation MicroSim
// Shows how virtual environments isolate packages from each other
// Bloom Level: Apply (L3) - interactive exploration
// MicroSim template version 2026.02

let canvasWidth = 580;
let drawHeight = 400;
let controlHeight = 70;
let canvasHeight = drawHeight + controlHeight;
let containerWidth;
let containerHeight = canvasHeight;

let margin = 20;
let defaultTextSize = 14;

// Environments
let envs = [
  {
    name: 'project-a',
    color: [60, 130, 200],
    packages: [{ name: 'pandas', version: '1.5' }, { name: 'numpy', version: '1.23' }],
    x: 0, y: 0, w: 0, h: 0
  },
  {
    name: 'project-b',
    color: [50, 160, 80],
    packages: [{ name: 'pandas', version: '2.1' }, { name: 'matplotlib', version: '3.8' }],
    x: 0, y: 0, w: 0, h: 0
  }
];

let selectedEnv = 0;
let statusMessage = '';
let statusColor = [40, 40, 40];
let addPkgButton, env1Button, env2Button, resetButton;

let packageOptions = [
  { name: 'pandas', version: '1.5' },
  { name: 'pandas', version: '2.1' },
  { name: 'numpy', version: '1.23' },
  { name: 'matplotlib', version: '3.8' },
  { name: 'scikit-learn', version: '1.3' },
  { name: 'seaborn', version: '0.12' }
];
let pkgIdx = 0;
let pkgSelect;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  const mainElement = document.querySelector('main');
  canvas.parent(mainElement);
  textSize(defaultTextSize);

  // Controls row
  let ctrlY = drawHeight + 8;

  env1Button = createButton('Select: project-a');
  env1Button.position(margin, ctrlY);
  env1Button.mousePressed(() => { selectedEnv = 0; updateStatus('Selected environment: project-a', [60, 130, 200]); });

  env2Button = createButton('Select: project-b');
  env2Button.position(margin + 145, ctrlY);
  env2Button.mousePressed(() => { selectedEnv = 1; updateStatus('Selected environment: project-b', [50, 160, 80]); });

  pkgSelect = createSelect();
  pkgSelect.position(margin + 295, ctrlY);
  for (let p of packageOptions) {
    pkgSelect.option(p.name + ' v' + p.version);
  }

  addPkgButton = createButton('Install Package');
  addPkgButton.position(margin + 480, ctrlY);
  addPkgButton.mousePressed(installPackage);

  resetButton = createButton('Reset');
  resetButton.position(margin + 590, ctrlY);
  resetButton.mousePressed(resetEnvs);

  updateStatus('Each environment is completely isolated — try installing the same package with different versions!', [40, 40, 40]);

  describe('Interactive virtual environment isolation simulator. Select an environment and install packages to see how isolation prevents version conflicts between project-a and project-b.', LABEL);
}

function installPackage() {
  let selText = pkgSelect.value();
  let parts = selText.split(' v');
  let pkgName = parts[0];
  let pkgVer = parts[1];
  let env = envs[selectedEnv];

  // Check for same package different version conflict
  let existing = env.packages.find(p => p.name === pkgName);
  if (existing) {
    if (existing.version === pkgVer) {
      updateStatus('⚠ ' + pkgName + ' v' + pkgVer + ' is already installed in ' + env.name, [180, 120, 0]);
    } else {
      updateStatus('⚠ Conflict! ' + pkgName + ' v' + existing.version + ' already installed. Overwriting with v' + pkgVer, [200, 60, 60]);
      existing.version = pkgVer;
    }
    return;
  }

  if (env.packages.length >= 6) {
    updateStatus('Environment ' + env.name + ' is full for this demo.', [150, 80, 0]);
    return;
  }

  env.packages.push({ name: pkgName, version: pkgVer });
  updateStatus('✓ Installed ' + pkgName + ' v' + pkgVer + ' in ' + env.name + ' only — other envs not affected!', [40, 140, 60]);
}

function resetEnvs() {
  envs[0].packages = [{ name: 'pandas', version: '1.5' }, { name: 'numpy', version: '1.23' }];
  envs[1].packages = [{ name: 'pandas', version: '2.1' }, { name: 'matplotlib', version: '3.8' }];
  updateStatus('Reset to default state. Notice: both environments have pandas, but DIFFERENT versions!', [40, 40, 40]);
}

function updateStatus(msg, col) {
  statusMessage = msg;
  statusColor = col;
}

function draw() {
  updateCanvasSize();
  background(240);

  // Drawing region
  fill(240);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);

  // Control region
  fill(255);
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Title
  noStroke();
  fill(30);
  textAlign(CENTER, TOP);
  textSize(17);
  text('Virtual Environment Isolation', canvasWidth / 2, margin - 4);

  // Base system platform
  let platformH = 38;
  let platformY = drawHeight - margin - platformH;
  fill(90, 90, 100);
  stroke(60);
  strokeWeight(1);
  rect(margin, platformY, canvasWidth - margin * 2, platformH, 6);
  noStroke();
  fill(255);
  textAlign(CENTER, CENTER);
  textSize(13);
  text('Base System: Python 3.11 Installation', canvasWidth / 2, platformY + platformH / 2);

  // Compute env boxes
  let envAreaH = platformY - margin - 50;
  let envW = (canvasWidth - margin * 3) / 2;
  let envH = envAreaH;

  for (let i = 0; i < envs.length; i++) {
    let env = envs[i];
    let ex = margin + i * (envW + margin);
    let ey = margin + 30;

    env.x = ex; env.y = ey; env.w = envW; env.h = envH;

    let c = env.color;
    let isSelected = (i === selectedEnv);

    // Bubble background
    fill(c[0], c[1], c[2], 40);
    if (isSelected) {
      stroke(c[0], c[1], c[2]);
      strokeWeight(3);
    } else {
      stroke(c[0], c[1], c[2], 120);
      strokeWeight(1.5);
    }
    rect(ex, ey, envW, envH, 14);

    // Env header
    fill(c[0], c[1], c[2]);
    noStroke();
    rect(ex, ey, envW, 28, 14, 14, 0, 0);
    fill(255);
    textAlign(CENTER, CENTER);
    textSize(13);
    text((isSelected ? '▶ ' : '') + env.name, ex + envW / 2, ey + 14);

    // Packages
    let pkgStartY = ey + 38;
    let pkgH = 30;
    let pkgW = envW - 20;
    let pkgX = ex + 10;

    for (let j = 0; j < env.packages.length; j++) {
      let pkg = env.packages[j];
      let py = pkgStartY + j * (pkgH + 4);

      // Check if this package has version conflicts across envs
      let otherEnv = envs[1 - i];
      let conflict = otherEnv.packages.find(p => p.name === pkg.name && p.version !== pkg.version);
      let sameVer = otherEnv.packages.find(p => p.name === pkg.name && p.version === pkg.version);

      if (conflict) {
        fill(80, 180, 80); // green = different version, isolated OK
      } else if (sameVer) {
        fill(100, 160, 240); // blue = same version
      } else {
        fill(200, 200, 210); // gray = unique to this env
      }
      stroke(160);
      strokeWeight(1);
      rect(pkgX, py, pkgW, pkgH - 4, 5);

      noStroke();
      fill(30);
      textAlign(LEFT, CENTER);
      textSize(12);
      text(pkg.name, pkgX + 8, py + (pkgH - 4) / 2);

      textAlign(RIGHT, CENTER);
      fill(80);
      text('v' + pkg.version, pkgX + pkgW - 8, py + (pkgH - 4) / 2);

      if (conflict) {
        textAlign(CENTER, CENTER);
        fill(0, 130, 50);
        textSize(10);
        text('✓ isolated', pkgX + pkgW / 2, py + (pkgH - 4) / 2);
      }
    }
  }

  // Legend
  let lx = margin, ly = drawHeight - margin - platformH - 28;
  noStroke();
  fill(80, 180, 80);
  rect(lx, ly, 12, 12, 2);
  fill(60);
  textSize(11);
  textAlign(LEFT, CENTER);
  text('Same package, different versions (isolated!)', lx + 16, ly + 6);

  fill(100, 160, 240);
  rect(lx + 240, ly, 12, 12, 2);
  fill(60);
  text('Same package, same version', lx + 256, ly + 6);

  // Status message
  let msgColor = color(statusColor[0], statusColor[1], statusColor[2]);
  fill(msgColor);
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(12);
  textWrap(WORD);
  text(statusMessage, margin, drawHeight + 36, canvasWidth - margin * 2 - 100, 28);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  // Reposition buttons
  let ctrlY = drawHeight + 8;
  env1Button.position(margin, ctrlY);
  env2Button.position(margin + 145, ctrlY);
  pkgSelect.position(margin + 295, ctrlY);
  addPkgButton.position(margin + 480, ctrlY);
  resetButton.position(margin + 590, ctrlY);
  redraw();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}
