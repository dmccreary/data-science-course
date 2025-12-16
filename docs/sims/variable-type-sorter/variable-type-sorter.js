// Variable Type Sorter MicroSim
// An interactive quiz game for classifying variable types

// Canvas dimensions
let canvasWidth = 400;
let drawHeight = 250;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;

// Game state
let score = 0;
let totalQuestions = 10;
let currentQuestion = null;
let gameState = 'playing'; // 'playing', 'correct', 'wrong', 'complete'
let feedbackTimer = 0;
let feedbackDuration = 180; // frames to show feedback (3 seconds at 60fps)

// Buttons
let continuousBtn, discreteBtn, ordinalBtn, nominalBtn, playAgainBtn;
let buttonY;
let buttonWidth = 90;
let buttonHeight = 35;

// Completion state
let completeStartFrame = 0;
let celebrationDuration = 300; // 5 seconds at 60fps

// Celebration particles
let particles = [];

// Sad face animation
let sadFaceAlpha = 0;
let sadFaceY = 0;

// Question bank
const questions = [
    { variable: 'Height in inches', answer: 'Continuous', hint: 'Can be any decimal value' },
    { variable: 'Number of pets owned', answer: 'Discrete', hint: 'Must be whole numbers' },
    { variable: 'Favorite color', answer: 'Nominal', hint: 'No natural order between colors' },
    { variable: 'Customer satisfaction (Poor, Fair, Good, Excellent)', answer: 'Ordinal', hint: 'Has a meaningful ranking' },
    { variable: 'Temperature in Celsius', answer: 'Continuous', hint: 'Can be 98.6, 98.65, etc.' },
    { variable: 'Number of bedrooms', answer: 'Discrete', hint: 'You can\'t have 2.7 bedrooms' },
    { variable: 'Blood type (A, B, AB, O)', answer: 'Nominal', hint: 'Just different categories' },
    { variable: 'Movie rating (1-5 stars)', answer: 'Ordinal', hint: '5 stars is better than 1' },
    { variable: 'Weight in kilograms', answer: 'Continuous', hint: 'Can be any value with decimals' },
    { variable: 'Number of siblings', answer: 'Discrete', hint: 'Count them: 0, 1, 2, 3...' },
    { variable: 'Country of birth', answer: 'Nominal', hint: 'No ranking between countries' },
    { variable: 'Education level (HS, BS, MS, PhD)', answer: 'Ordinal', hint: 'PhD > MS > BS > HS' },
    { variable: 'Time to complete a race', answer: 'Continuous', hint: 'Measured to milliseconds' },
    { variable: 'Number of cars in parking lot', answer: 'Discrete', hint: 'Whole cars only!' },
    { variable: 'Marital status', answer: 'Nominal', hint: 'Single, Married - no order' },
    { variable: 'Pain level (1-10 scale)', answer: 'Ordinal', hint: '10 is worse than 1' },
    { variable: 'Distance traveled in miles', answer: 'Continuous', hint: 'Can be 5.2378 miles' },
    { variable: 'Number of books read', answer: 'Discrete', hint: 'Count whole books' },
    { variable: 'Zodiac sign', answer: 'Nominal', hint: 'No ranking between signs' },
    { variable: 'Spiciness (Mild, Medium, Hot)', answer: 'Ordinal', hint: 'Hot > Medium > Mild' },
    { variable: 'Body mass index (BMI)', answer: 'Continuous', hint: 'Calculated with decimals' },
    { variable: 'Number of children', answer: 'Discrete', hint: 'Whole people only' },
    { variable: 'Eye color', answer: 'Nominal', hint: 'Blue is not "more" than brown' },
    { variable: 'T-shirt size (S, M, L, XL)', answer: 'Ordinal', hint: 'XL > L > M > S' },
    { variable: 'Annual income in dollars', answer: 'Continuous', hint: 'Can include cents' },
    { variable: 'Number of email subscribers', answer: 'Discrete', hint: 'Count subscribers' },
    { variable: 'Favorite ice cream flavor', answer: 'Nominal', hint: 'Just preferences, no order' },
    { variable: 'Grade (A, B, C, D, F)', answer: 'Ordinal', hint: 'A is better than F' }
];

let usedQuestions = [];

function setup() {
    updateCanvasSize();
    const canvas = createCanvas(canvasWidth, canvasHeight);
    canvas.parent(document.querySelector('main'));

    buttonY = drawHeight + 25;
    createButtons();
    selectNewQuestion();

    describe('Variable Type Sorter quiz game where players classify variables as Continuous, Discrete, Ordinal, or Nominal', LABEL);
}

function createButtons() {
    let spacing = 10;
    let totalWidth = (buttonWidth * 4) + (spacing * 3);
    let startX = (canvasWidth - totalWidth) / 2;

    continuousBtn = createButton('Continuous');
    continuousBtn.position(startX, buttonY);
    continuousBtn.size(buttonWidth, buttonHeight);
    continuousBtn.mousePressed(() => checkAnswer('Continuous'));
    continuousBtn.style('background-color', '#0984e3');
    continuousBtn.style('color', 'white');
    continuousBtn.style('border', 'none');
    continuousBtn.style('border-radius', '6px');
    continuousBtn.style('font-size', '14px');
    continuousBtn.style('cursor', 'pointer');

    discreteBtn = createButton('Discrete');
    discreteBtn.position(startX + buttonWidth + spacing, buttonY);
    discreteBtn.size(buttonWidth, buttonHeight);
    discreteBtn.mousePressed(() => checkAnswer('Discrete'));
    discreteBtn.style('background-color', '#6c5ce7');
    discreteBtn.style('color', 'white');
    discreteBtn.style('border', 'none');
    discreteBtn.style('border-radius', '6px');
    discreteBtn.style('font-size', '14px');
    discreteBtn.style('cursor', 'pointer');

    ordinalBtn = createButton('Ordinal');
    ordinalBtn.position(startX + (buttonWidth + spacing) * 2, buttonY);
    ordinalBtn.size(buttonWidth, buttonHeight);
    ordinalBtn.mousePressed(() => checkAnswer('Ordinal'));
    ordinalBtn.style('background-color', '#00b894');
    ordinalBtn.style('color', 'white');
    ordinalBtn.style('border', 'none');
    ordinalBtn.style('border-radius', '6px');
    ordinalBtn.style('font-size', '14px');
    ordinalBtn.style('cursor', 'pointer');

    nominalBtn = createButton('Nominal');
    nominalBtn.position(startX + (buttonWidth + spacing) * 3, buttonY);
    nominalBtn.size(buttonWidth, buttonHeight);
    nominalBtn.mousePressed(() => checkAnswer('Nominal'));
    nominalBtn.style('background-color', '#e17055');
    nominalBtn.style('color', 'white');
    nominalBtn.style('border', 'none');
    nominalBtn.style('border-radius', '6px');
    nominalBtn.style('font-size', '14px');
    nominalBtn.style('cursor', 'pointer');

    // Play Again button (hidden initially)
    playAgainBtn = createButton('Play Again');
    playAgainBtn.position(canvasWidth / 2 - 60, buttonY);
    playAgainBtn.size(120, buttonHeight);
    playAgainBtn.mousePressed(resetGame);
    playAgainBtn.style('background-color', '#27ae60');
    playAgainBtn.style('color', 'white');
    playAgainBtn.style('border', 'none');
    playAgainBtn.style('border-radius', '6px');
    playAgainBtn.style('font-size', '16px');
    playAgainBtn.style('font-weight', 'bold');
    playAgainBtn.style('cursor', 'pointer');
    playAgainBtn.hide();
}

function repositionButtons() {
    let spacing = 10;
    let totalWidth = (buttonWidth * 4) + (spacing * 3);
    let startX = (canvasWidth - totalWidth) / 2;

    continuousBtn.position(startX, buttonY);
    discreteBtn.position(startX + buttonWidth + spacing, buttonY);
    ordinalBtn.position(startX + (buttonWidth + spacing) * 2, buttonY);
    nominalBtn.position(startX + (buttonWidth + spacing) * 3, buttonY);
    playAgainBtn.position(canvasWidth / 2 - 60, buttonY);
}

function hideAnswerButtons() {
    continuousBtn.hide();
    discreteBtn.hide();
    ordinalBtn.hide();
    nominalBtn.hide();
}

function showAnswerButtons() {
    continuousBtn.show();
    discreteBtn.show();
    ordinalBtn.show();
    nominalBtn.show();
}

function showPlayAgainButton() {
    playAgainBtn.show();
}

function resetGame() {
    score = 0;
    gameState = 'playing';
    usedQuestions = [];
    particles = [];
    completeStartFrame = 0;
    playAgainBtn.hide();
    showAnswerButtons();
    selectNewQuestion();
}

function selectNewQuestion() {
    // If we've used all questions, reset
    if (usedQuestions.length >= questions.length) {
        usedQuestions = [];
    }

    // Find an unused question
    let available = questions.filter((q, i) => !usedQuestions.includes(i));
    let randomIndex = floor(random(available.length));
    let originalIndex = questions.indexOf(available[randomIndex]);
    usedQuestions.push(originalIndex);
    currentQuestion = available[randomIndex];
}

function checkAnswer(answer) {
    if (gameState !== 'playing') return;

    if (answer === currentQuestion.answer) {
        score++;
        gameState = 'correct';
        feedbackTimer = feedbackDuration;
        createCelebration();

        if (score >= totalQuestions) {
            gameState = 'complete';
            completeStartFrame = frameCount;
            hideAnswerButtons();
            showPlayAgainButton();
        }
    } else {
        gameState = 'wrong';
        feedbackTimer = feedbackDuration;
        sadFaceAlpha = 255;
        sadFaceY = drawHeight / 2;
    }
}

function createCelebration() {
    particles = [];
    let colors = ['#FF6B6B', '#FF8E53', '#FFD93D', '#6BCB77', '#4D96FF', '#9B59B6', '#FF6B9D'];

    for (let i = 0; i < 50; i++) {
        particles.push({
            x: canvasWidth / 2,
            y: drawHeight / 2,
            vx: random(-8, 8),
            vy: random(-12, -4),
            size: random(8, 16),
            color: colors[floor(random(colors.length))],
            alpha: 255,
            rotation: random(TWO_PI),
            rotationSpeed: random(-0.2, 0.2),
            gravity: 0.3,
            shape: floor(random(3)) // 0: circle, 1: star, 2: square
        });
    }
}

function updateParticles() {
    for (let i = particles.length - 1; i >= 0; i--) {
        let p = particles[i];
        p.x += p.vx;
        p.vy += p.gravity;
        p.y += p.vy;
        p.rotation += p.rotationSpeed;
        p.alpha -= 4;

        if (p.alpha <= 0 || p.y > drawHeight) {
            particles.splice(i, 1);
        }
    }
}

function drawParticles() {
    for (let p of particles) {
        push();
        translate(p.x, p.y);
        rotate(p.rotation);
        fill(color(p.color + hex(floor(p.alpha), 2).slice(-2)));
        noStroke();

        if (p.shape === 0) {
            ellipse(0, 0, p.size);
        } else if (p.shape === 1) {
            drawStar(0, 0, p.size / 4, p.size / 2, 5);
        } else {
            rectMode(CENTER);
            rect(0, 0, p.size, p.size);
        }
        pop();
    }
}

function drawStar(x, y, radius1, radius2, npoints) {
    let angle = TWO_PI / npoints;
    let halfAngle = angle / 2.0;
    beginShape();
    for (let a = -HALF_PI; a < TWO_PI - HALF_PI; a += angle) {
        let sx = x + cos(a) * radius2;
        let sy = y + sin(a) * radius2;
        vertex(sx, sy);
        sx = x + cos(a + halfAngle) * radius1;
        sy = y + sin(a + halfAngle) * radius1;
        vertex(sx, sy);
    }
    endShape(CLOSE);
}

function draw() {
    updateCanvasSize();

    // Drawing area
    fill('aliceblue');
    stroke('silver');
    strokeWeight(1);
    rect(0, 0, canvasWidth, drawHeight);

    // Control area
    fill('white');
    noStroke();
    rect(0, drawHeight, canvasWidth, controlHeight);

    // Draw score in upper right
    fill('#333');
    noStroke();
    textSize(18);
    textAlign(RIGHT, TOP);
    text('Score: ' + score + '/' + totalQuestions, canvasWidth - margin, margin);

    // Draw title
    textAlign(CENTER, TOP);
    textSize(22);
    fill('#2d3436');
    text('Variable Type Sorter', canvasWidth / 2, margin);

    if (gameState === 'complete') {
        drawCompleteScreen();
    } else {
        drawQuestionBox();
        drawCategoryLabels();

        if (gameState === 'correct') {
            drawCorrectFeedback();
            updateParticles();
            drawParticles();
            feedbackTimer--;
            if (feedbackTimer <= 0 && score < totalQuestions) {
                gameState = 'playing';
                selectNewQuestion();
            }
        } else if (gameState === 'wrong') {
            drawWrongFeedback();
            feedbackTimer--;
            if (feedbackTimer <= 0) {
                gameState = 'playing';
            }
        }
    }
}

function drawQuestionBox() {
    // Question box
    let boxWidth = canvasWidth - 60;
    let boxHeight = 100;
    let boxX = 30;
    let boxY = 70;

    // Box shadow
    fill(200, 200, 200, 100);
    noStroke();
    rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 10);

    // Box
    fill(255);
    stroke('#ddd');
    strokeWeight(2);
    rect(boxX, boxY, boxWidth, boxHeight, 10);

    // Question text
    fill('#2d3436');
    noStroke();
    textSize(14);
    textAlign(CENTER, TOP);
    text('What type of variable is:', canvasWidth / 2, boxY + 15);

    textSize(20);
    textAlign(CENTER, CENTER);
    textWrap(WORD);
    text(currentQuestion.variable, boxX + 15, boxY + boxHeight / 2 + 10, boxWidth - 30);
}

function drawCategoryLabels() {
    // Category explanations
    let labelY = 200;
    textSize(12);
    textAlign(CENTER, TOP);

    fill('#0984e3');
    text('Numerical\n(decimals OK)', canvasWidth * 0.15, labelY);

    fill('#6c5ce7');
    text('Numerical\n(whole numbers)', canvasWidth * 0.38, labelY);

    fill('#00b894');
    text('Categorical\n(with order)', canvasWidth * 0.62, labelY);

    fill('#e17055');
    text('Categorical\n(no order)', canvasWidth * 0.85, labelY);

    // Instructions
    fill('#636e72');
    textSize(14);
    textAlign(CENTER, TOP);
    text('Click the correct variable type below:', canvasWidth / 2, drawHeight + 8);
}

function drawCorrectFeedback() {
    // White rectangle background
    let boxWidth = 280;
    let boxHeight = 120;
    let boxX = (canvasWidth - boxWidth) / 2;
    let boxY = (drawHeight - boxHeight) / 2 + 20;

    // Shadow
    fill(0, 0, 0, 30);
    noStroke();
    rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 12);

    // White background with green border
    fill(255);
    stroke('#27ae60');
    strokeWeight(3);
    rect(boxX, boxY, boxWidth, boxHeight, 12);

    // Correct text
    fill('#27ae60');
    noStroke();
    textSize(32);
    textAlign(CENTER, CENTER);
    text('Correct!', canvasWidth / 2, boxY + 40);

    textSize(14);
    fill('#2d3436');
    textWrap(WORD);
    text(currentQuestion.hint, boxX + 15, boxY + 80, boxWidth - 30);
}

function drawWrongFeedback() {
    // White rectangle background
    let boxWidth = 280;
    let boxHeight = 160;
    let boxX = (canvasWidth - boxWidth) / 2;
    let boxY = (drawHeight - boxHeight) / 2 - 10;

    // Shadow
    fill(0, 0, 0, 30);
    noStroke();
    rect(boxX + 4, boxY + 4, boxWidth, boxHeight, 12);

    // White background with red border
    fill(255);
    stroke('#e74c3c');
    strokeWeight(3);
    rect(boxX, boxY, boxWidth, boxHeight, 12);

    // Sad face
    let faceX = canvasWidth / 2;
    let faceY = boxY + 45;
    let faceSize = 50;

    // Face circle
    fill('#f8d7da');
    stroke('#e74c3c');
    strokeWeight(2);
    ellipse(faceX, faceY, faceSize);

    // Eyes
    fill('#e74c3c');
    noStroke();
    ellipse(faceX - 10, faceY - 6, 6, 6);
    ellipse(faceX + 10, faceY - 6, 6, 6);

    // Sad mouth
    noFill();
    stroke('#e74c3c');
    strokeWeight(2);
    arc(faceX, faceY + 12, 20, 12, PI, TWO_PI);

    // Try again text
    fill('#c0392b');
    noStroke();
    textSize(22);
    textAlign(CENTER, CENTER);
    text('Try Again!', canvasWidth / 2, boxY + 95);

    textSize(13);
    fill('#636e72');
    textWrap(WORD);
    text('Hint: ' + currentQuestion.hint, boxX + 15, boxY + 125, boxWidth - 30);
}

function drawCompleteScreen() {
    // Calculate elapsed time and whether animations should still run
    let elapsedFrames = frameCount - completeStartFrame;
    let animationsActive = elapsedFrames < celebrationDuration;

    // Use frozen frame for animations after 5 seconds
    let animFrame = animationsActive ? frameCount : (completeStartFrame + celebrationDuration);

    // Victory background
    let gradient = drawingContext;
    let grd = gradient.createLinearGradient(0, 0, 0, drawHeight);
    grd.addColorStop(0, '#667eea');
    grd.addColorStop(1, '#764ba2');
    gradient.fillStyle = grd;
    gradient.fillRect(0, 0, canvasWidth, drawHeight);

    // Stars decoration (animated only during first 5 seconds)
    fill(255, 255, 255, 100);
    noStroke();
    for (let i = 0; i < 20; i++) {
        let sx = (animFrame * 0.5 + i * 50) % canvasWidth;
        let sy = (sin(animFrame * 0.02 + i) * 50) + 100 + i * 15;
        drawStar(sx, sy % drawHeight, 3, 6, 5);
    }

    // Trophy icon
    fill('#FFD700');
    stroke('#DAA520');
    strokeWeight(3);
    let trophyX = canvasWidth / 2;
    let trophyY = 100;

    // Trophy cup
    beginShape();
    vertex(trophyX - 30, trophyY - 20);
    vertex(trophyX - 25, trophyY + 20);
    vertex(trophyX + 25, trophyY + 20);
    vertex(trophyX + 30, trophyY - 20);
    endShape(CLOSE);

    // Trophy base
    rect(trophyX - 15, trophyY + 20, 30, 10, 2);
    rect(trophyX - 20, trophyY + 30, 40, 8, 2);

    // Handles
    noFill();
    stroke('#DAA520');
    strokeWeight(4);
    arc(trophyX - 30, trophyY, 15, 25, HALF_PI, PI + HALF_PI);
    arc(trophyX + 30, trophyY, 15, 25, -HALF_PI, HALF_PI);

    // Main text
    fill('white');
    noStroke();
    textSize(28);
    textAlign(CENTER, CENTER);
    text('Side Quest Complete!', canvasWidth / 2, 180);

    textSize(18);
    text('Data Scientist Health Points', canvasWidth / 2, 230);

    // Health bar
    let barWidth = 200;
    let barHeight = 25;
    let barX = (canvasWidth - barWidth) / 2;
    let barY = 260;

    // Bar background
    fill(255, 255, 255, 50);
    rect(barX, barY, barWidth, barHeight, 5);

    // Bar fill (animated only during first 5 seconds, then stays full)
    let fillWidth = animationsActive ? map(sin(animFrame * 0.1), -1, 1, barWidth * 0.8, barWidth) : barWidth;
    fill('#27ae60');
    rect(barX, barY, fillWidth, barHeight, 5);

    // HP text
    fill('white');
    textSize(14);
    text('+100 HP', canvasWidth / 2, barY + barHeight / 2);

    // Achievement unlocked
    fill('#FFD700');
    textSize(16);
    text('Achievement: Variable Master', canvasWidth / 2, 320);

    // Celebration particles - only during first 5 seconds
    if (animationsActive) {
        if (frameCount % 10 === 0) {
            createCelebration();
        }
        updateParticles();
        drawParticles();
    }
}

function windowResized() {
    updateCanvasSize();
    resizeCanvas(canvasWidth, canvasHeight);
    repositionButtons();
}

function updateCanvasSize() {
    const container = document.querySelector('main');
    if (container) {
        canvasWidth = container.offsetWidth;
    }
}
