let model;
let predictionImage;

async function preload() {
  model = await tf.loadLayersModel('models/handwritten-digits-models.json');
}

function setup() {
  createCanvas(400, 400);
  background(0);
  strokeWeight(40);
  noFill();
  stroke(255);

  predictionImage = createImage(28, 28);
  predictionImage.loadPixels();

  const clearButton = document.getElementById('clearButton');
  clearButton.addEventListener('click', clearCanvas);
}

function draw() {
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }

  if (model) {
    // Make prediction
    const [prediction, confidence, saliencyMap] = predictDigit();

    // Display prediction
    const predictionElement = document.getElementById('prediction');
    const predictedDigit = prediction.indexOf(Math.max(...prediction));
    predictionElement.textContent = predictedDigit;

    // Display confidence
    const confidenceElement = document.getElementById('confidence');
    confidenceElement.textContent = `${Math.round(confidence * 100)}%`;

    // Update prediction image
    predictionImage.loadPixels();
    for (let i = 0; i < 784; i++) {
      const pixelIndex = i * 4;
      const pixelValue = predictionImage.pixels[i * 4];
      predictionImage.pixels[pixelIndex] = pixelValue;
      predictionImage.pixels[pixelIndex + 1] = pixelValue;
      predictionImage.pixels[pixelIndex + 2] = pixelValue;
      predictionImage.pixels[pixelIndex + 3] = 255;
    }
    predictionImage.updatePixels();

    // Display prediction image
    //image(predictionImage, width + 10, 0, 280, 280);

    // Display saliency map
    //image(saliencyMap, width + 10, 300, 280, 280);
  }
}

function predictDigit() {
  let img = get();
  img.resize(28, 28);
  img.loadPixels();
  let input = [];
  for (let i = 0; i < 784; i++) {
    input[i] = img.pixels[i * 4] / 255.0;
  }
  let xs = tf.tensor2d(input, [1, 784]);
  let reshapedXs = xs.reshape([1, 28, 28, 1]);
  let result = model.predict(reshapedXs);
  let index = result.argMax(1).dataSync()[0];

  // Make prediction
  const [prediction, confidence] = tf.tidy(() => {
    const prediction = model.predict(reshapedXs).dataSync();
    const confidence = Math.max(...prediction);
    return [prediction, confidence];
  });

  // Compute saliency map
  const saliencyMap = tf.tidy(() => {
    const gradient = tf.grad(x => model.predict(reshapedXs))(reshapedXs);
    const absGradient = tf.abs(gradient);
    const maxAbsGradient = tf.max(absGradient, axis=3);
    return maxAbsGradient.reshape([28, 28]);
  });

  return [prediction, confidence, saliencyMap];
}


function keyPressed() {
  if (keyCode === ENTER) {
    //predictDigit();
    clear();
    background(0);
    stroke(255);
  }
}

function clearCanvas() {
  clear();
  background(0);
  stroke(255);
}
