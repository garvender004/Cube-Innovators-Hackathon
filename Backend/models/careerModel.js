const tf = require("@tensorflow/tfjs-node");

// Simulated dataset (Skills & Interests → Career)
const trainingData = [
  { input: [1, 0, 1, 0, 1, 0, 0], output: [1, 0, 0] }, // Web Development
  { input: [0, 1, 0, 1, 0, 1, 0], output: [0, 1, 0] }, // Data Science
  { input: [0, 0, 0, 0, 1, 1, 1], output: [0, 0, 1] }, // Cybersecurity
];

// Convert data to tensors
const xs = tf.tensor2d(trainingData.map(d => d.input));
const ys = tf.tensor2d(trainingData.map(d => d.output));

// Build a Neural Network Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, activation: "relu", inputShape: [7] }));
model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

// Compile Model
model.compile({ optimizer: "adam", loss: "categoricalCrossentropy" });

// Train Model
async function trainModel() {
  await model.fit(xs, ys, { epochs: 100 });
  console.log("✅ AI Model Trained Successfully!");
}

// Function to Predict Career
async function predictCareer(userInput) {
  const prediction = model.predict(tf.tensor2d([userInput]));
  const output = await prediction.data();
  const careerIndex = output.indexOf(Math.max(...output));

  const careers = ["Web Development", "Data Science", "Cybersecurity"];
  return careers[careerIndex];
}

// Train the AI model before exporting
trainModel();

module.exports = { predictCareer };
