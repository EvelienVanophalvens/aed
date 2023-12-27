const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// Just add more buttons in HTML to allow classification of more classes of data!
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // For mobile.
  dataCollectorButtons[i].addEventListener('touchend', gatherDataForClass);

  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}


let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;


/**
 * Loads the MobileNet model and warms it up so ready for use.
 **/
async function loadMobileNetFeatureModel() {
  const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  
  // Warm up the model by passing zeros through it once.
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}

loadMobileNetFeatureModel();


for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [1024]}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

model.compile({
  optimizer: tf.train.adam(1e-30),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});



async function gatherDataForClass() {
    console.log('Gathering data for class ' + this.getAttribute('data-name'));
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  
    // get all images from the aed's folder and store them in an array of image urls
    let imageUrls = [
        "data/aed_0.jpg",
        "data/aed_1.jpg",
        "data/aed_2.jpg",
        "data/aed_3.jpg",
        "data/aed_4.jpg",
        "data/aed_5.jpg",
        "data/aed_6.jpg",
        "data/aed_7.jpg",
        "data/aed_8.jpg",
        "data/aed_9.jpg",
        "data/aed_10.jpg",
        "data/aed_11.jpg",
        "data/aed_12.jpg",
        "data/aed_13.jpg",
        "data/aed_14.jpg",
        "data/aed_15.jpg",
        "data/aed_16.jpg",
        "data/aed_17.jpg",
        "data/aed_18.jpg",
        "data/aed_19.jpg",
        "data/aed_20.jpg",
        "data/aed_21.jpg",
        "data/aed_22.jpg",
        "data/aed_23.jpg",
        "data/aed_24.jpg",
        "data/aed_25.jpg",
        "data/aed_26.jpg",
        "data/aed_27.jpg",
        "data/aed_28.jpg",
        "data/aed_29.jpg",
        "data/aed_30.jpg",
        "data/aed_31.jpg",
        "data/aed_32.jpg",
        "data/aed_33.jpg",
        "data/aed_34.jpg",
        "data/aed_35.jpg",
        "data/aed_36.jpg",
        "data/aed_37.jpg",
        "data/aed_38.jpg",
        "data/aed_39.jpg",
        "data/aed_40.jpg",
        "data/aed_41.jpg",
        "data/aed_42.jpg",
        "data/aed_43.jpg",
        "data/aed_44.jpg",
        "data/aed_45.jpg",
        "data/aed_46.jpg",
        "data/aed_47.jpg",
        "data/aed_48.jpg",
        "data/aed_49.jpg",
    ];
  
    for (let i = 0; i < imageUrls.length; i++) {
        console.log(imageUrls[i]);
        let img = new Image();
        img.src = imageUrls[i];
        await new Promise((resolve) => img.onload = resolve);
    
        let imageFeatures = tf.tidy(() => {
          let imgTensor = tf.browser.fromPixels(img);
          let resizedTensorFrame = tf.image.resizeBilinear(
              imgTensor, 
              [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
              true
          );
          let normalizedTensorFrame = resizedTensorFrame.div(255);
          return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
        });
    
        trainingDataInputs.push(imageFeatures);
        trainingDataOutputs.push(gatherDataState);
        
        if (examplesCount[gatherDataState] === undefined) {
          examplesCount[gatherDataState] = 0;
          console.log(examplesCount[gatherDataState]);
        }
        examplesCount[gatherDataState]++;
        document.getElementById("image").src = imageUrls[i];
      }
  }


function dataGatherLoop() {
    if (gatherDataState !== STOP_DATA_GATHER) {
        // Ensure tensors are cleaned up.
        let imageFeatures = calculateFeaturesOnCurrentFrame();
    
        trainingDataInputs.push(imageFeatures);
        trainingDataOutputs.push(gatherDataState);
        
        // Intialize array index element if currently undefined.
        if (examplesCount[gatherDataState] === undefined) {
          examplesCount[gatherDataState] = 0;
        }
        // Increment counts of examples for user interface to show.
        examplesCount[gatherDataState]++;
    
        STATUS.innerText = '';
        for (let n = 0; n < CLASS_NAMES.length; n++) {
          STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
        }
    
        window.requestAnimationFrame(dataGatherLoop);
      }
}

let currentImageIndex = 0;
  let imageUrls = [
      "testData/aed_0.jpg",
      "testData/aed_1.jpeg",
      "testData/aed_3.jpeg",
      "testData/aed_4.jpg",
  ];


  async function calculateFeaturesOnCurrentFrame() {
    console.log(imageUrls[currentImageIndex]);
    // Load the image
    let img = new Image();
    img.src = imageUrls[currentImageIndex];
    await new Promise((resolve) => img.onload = resolve);

    //set image to image tag
    document.getElementById("image").src = imageUrls[currentImageIndex];
    return tf.tidy(function() {
      // Grab pixels from current image.
      let imageAsTensor = tf.browser.fromPixels(img);
      // Resize image tensor to be 224 x 224 pixels which is needed by MobileNet for input.
      let resizedTensorFrame = tf.image.resizeBilinear(
          imageAsTensor, 
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
      );
  
      let normalizedTensorFrame = resizedTensorFrame.div(255);
  
      // Increment the current image index so the next image is used next time
      currentImageIndex = (currentImageIndex + 1) % imageUrls.length;
  
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
  }


  async function trainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'float32'); // Change here
    let inputsAsTensor = tf.stack(trainingDataInputs);
    
    let results = await model.fit(inputsAsTensor, outputsAsTensor, { // Change here
      shuffle: true,
      batchSize: 50,
      epochs: 10,
      callbacks: {onEpochEnd: logProgress}
    });
  
      // Evaluate the model on the training data
      const evalResult = model.evaluate(inputsAsTensor, outputsAsTensor);
      console.log('Training data evaluation result:', evalResult);
  
      console.log('Inputs tensor:', inputsAsTensor);
      console.log('Outputs tensor:', outputsAsTensor);
  
    // Dispose of tensors after fitting
    outputsAsTensor.dispose();
    inputsAsTensor.dispose();
  
  
    predict = true;
    predictLoop();

    const saveResults = await model.save('downloads://my-model');
    console.log(saveResults);
     
  }
  /*async function loadModelAndPredict() {
    console.log('Loading model...');  
    try {
      // Load the model from the file.
      const loadedModel = await tf.loadLayersModel('./my-model.json');
  
      // Dispose of the existing model to free up memory
      loadedModel.dispose(); 
      
      // Assign the loaded model to the global 'model' variable
  
      // Start predictions with the loaded model
      predict = true;
      predictLoop();
    } catch (error) {
      console.error('Error loading the model:', error);
    }
  }*/



async function predictLoop() {
    let imageFeatures = await calculateFeaturesOnCurrentFrame();
    
    tf.tidy(function() {
        let prediction = model.predict(imageFeatures.expandDims()).squeeze();
        if (prediction.rank > 0) {
            let highestIndex = prediction.argMax(0).dataSync()[0];
            let predictionArray = prediction.arraySync();
            STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        } else {
            // Handle the case where prediction is a scalar
            let predictionValue = prediction.arraySync();
            STATUS.innerText = 'Prediction: ' + (predictionValue > 0.5 ? CLASS_NAMES[1] : CLASS_NAMES[0]) + ' with ' + Math.floor(predictionValue * 100) + '% confidence';
        }
    });
}


/**
 * Log training progress.
 **/
function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}


function reset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;
    STATUS.innerText = 'No data collected';
    
    console.log('Tensors in memory: ' + tf.memory().numTensors);
}