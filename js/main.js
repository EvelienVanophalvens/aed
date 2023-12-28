let model;
window.onload = async function() {
    //get location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
        }, function(error) {
            console.error("Error occurred while getting location: " + error.message);
        });
    } else {
        console.error("Geolocation is not supported by this browser.");
    }

    //load model
    const imgData = localStorage.getItem('capturedImage');
    const imgElement = document.getElementById('capturedimg');
    imgElement.src = imgData;

    // Load the model
    model = await tf.loadLayersModel('./model/my-model.json');

    // Preprocess the image
    let imgTensor = tf.browser.fromPixels(imgElement).toFloat();
    imgTensor = imgTensor.resizeBilinear([32, 32]); // Resize to 32x32 pixels
    imgTensor = imgTensor.mean(2).expandDims(2).div(tf.scalar(255)); // Convert to grayscale and normalize
    imgTensor = imgTensor.reshape([1, 1024]); // Reshape to a 1D tensor of length 1024
    // Make a prediction
    const prediction = model.predict(imgTensor);

    // Interpret the prediction
    const predictionLabel = prediction.dataSync()[0]; // Get the index of the maximum value
    if (predictionLabel > 0.5) {
        console.log("The image is an AED.");
        //add to database

        //add yes to the localstorage
        document.querySelector('.succes').style.display = 'block';
        document.querySelector('.error').style.display = 'none';
        document.querySelector('.placingerror').style.display = 'none';
        document.querySelector('.angleerror').style.display = 'none';
        document.querySelector('.addnewerror').style.display = 'none';
    } else {
        console.log("The image is not an AED.");
        console.log(predictionLabel);
        document.querySelector('.succes').style.display = 'none';
        document.querySelector('.error').style.display = 'block';
        document.querySelector('.iferror').style.display = 'block';
        document.querySelector('.placing').style.display = 'none';
        document.querySelector('.angle').style.display = 'none';
        document.querySelector('.addnew').style.display = 'none';
    }
};