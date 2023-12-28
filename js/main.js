let model;
window.onload = async function() {
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
        //add yes to the localstorage
        document.querySelector('.adding').style.display = 'block';
        document.querySelector('.succes').style.display = 'none';
        document.querySelector('.error').style.display = 'none';
        document.querySelector('.placingerror').style.display = 'none';
        document.querySelector('.angleerror').style.display = 'none';
        document.querySelector('.addnewerror').style.display = 'none';
        // Get location
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(async function(position) {
                console.log("Latitude: " + position.coords.latitude);
                console.log("Longitude: " + position.coords.longitude);

                // Set location to address
                var requestOptions = {
                    method: 'GET',
                };

                // Use the latitude and longitude in the fetch URL
                let url = `https://api.geoapify.com/v1/geocode/reverse?lat=${position.coords.latitude}&lon=${position.coords.longitude}&apiKey=8a4cb43d586846008016b65d159b0d37`;

                fetch(url, requestOptions)
                    .then(response => response.json())
                    // .then(result => console.log(result))
                    .then(result => {
                            city = result['features'][0].properties.city;
                            console.log(city);
                            street = result['features'][0].properties.street + " " + result['features'][0].properties.housenumber;
                            console.log(street);
                            document.getElementById('city').innerHTML = city;
                            document.getElementById('street').innerHTML = street;
                            //send to db
                            
                            //added to db (still have to do it right after failure AND succes of db adding)
                            document.querySelector('.adding').style.display = 'none';
                            document.querySelector('.succes').style.display = 'block';
                        }
                        )
                    .catch(error => console.log('error', error));
            }, function(error) {
                console.error("Error occurred while getting location: " + error.message);
            });
        } else {
            console.error("Geolocation is not supported by this browser.");
        }
    } else {
        console.log("The image is not an AED.");
        console.log(predictionLabel);
        document.querySelector('.adding').style.display = 'none';
        document.querySelector('.succes').style.display = 'none';
        document.querySelector('.error').style.display = 'block';
        document.querySelector('.placing').style.display = 'none';
        document.querySelector('.angle').style.display = 'none';
        document.querySelector('.addnew').style.display = 'none';
    }
};