window.onload = async function() {
    //get location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            console.log("Latitude: " + position.coords.latitude);
            console.log("Longitude: " + position.coords.longitude);
        }, function(error) {
            console.error("Error occurred while getting location: " + error.message);
        });
    } else {
        console.error("Geolocation is not supported by this browser.");
    }
};