// Pre-create the canvas and context to reuse for each frame
const canvas     = document.createElement('canvas');
const context    = canvas.getContext('2d');
const imgElement = document.getElementById('video_feed');

const customSource = {
    getFrame(maxSize) {
        return new Promise((resolve) => {
            // Only resize the canvas if the image size changes
            if (canvas.width !== imgElement.naturalWidth || canvas.height !== imgElement.naturalHeight) {
                canvas.width  = imgElement.naturalWidth;
                canvas.height = imgElement.naturalHeight;
            }

            // Draw image onto canvas
            context.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

            // Convert canvas to ImageData
            resolve(context.getImageData(0, 0, canvas.width, canvas.height));
        });
    },
    start() {
        // Implement if there's a way or need to start the stream
    },
    stop() {
        // Implement if there's a way or need to stop the stream
    },
    get stopped() {
        // Implement to return the camera stream status
    }
};

// SDK MorphCast
CY.loader()
    .licenseKey("fc293a6733f11a98a89a3b5cc9ec8ed802715e2c5862")
    .source(customSource)
    .addModule(CY.modules().FACE_DETECTOR.name)
    .addModule(CY.modules().FACE_AGE.name)
    .addModule(CY.modules().FACE_GENDER.name)
    .addModule(CY.modules().FACE_EMOTION.name)
    .addModule(CY.modules().FACE_ATTENTION.name)
    .load().then(({ start }) => {
        start();
    });

// Result
const det_div = document.querySelector("#face_detect");
const age_div = document.querySelector("#face_age");
const gen_div = document.querySelector("#face_gender");
const emo_div = document.querySelector("#face_emotion");
const options = {
    chart: {
      height: 350,
      width: 500,
      type: 'bar'
    },
    yaxis: {
      min:0,
      max:100
    },
    series:[],
    title: {
      text: ''
    },
    labels:[]
};
const chart = new ApexCharts(document.querySelector("#chart"), options);
chart.render();

var send_det;
window.addEventListener(CY.modules().FACE_DETECTOR.eventName, (evt) => {
    console.log('FACE_DETECTOR result', evt.detail);
    det_div.innerHTML = 'Detect: ' + evt.detail.totalFaces;
    send_det = evt.detail.totalFaces;
});

var send_age;
window.addEventListener(CY.modules().FACE_AGE.eventName, (evt) => {
    console.log('FACE_AGE result', evt.detail);
    age_div.innerHTML = 'Age: ' + evt.detail.output.numericAge;
    send_age = evt.detail.output.numericAge;
});

var send_gen;
window.addEventListener(CY.modules().FACE_GENDER.eventName, (evt) => {
    console.log('FACE_GENDER result', evt.detail);
    gen_div.innerHTML = 'Gender: ' + evt.detail.output.mostConfident;
    const send_gen = evt.detail.output.mostConfident;
});

var send_emo_dom;
var emotions;
window.addEventListener(CY.modules().FACE_EMOTION.eventName, (evt) => {
    console.log('FACE_EMOTION result', evt.detail);
    emo_div.innerHTML = 'Emotion: ' + evt.detail.output.dominantEmotion;
    send_emo_dom = evt.detail.output.dominantEmotion;

    // data for the histogram
    emotions = evt.detail.output.emotion;
    const labels   = [];
    const data     = [];

    Object.keys(emotions).forEach(k => {
      labels.push(k);
      data.push(parseInt((emotions[k] * 100).toFixed(0)));
    });

    chart.updateOptions({
      labels: labels,
      series:[{
        name: 'Emotion',
        data: data
      }]
    });
});

var attention;
window.addEventListener(CY.modules().FACE_ATTENTION.eventName, (evt) => {
    console.log('FACE_ATTENTION result', evt.detail);
    attention = evt.detail.output.attention;
    const elem = document.getElementById("myBar");
    elem.style.width = attention * 100 + "%";
});


function inviaDati(){
    var url = '/api/morphcast';
    var dati = {
        det: send_det,
        age: send_age,
        gen: send_gen,
        emo_dom: send_emo_dom,
        emo: emotions,
        att: attention
    };

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dati)
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
    })
    .catch(error => {
        console.error('Errore:', error);
    });
}

// Invia i dati ogni 10 secondi
setInterval(inviaDati, 10000);
