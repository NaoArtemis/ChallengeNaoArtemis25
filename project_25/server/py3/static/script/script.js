document.querySelector("#home").addEventListener("click", function() {
    fetch("/home", { method: "GET" })
    .then(response => {
        if (response.redirected) {
            window.location.href = response.url;
        } else {
            window.location.href = "/home";
        }
    })
    .catch(error => console.error("Errore:", error));
});


document.querySelector("#volume").addEventListener("click", function (event) {
    event.stopPropagation();
    const volumeControl = document.querySelector("#volume-control");
    volumeControl.style.display = volumeControl.style.display === "block" ? "none" : "block";
});

document.addEventListener("click", function () {
    const volumeControl = document.querySelector("#volume-control");
    volumeControl.style.display = "none";
});

document.querySelector("#volume_slider").addEventListener("input", function () {
    // Otteniamo il valore dello slider
    const volume = this.value;
    // Aggiorniamo l'elemento che mostra il valore del volume
    document.querySelector("#volume_value").textContent = volume;

    // Inviamo il valore (modificato, ad esempio volume + 20) al server
    fetch("/set_volume", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ volume_level: parseInt(volume) + 20 })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Errore nella richiesta");
        }
        return response.json();
    })
    .then(data => {
        console.log("Risposta del server:", data);
    })
    .catch(error => {
        console.error("Errore:", error);
    });
});




document.querySelector("#battery-icon").addEventListener("mouseover", function () {
    const batteryLevelElement = document.querySelector("#battery-level");
    fetch('/nao/battery')
        .then(response => {
            if (!response.ok) {
                throw new Error('Errore nel recupero del livello di batteria');
            }
            return response.json();
        })
        .then(data => {
            batteryLevelElement.textContent = 'Livello batteria:'+ data.battery_level+'%x';
            batteryLevelElement.style.display = "block";
        })
        .catch(error => {
            console.error('Errore durante il recupero:', error);
            batteryLevelElement.textContent = 'Errore nel recupero.';
            batteryLevelElement.style.display = "block"; // Mostra il tooltip anche in caso di errore
        });
});

document.querySelector("#battery-icon").addEventListener("mouseout", function () {
    const batteryLevelElement = document.querySelector("#battery-level");
    batteryLevelElement.style.display = "none";
});

document.querySelector("#logout").addEventListener("click", function() {
        fetch("/logout", { method: "GET" })
        .then(response => {
            if (response.redirected) {
            window.location.href = response.url;
            }
        })
        .catch(error => console.error("Errore:", error));
    });

document.querySelector("#battery-icon").addEventListener("mouseout", function () {
    const batteryLevelElement = document.querySelector("#battery-level");
    batteryLevelElement.style.display = "none";
});


document.querySelector("#color_eye_button").addEventListener("click", function () {
    const dropdown = document.querySelector("#eye_color_dropdown");
    dropdown.style.display = dropdown.style.display === "none" ? "block" : "none";
});

document.querySelector("#eye_color_dropdown").addEventListener("change", function () {
    const selectedOption = this.value;
    if (selectedOption !== "") {
        fetch("/api/movement/" + selectedOption, { method: "GET" })
            .then(response => response.json())
            .then(data => console.log("Risposta API:", data))
            .catch(error => console.error("Errore:", error));
    }
});

document.querySelector("#wakeup").addEventListener("click", function () {
    fetch("/api/movement/nao_wakeup", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#autonomo").addEventListener("click", function () {
    fetch("/api/movement/nao_autonomous_life", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#stand_init").addEventListener("click", function () {
    fetch("/api/movement/standInit", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#head_touch").addEventListener("click", function () {
    fetch("/nao_touch_head_audiorecorder", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data));
});

document.querySelector("#stop_train").addEventListener("click", function () {
    fetch("/api/movement/nao_train_move", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#stand").addEventListener("click", function () {
    fetch("/api/movement/stand", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

// Funzioni per i tasti del joystick
document.querySelector("#start").addEventListener("click", function () {
    fetch("/api/movement/start", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#sinistra").addEventListener("click", function () {
    fetch("/api/movement/left", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#stop").addEventListener("click", function () {
    fetch("/api/movement/stop", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#destra").addEventListener("click", function () {
    fetch("/api/movement/right", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

document.querySelector("#back").addEventListener("click", function () {
    fetch("/api/movement/back", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
});

// Funzioni per inviare testo al NAO
function inviaTestoAlNAO() {
    const testo = document.getElementById("testoInput").value;
    fetch("/tts_to_nao", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "message=" + encodeURIComponent(testo)
    })
    .then(response => response.json())
    .then(data => console.log("Testo inviato con successo al NAO!"))
    .catch(error => console.error("Errore:", error));
}

function inviaTestoAlNAOai() {
    const testo = document.getElementById("testoInput_ai").value;
    fetch("/tts_to_nao_ai", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "message_ai=" + encodeURIComponent(testo)
    })
    .then(response => response.json())
    .then(data => console.log("Testo inviato con successo al NAO con AI!"))
    .catch(error => console.error("Errore:", error));
}

function showNaoCam() {
    document.getElementById("webcam-feed").src = naoCamUrl;
}

function showArucoCam() {
    document.getElementById("webcam-feed").src = arucoCamUrl;
}

function stopCam() {
    document.getElementById("webcam-feed").src = noCamUrl;
}

window.onload = function() {
    stopCam();
};




        /* script fatto prima
        #webcam
                document.addEventListener('DOMContentLoaded', function() {
    const naoBtn = document.getElementById('nao-cam-btn');
    const arucoBtn = document.getElementById('aruco-cam-btn');
    const naoCam = document.getElementById('nao-cam');
    const arucoCam = document.getElementById('aruco-cam');

    function toggleCam(selectedCam) {
        if((selectedCam === 'nao' && naoCam.classList.contains('active')) || 
        (selectedCam === 'aruco' && arucoCam.classList.contains('active'))) {
            naoCam.classList.remove('active');
            arucoCam.classList.remove('active');
            naoBtn.classList.remove('active');
            arucoBtn.classList.remove('active');
            return;
        }
        
                naoCam.classList.remove('active');
                arucoCam.classList.remove('active');
                naoBtn.classList.remove('active');
                arucoBtn.classList.remove('active');
                
                if(selectedCam === 'nao') {
                    naoCam.classList.add('active');
                    naoBtn.classList.add('active');
                } else {
                    arucoCam.classList.add('active');
                    arucoBtn.classList.add('active');
                }
            }

            naoBtn.addEventListener('click', () => toggleCam('nao'));
            arucoBtn.addEventListener('click', () => toggleCam('aruco'));
        });


        
        
        
        */