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
        body: JSON.stringify({ volume_level: parseInt(volume) })
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


function color_eye() {
    const dropdown = document.querySelector("#eye_color_dropdown");
    dropdown.style.display = dropdown.style.display === "none" ? "block" : "none";
};

document.querySelector("#eye_color_dropdown").addEventListener("change", function () {
    const selectedOption = this.value;
    if (selectedOption !== "") {
        fetch("/api/movement/" + selectedOption, { method: "GET" })
            .then(response => response.json())
            .then(data => console.log("Risposta API:", data))
            .catch(error => console.error("Errore:", error));
    }
});

function wake_up() {
    fetch("/api/movement/nao_wakeup", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function sit_down() {
    fetch("/api/movement/nao_sitdown", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function stand_init() {
    fetch("/api/movement/standInit", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function head_touch() {
    fetch("/nao_touch_head", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function stop_train() {
    fetch("/api/movement/nao_train_move", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};
function stand() {
    fetch("/api/movement/stand", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function a_life() {
    fetch("/api/movement/nao_autonomous_life", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function a_life_stop() {
    fetch("/api/movement/nao_autonomous_life_state", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

// Funzioni per i tasti del joystick
function start() {
    fetch("/api/movement/start", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function left() {
    fetch("/api/movement/left", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function stop() {
    fetch("/api/movement/stop", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function right() {
    fetch("/api/movement/right", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};

function back() {
    fetch("/api/movement/back", { method: "GET" })
        .then(response => response.json())
        .then(data => console.log("Risposta API:", data))
        .catch(error => console.error("Errore:", error));
};
function fake1(){
    testo="Sostituire la giocatrice col bracciale nero"
    fetch("/tts_to_nao", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "message=" + encodeURIComponent(testo)
    })
    .then(response => response.json())
    .then(data => console.log("Testo inviato con successo al NAO!"))
    .catch(error => console.error("Errore:", error));
}
function fake2(){
    testo="Sostituire la giocatrice col bracciale marrone"
    fetch("/tts_to_nao", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "message=" + encodeURIComponent(testo)
    })
    .then(response => response.json())
    .then(data => console.log("Testo inviato con successo al NAO!"))
    .catch(error => console.error("Errore:", error));
}
function fake3(){
    testo="prova prova " 
    fetch("/tts_to_nao", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "message=" + encodeURIComponent(testo)
    })
    .then(response => response.json())
    .then(data => console.log("Testo inviato con successo al NAO!"))
    .catch(error => console.error("Errore:", error));
}
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


        /* script fatto prima
        
        */