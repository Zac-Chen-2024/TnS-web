document.getElementById("voice-btn").addEventListener("click", function() {
    let recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.start();
    recognition.onresult = function(event) {
        let recognizedText = event.results[0][0].transcript;
        console.log("语音输入:", recognizedText);
    };
});