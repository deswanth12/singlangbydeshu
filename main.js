const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const outputTextElement = document.getElementById('outputText');

// --- Temporal Smoother (JavaScript version) ---
class TemporalSmoother {
    constructor(maxlen = 9) {
        this.buffer = [];
        this.maxlen = maxlen;
    }

    add(label) {
        if (label) {
            this.buffer.push(label);
            if (this.buffer.length > this.maxlen) {
                this.buffer.shift(); // Remove the oldest element
            }
        }
    }

    get() {
        if (this.buffer.length === 0) {
            return null;
        }
        const counts = this.buffer.reduce((acc, val) => {
            acc[val] = (acc[val] || 0) + 1;
            return acc;
        }, {});
        return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
    }
}

const smoother = new TemporalSmoother();
let lastSpoken = "";
let lastText = "";

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    // Ensure results.image is available before drawing
    if (results.image) {
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const handLandmarks = results.multiHandLandmarks[0]; // Use the first hand

        // Draw landmarks
        drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        drawLandmarks(canvasCtx, handLandmarks, { color: '#FF0000', lineWidth: 2 });

        // Prepare data for backend
        const landmarks_for_backend = handLandmarks.map(lm => [lm.x, lm.y, lm.z]);

        // Send to backend for classification
        fetch('/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: landmarks_for_backend })
        })
        .then(response => response.json())
        .then(data => {
            if (data.text) {
                smoother.add(data.text);
                lastText = smoother.get() || data.text;

                // Update display text
                outputTextElement.innerText = `Text: ${lastText}`;

                // Speak only when text changes
                if (lastText !== lastSpoken) {
                    const utterance = new SpeechSynthesisUtterance(lastText);
                    window.speechSynthesis.speak(utterance);
                    lastSpoken = lastText;
                }
            }
        })
        .catch(error => console.error('Error:', error));

    } else {
        outputTextElement.innerText = 'No hand detected';
    }

    canvasCtx.restore();
}

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});
camera.start();