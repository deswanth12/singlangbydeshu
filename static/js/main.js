const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

const currentSignTextEl = document.getElementById('currentSignText');
const currentModeTagEl = document.getElementById('currentModeTag');
const confBarFillEl = document.getElementById('confBarFill');

const fFillThumbL = document.getElementById('fFillThumbL');
const fFillIndexL = document.getElementById('fFillIndexL');
const fFillMiddleL = document.getElementById('fFillMiddleL');
const fFillRingL = document.getElementById('fFillRingL');
const fFillPinkyL = document.getElementById('fFillPinkyL');

const fFillThumbR = document.getElementById('fFillThumbR');
const fFillIndexR = document.getElementById('fFillIndexR');
const fFillMiddleR = document.getElementById('fFillMiddleR');
const fFillRingR = document.getElementById('fFillRingR');
const fFillPinkyR = document.getElementById('fFillPinkyR');

const sentenceBufferEl = document.getElementById('sentenceBuffer');
const soundwaveAnimEl = document.getElementById('soundwaveAnim');
const historyListEl = document.getElementById('historyList');

const voiceSelectEl = document.getElementById('voiceSelect');
const langSelectEl = document.getElementById('langSelect');
const speechRateEl = document.getElementById('speechRate');
const toggleTTSEl = document.getElementById('toggleTTS');
const themeSelectEl = document.getElementById('themeSelect');

const dictGridEl = document.getElementById('dictGrid');
const dictionaryModalEl = document.getElementById('dictionaryModal');
const speechToSignModalEl = document.getElementById('speechToSignModal');
const recordCustomModalEl = document.getElementById('recordCustomModal');
const quizHudPanelEl = document.getElementById('quizHudPanel');
const quizTargetTextEl = document.getElementById('quizTargetText');
const quizScoreTextEl = document.getElementById('quizScoreText');
const quizTimerTextEl = document.getElementById('quizTimerText');

const toastContainerEl = document.getElementById('toastContainer');
const fpsDisplayEl = document.getElementById('fpsDisplay');

// State Management
let sentenceArray = [];
let lastSpokenText = "";
let lastAddedText = "";
let lastAddedTime = 0;
let currentStableGesture = null;
let gestureStableFrames = 0;
const SAME_SIGN_COOLDOWN_MS = 1500;
let lastClassifyTime = 0;
const CLASSIFY_INTERVAL_MS = 100;

let frameCount = 0;
let lastFpsTime = Date.now();
let historyItems = [];
let gestureDict = [];
let activeCategory = 'ALL';
let currentLandmarksPayload = null;


// Quiz Game State
let isQuizActive = false;
let quizTarget = null;
let quizScore = 0;
let quizStreak = 0;
let quizTimer = 10;
let quizInterval = null;

// Theme state
let currentTheme = 'CYBERPUNK';

// Audio Context Feedback
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playChime(freq = 440, duration = 0.12) {
    try {
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {}
}

// Toast Notifications
function showToast(message, icon = '✨') {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    toastContainerEl.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(10px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// --- Temporal Smoother ---
class TemporalSmoother {
    constructor(maxlen = 4) {
        this.buffer = [];
        this.maxlen = maxlen;
    }

    add(label) {
        if (label) {
            this.buffer.push(label);
            if (this.buffer.length > this.maxlen) {
                this.buffer.shift();
            }
        }
    }

    getDominant() {
        if (this.buffer.length === 0) return { label: null, confidence: 0 };
        const counts = {};
        this.buffer.forEach(val => counts[val] = (counts[val] || 0) + 1);
        
        let maxLabel = null;
        let maxCount = 0;
        for (const [lbl, count] of Object.entries(counts)) {
            if (count > maxCount) {
                maxCount = count;
                maxLabel = lbl;
            }
        }
        return { label: maxLabel, confidence: maxCount / this.buffer.length };
    }

    reset() {
        this.buffer = [];
    }
}

const smoother = new TemporalSmoother();

// Web Speech Synthesis
let synthVoices = [];
function populateVoices() {
    if ('speechSynthesis' in window) {
        synthVoices = window.speechSynthesis.getVoices();
        const selectedLang = langSelectEl.value || 'en';
        
        voiceSelectEl.innerHTML = '<option value="">Default System Voice</option>';
        synthVoices.forEach((voice, i) => {
            if (voice.lang.toLowerCase().startsWith(selectedLang)) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `${voice.name} (${voice.lang})`;
                voiceSelectEl.appendChild(option);
            }
        });
    }
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = populateVoices;
    populateVoices();
}

langSelectEl.addEventListener('change', populateVoices);

function speakText(text, forceSpeak = false) {
    if ((!toggleTTSEl.checked && !forceSpeak) || !text || text === "Detecting...") return;
    if (!forceSpeak && text === lastSpokenText) return;

    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        if (voiceSelectEl.value !== "") {
            const idx = parseInt(voiceSelectEl.value, 10);
            if (synthVoices[idx]) {
                utterance.voice = synthVoices[idx];
            }
        }
        utterance.rate = parseFloat(speechRateEl.value) || 1.0;

        utterance.onstart = () => { soundwaveAnimEl.style.display = 'flex'; };
        utterance.onend = () => { soundwaveAnimEl.style.display = 'none'; };
        utterance.onerror = () => { soundwaveAnimEl.style.display = 'none'; };

        window.speechSynthesis.speak(utterance);
        lastSpokenText = text;
    }
}

toggleTTSEl.addEventListener('change', () => {
    if (!toggleTTSEl.checked) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        soundwaveAnimEl.style.display = 'none';
        lastSpokenText = "";
        showToast("Auto Text-to-Speech Disabled 🔇", "ℹ️");
    } else {
        lastSpokenText = "";
        showToast("Auto Text-to-Speech Enabled 🔊", "✨");
    }
});

// UI Sentence Display with Word Chips
function updateSentenceDisplay() {
    if (sentenceArray.length === 0) {
        sentenceBufferEl.innerHTML = `<span style="color: var(--text-muted); font-size: 0.95rem; font-style: italic;">Perform gestures in front of the webcam to form words and sentences...</span>`;
    } else {
        sentenceBufferEl.innerHTML = sentenceArray.map((word, idx) => {
            if (word === ' ') {
                return `<span style="color: var(--primary-cyan); font-weight: 700; cursor: pointer;" title="Click to remove space" onclick="removeWord(${idx})">␣</span>`;
            }
            return `
                <span class="word-chip">
                    <span>${word}</span>
                    <span class="word-chip-delete" onclick="removeWord(${idx})">✕</span>
                </span>
            `;
        }).join('');
    }
}

window.removeWord = function(idx) {
    sentenceArray.splice(idx, 1);
    updateSentenceDisplay();
    playChime(350, 0.1);
};

function addHistoryItem(text) {
    if (!text || text === "Detecting...") return;
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    historyItems.unshift({ text, time: timeStr });
    if (historyItems.length > 15) historyItems.pop();

    historyListEl.innerHTML = historyItems.map(item => `
        <li class="timeline-item">
            <span class="timeline-text">${item.text}</span>
            <span class="timeline-time">${item.time}</span>
        </li>
    `).join('');
}

// Dual Left & Right Finger Meter Updates
function updateFingerMeters(handLm, side = 'L') {
    const wrist = handLm[0];
    const getRatio = (tip, mcp) => {
        const d1 = Math.hypot(handLm[tip].x - wrist.x, handLm[tip].y - wrist.y);
        const d2 = Math.hypot(handLm[mcp].x - wrist.x, handLm[mcp].y - wrist.y) + 1e-5;
        return Math.min(100, Math.max(10, Math.round((d1 / d2) * 55)));
    };

    if (side === 'L') {
        if (fFillThumbL) fFillThumbL.style.height = `${getRatio(4, 2)}%`;
        if (fFillIndexL) fFillIndexL.style.height = `${getRatio(8, 5)}%`;
        if (fFillMiddleL) fFillMiddleL.style.height = `${getRatio(12, 9)}%`;
        if (fFillRingL) fFillRingL.style.height = `${getRatio(16, 13)}%`;
        if (fFillPinkyL) fFillPinkyL.style.height = `${getRatio(20, 17)}%`;
    } else {
        if (fFillThumbR) fFillThumbR.style.height = `${getRatio(4, 2)}%`;
        if (fFillIndexR) fFillIndexR.style.height = `${getRatio(8, 5)}%`;
        if (fFillMiddleR) fFillMiddleR.style.height = `${getRatio(12, 9)}%`;
        if (fFillRingR) fFillRingR.style.height = `${getRatio(16, 13)}%`;
        if (fFillPinkyR) fFillPinkyR.style.height = `${getRatio(20, 17)}%`;
    }
}

function clearFingerMeters(side = 'L') {
    const setZero = (el) => { if (el) el.style.height = '0%'; };
    if (side === 'L') {
        setZero(fFillThumbL); setZero(fFillIndexL); setZero(fFillMiddleL); setZero(fFillRingL); setZero(fFillPinkyL);
    } else {
        setZero(fFillThumbR); setZero(fFillIndexR); setZero(fFillMiddleR); setZero(fFillRingR); setZero(fFillPinkyR);
    }
}


// Dynamic Theme Canvas Skeleton Drawing
function drawThemedLandmarks(ctx, handLandmarks, labelStr = 'Hand') {
    const isLeft = labelStr.toLowerCase() === 'left';

    let jointColor, connColor;

    if (currentTheme === 'MATRIX') {
        jointColor = '#00FF66';
        connColor = 'rgba(0, 255, 102, 0.85)';
    } else if (currentTheme === 'RAINBOW') {
        const hue = (frameCount * 6) % 360;
        jointColor = `hsl(${hue}, 100%, 60%)`;
        connColor = `hsl(${(hue + 40) % 360}, 100%, 50%)`;
    } else if (currentTheme === 'MINIMAL') {
        jointColor = '#FFFFFF';
        connColor = 'rgba(255, 255, 255, 0.4)';
    } else {
        // CYBERPUNK
        jointColor = isLeft ? '#00F2FE' : '#FF0844';
        connColor = isLeft ? 'rgba(0, 242, 254, 0.85)' : 'rgba(255, 8, 68, 0.85)';
    }

    drawConnectors(ctx, handLandmarks, HAND_CONNECTIONS, { color: connColor, lineWidth: 5 });
    drawLandmarks(ctx, handLandmarks, { color: jointColor, fillColor: '#FFFFFF', lineWidth: 2, radius: 5 });

    const wrist = handLandmarks[0];
    const xPx = wrist.x * canvasElement.width;
    const yPx = wrist.y * canvasElement.height + 25;

    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillStyle = jointColor;
    ctx.shadowColor = 'black';
    ctx.shadowBlur = 4;
    ctx.fillText(`${labelStr.toUpperCase()} HAND`, xPx - 30, yPx);
    ctx.shadowBlur = 0;
}

themeSelectEl.addEventListener('change', (e) => {
    currentTheme = e.target.value;
    showToast(`Switched Theme to ${e.target.options[e.target.selectedIndex].text}`, "🎨");
});

// MediaPipe Frame Results Callback
function onResults(results) {
    frameCount++;
    const now = Date.now();
    if (now - lastFpsTime >= 1000) {
        fpsDisplayEl.textContent = `FPS: ${frameCount}`;
        frameCount = 0;
        lastFpsTime = now;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.image) {
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    }

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const handCount = results.multiHandLandmarks.length;
        // Swap MediaPipe selfie camera mirror orientation (Left <-> Right)
        const handednessList = (results.multiHandedness || []).map(h => {
            if (h.label === 'Left') return 'Right';
            if (h.label === 'Right') return 'Left';
            return h.label;
        });


        let sawLeft = false, sawRight = false;

        results.multiHandLandmarks.forEach((handLm, idx) => {
            const hLabel = handednessList[idx] || (idx === 0 ? 'Left' : 'Right');
            drawThemedLandmarks(canvasCtx, handLm, hLabel);
            if (hLabel.toLowerCase() === 'left') {
                updateFingerMeters(handLm, 'L');
                sawLeft = true;
            } else {
                updateFingerMeters(handLm, 'R');
                sawRight = true;
            }
        });

        if (!sawLeft) clearFingerMeters('L');
        if (!sawRight) clearFingerMeters('R');

        if (now - lastClassifyTime > CLASSIFY_INTERVAL_MS) {
            lastClassifyTime = now;

            const landmarksPayload = results.multiHandLandmarks.map(handLm => 
                handLm.map(lm => [lm.x, lm.y, lm.z])
            );
            currentLandmarksPayload = landmarksPayload;

            fetch('/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    landmarks: landmarksPayload,
                    handedness: handednessList
                })
            })
            .then(res => res.json())
            .then(data => {
                if (data.text) {
                    smoother.add(data.text);
                    const { label, confidence } = smoother.getDominant();
                    const detectedText = label || data.text;
                    const confPercent = Math.round(confidence * 100);

                    const modeTag = handCount > 1 ? "👐 Dual-Hand Mode Active" : "✋ Single-Hand Mode";
                    currentSignTextEl.textContent = detectedText;
                    currentModeTagEl.textContent = `${modeTag}`;
                    confBarFillEl.style.width = `${confPercent}%`;

                    // Check Quiz Game match
                    if (isQuizActive && quizTarget) {
                        if (detectedText.toLowerCase().includes(quizTarget.name.toLowerCase())) {
                            quizScore += 100;
                            quizStreak += 1;
                            playChime(880, 0.25);
                            showToast(`Correct! Matched ${quizTarget.name} 🎉 (+100 pts)`, "🎯");
                            nextQuizTarget();
                        }
                    }

                    // Anti-repetition debouncing & stability check
                    if (confPercent >= 45 && detectedText !== "Detecting..." && detectedText !== "UNKNOWN") {
                        if (detectedText === currentStableGesture) {
                            gestureStableFrames++;
                        } else {
                            currentStableGesture = detectedText;
                            gestureStableFrames = 1;
                        }

                        const timeSinceLastAdd = now - lastAddedTime;
                        const isNewGesture = detectedText !== lastAddedText;
                        const isCooldownPassed = timeSinceLastAdd > SAME_SIGN_COOLDOWN_MS;

                        // Only add if held stably for at least 3 frames AND (it's a new sign OR cooldown passed)
                        if (gestureStableFrames >= 3 && (isNewGesture || isCooldownPassed)) {
                            sentenceArray.push(detectedText);
                            lastAddedText = detectedText;
                            lastAddedTime = now;
                            gestureStableFrames = 0;

                            updateSentenceDisplay();
                            addHistoryItem(detectedText);
                            speakText(detectedText);
                            playChime(587, 0.1);
                        }
                    } else {
                        gestureStableFrames = 0;
                    }
                }
            })
            .catch(err => console.error('Classification error:', err));
        }
    } else {
        currentSignTextEl.textContent = "Waiting for hand...";
        confBarFillEl.style.width = '0%';
        currentModeTagEl.textContent = "✋ Single-Hand Mode";
        clearFingerMeters('L');
        clearFingerMeters('R');
        gestureStableFrames = 0;
        currentStableGesture = null;
        // Do NOT reset lastAddedText here so brief frame drops don't re-trigger duplicate words!
    }

    canvasCtx.restore();
}

// MediaPipe Setup & Camera Initialization
let currentFacingMode = 'user';
let camera = null;

function startCamera() {
    if (camera) {
        try { camera.stop(); } catch(e) {}
    }

    camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({ image: videoElement });
        },
        width: 640,
        height: 480,
        facingMode: currentFacingMode
    });
    camera.start();
}

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);
startCamera();

// Flip Camera Button Event Listener
document.getElementById('btnFlipCamera').addEventListener('click', () => {
    currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    startCamera();
    showToast(`Switched to ${currentFacingMode === 'user' ? 'Front Selfie' : 'Rear Camera'}`, "🔄");
});


// Controls & Button Actions
document.getElementById('btnSpeakSentence').addEventListener('click', () => {
    const fullText = sentenceArray.join(' ');
    if (fullText) {
        speakText(fullText, true);
        playChime(659, 0.2);
    } else {
        showToast("Sentence buffer is empty!", "⚠️");
    }
});

document.getElementById('btnCopySentence').addEventListener('click', () => {
    const text = sentenceArray.join(' ');
    if (text) {
        navigator.clipboard.writeText(text).then(() => {
            showToast("Sentence copied to clipboard!", "📋");
            playChime(784, 0.15);
        });
    } else {
        showToast("No sentence to copy!", "⚠️");
    }
});

document.getElementById('btnDownloadTxt').addEventListener('click', () => {
    const text = sentenceArray.join(' ');
    if (!text) {
        showToast("No text to download!", "⚠️");
        return;
    }
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sign_translation_${Date.now()}.txt`;
    a.click();
    showToast("Downloaded sentence file!", "💾");
});

document.getElementById('btnShareWhatsApp').addEventListener('click', () => {
    const text = sentenceArray.join(' ');
    if (!text) {
        showToast("No text to share!", "⚠️");
        return;
    }
    const waUrl = `https://api.whatsapp.com/send?text=${encodeURIComponent("Sign Language Translation: " + text)}`;
    window.open(waUrl, '_blank');
});

document.getElementById('btnAddSpace').addEventListener('click', () => {
    sentenceArray.push(' ');
    updateSentenceDisplay();
});

document.getElementById('btnBackspace').addEventListener('click', () => {
    sentenceArray.pop();
    updateSentenceDisplay();
});

document.getElementById('btnClearSentence').addEventListener('click', () => {
    sentenceArray = [];
    lastAddedText = "";
    lastSpokenText = "";
    smoother.reset();
    updateSentenceDisplay();
    showToast("Sentence cleared!", "🗑️");
});

document.getElementById('btnClearHistory').addEventListener('click', () => {
    historyItems = [];
    historyListEl.innerHTML = `<li class="timeline-item" style="color: var(--text-muted);">No gestures translated yet</li>`;
    showToast("Timeline log cleared!", "📜");
});

// Dictionary Modal & Category Filter Tabs
function renderDictionaryGrid() {
    const filtered = activeCategory === 'ALL' ? gestureDict : gestureDict.filter(g => g.category.includes(activeCategory));
    dictGridEl.innerHTML = filtered.map(g => `
        <div class="dict-card">
            <div class="dict-name">${g.name}</div>
            <div class="dict-cat-tag">${g.category}</div>
            <div class="dict-description">${g.description}</div>
            <div class="dict-tip">💡 ${g.tip}</div>
        </div>
    `).join('');
}

function loadGesturesData(callback) {
    fetch('/api/gestures')
        .then(res => res.json())
        .then(data => {
            gestureDict = data.gestures || [];
            if (callback) callback();
        });
}

document.getElementById('btnDictionary').addEventListener('click', () => {
    loadGesturesData(() => {
        renderDictionaryGrid();
        dictionaryModalEl.classList.add('active');
    });
});

document.getElementById('btnCloseDict').addEventListener('click', () => {
    dictionaryModalEl.classList.remove('active');
});

document.getElementById('categoryTabs').addEventListener('click', (e) => {
    if (e.target.classList.contains('tab-btn')) {
        document.querySelectorAll('#categoryTabs .tab-btn').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
        activeCategory = e.target.getAttribute('data-cat');
        renderDictionaryGrid();
    }
});

// --- Gamified Quiz Game Mode ---
function nextQuizTarget() {
    if (!gestureDict || gestureDict.length === 0) return;
    quizTarget = gestureDict[Math.floor(Math.random() * gestureDict.length)];
    quizTargetTextEl.textContent = `Show "${quizTarget.name}"`;
    quizScoreTextEl.textContent = `Score: ${quizScore} | Streak: ${quizStreak}`;
    quizTimer = 10;
    quizTimerTextEl.textContent = `⏳ ${quizTimer}s`;
}

document.getElementById('btnQuizGame').addEventListener('click', () => {
    isQuizActive = !isQuizActive;
    const btn = document.getElementById('btnQuizGame');
    if (isQuizActive) {
        btn.textContent = "🛑 Stop Game";
        quizScore = 0;
        quizStreak = 0;
        loadGesturesData(() => {
            quizHudPanelEl.style.display = 'flex';
            nextQuizTarget();
            quizInterval = setInterval(() => {
                quizTimer -= 1;
                quizTimerTextEl.textContent = `⏳ ${quizTimer}s`;
                if (quizTimer <= 0) {
                    quizStreak = 0;
                    showToast(`Time up! Target was ${quizTarget.name}`, "⌛");
                    nextQuizTarget();
                }
            }, 1000);
            showToast("🎮 Sign Language Quiz Game Started!", "🎯");
        });
    } else {
        btn.textContent = "🎮 Quiz Game";
        quizHudPanelEl.style.display = 'none';
        if (quizInterval) clearInterval(quizInterval);
        showToast(`Quiz Game Ended! Final Score: ${quizScore}`, "🏆");
    }
});

// --- Reverse Speech-to-Sign Visualizer ---
const micBtn = document.getElementById('micBtn');
const speechVisStatus = document.getElementById('speechVisStatus');
const speechFlashcardGrid = document.getElementById('speechFlashcardGrid');
let recognition = null;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRec();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        micBtn.classList.add('listening');
        speechVisStatus.textContent = "Listening... Speak now!";
    };

    recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        speechVisStatus.textContent = `You Spoke: "${transcript}"`;
        micBtn.classList.remove('listening');

        // Render flashcards for recognized words
        const words = transcript.split(' ');
        loadGesturesData(() => {
            speechFlashcardGrid.innerHTML = words.map(w => {
                const match = gestureDict.find(g => g.name.toLowerCase().includes(w.toLowerCase()));
                return `
                    <div class="flashcard-item">
                        <div class="flashcard-emoji">${match ? '🤟' : '💬'}</div>
                        <div class="flashcard-word">${w}</div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">${match ? match.description : 'Fingerspell each letter'}</div>
                    </div>
                `;
            }).join('');
        });
    };

    recognition.onerror = (e) => {
        micBtn.classList.remove('listening');
        speechVisStatus.textContent = `Speech error: ${e.error}`;
    };

    recognition.onend = () => {
        micBtn.classList.remove('listening');
    };
}

micBtn.addEventListener('click', () => {
    if (recognition) {
        recognition.start();
    } else {
        speechVisStatus.textContent = "Web Speech Recognition not supported in this browser.";
    }
});

document.getElementById('btnSpeechToSign').addEventListener('click', () => {
    speechToSignModalEl.classList.add('active');
});
document.getElementById('btnCloseSpeechToSign').addEventListener('click', () => {
    speechToSignModalEl.classList.remove('active');
});

// --- Custom Gesture Recorder ---
document.getElementById('btnRecordCustom').addEventListener('click', () => {
    recordCustomModalEl.classList.add('active');
});
document.getElementById('btnCloseRecordCustom').addEventListener('click', () => {
    recordCustomModalEl.classList.remove('active');
});

document.getElementById('btnStartRecording').addEventListener('click', () => {
    const name = document.getElementById('customSignName').value.trim();
    const tip = document.getElementById('customSignTip').value.trim();

    if (!name) {
        showToast("Please enter a gesture name!", "⚠️");
        return;
    }

    if (!currentLandmarksPayload) {
        showToast("No hand detected on camera to record!", "⚠️");
        return;
    }

    fetch('/api/save_custom_sign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: name,
            category: 'Custom',
            description: `User recorded custom sign: ${name}`,
            tip: tip || 'Custom hand landmark position',
            landmarks: currentLandmarksPayload
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            showToast(`Recorded Custom Sign "${name}"!`, "🎉");
            recordCustomModalEl.classList.remove('active');
            document.getElementById('customSignName').value = "";
            document.getElementById('customSignTip').value = "";
        }
    });
});

// Shortcuts
window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') {
        e.preventDefault();
        document.getElementById('btnAddSpace').click();
    } else if (e.code === 'Backspace') {
        document.getElementById('btnBackspace').click();
    } else if (e.key.toLowerCase() === 'c') {
        document.getElementById('btnClearSentence').click();
    } else if (e.key.toLowerCase() === 's') {
        document.getElementById('btnSpeakSentence').click();
    }
});
