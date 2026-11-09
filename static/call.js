/**
 * call.js – Main browser call logic.
 *
 * Responsibilities:
 *  - Request mic permission (echoCancellation + noiseSuppression)
 *  - Load AudioWorklet, capture PCM16 at 16 kHz
 *  - Connect WebSocket to /ws/call
 *  - Play ringing sound, then connect
 *  - Forward mic PCM to server as binary frames
 *  - Receive JSON control messages and binary audio chunks
 *  - Manage playback queue; flush on stop_audio (barge-in)
 *  - Render call UI: state, captions, service card, timer, mute, end
 *
 * No external libraries, no build step.
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let ws = null;
let audioCtx = null;
let workletNode = null;
let mediaStream = null;
let isMuted = false;
let isCaptionsOn = true;
let callStartTime = null;
let timerInterval = null;
let isEnded = false;
let isStarting = false;
let currentLanguage = 'ta-IN';
let currentServerState = 'idle';

const UI_TEXTS = {
  'ta-IN': {
    visitPrompt: 'அதிகாரப்பூர்வ பக்கத்தைப் பார்வையிட கீழே உள்ள பொத்தானைக் கிளிக் செய்யவும்',
    visitBtn: '🌐 அதிகாரப்பூர்வ இணையதளம்',
    thanksEnded: 'வள்ளியைத் தொடர்பு கொண்டதற்கு நன்றி! மேலும் விவரங்கள் தேவைப்பட்டால் என்னைத் தொடர்பு கொள்ளவும்.',
  },
  'hi-IN': {
    visitPrompt: 'आधिकारिक पृष्ठ पर जाने के लिए नीचे दिए गए बटन पर क्लिक करें',
    visitBtn: '🌐 आधिकारिक वेबसाइट',
    thanksEnded: 'वल्ली से संपर्क करने के लिए धन्यवाद! यदि आपको कोई और विवरण चाहिए तो मुझसे संपर्क करें।',
  },
  'te-IN': {
    visitPrompt: 'అధికారిక పేజీని సందర్శించడానికి క్రింది బటన్‌ను క్లిక్ చేయండి',
    visitBtn: '🌐 అధికారిక వెబ్‌సైట్',
    thanksEnded: 'వల్లిని సంప్రదించినందుకు ధన్యవాదాలు! మీకు ఏవైనా మరిన్ని వివరాలు కావాలంటే నన్ను సంప్రదించండి.',
  },
  'ml-IN': {
    visitPrompt: 'ഔദ്യോഗിക പേജ് സന്ദർശിക്കാൻ താഴെയുള്ള ബട്ടണിൽ ക്ലിക്കുചെയ്യുക',
    visitBtn: '🌐 ഔദ്യോഗിക വെബ്സൈറ്റ്',
    thanksEnded: 'വള്ളിയെ ബന്ധപ്പെട്ടതിന് നന്ദി! കൂടുതൽ വിവരങ്ങൾ ആവശ്യമുണ്ടെങ്കിൽ എന്നെ ബന്ധപ്പെടുക.',
  },
  'kn-IN': {
    visitPrompt: 'ಅಧಿಕೃತ ಪುಟಕ್ಕೆ ಭೇಟಿ ನೀಡಲು ಕೆಳಗಿನ ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ',
    visitBtn: '🌐 ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್',
    thanksEnded: 'ವಳ್ಳಿಯನ್ನು ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು! ಹೆಚ್ಚಿನ ವಿವರಗಳು ಬೇಕಾದರೆ ನನ್ನನ್ನು ಸಂಪರ್ಕಿಸಿ.',
  },
};

function onLanguageChange(lang) {
  currentLanguage = lang;
  const sel1 = document.getElementById('lang-select');
  const sel2 = document.getElementById('call-lang-select');
  if (sel1) sel1.value = lang;
  if (sel2) sel2.value = lang;

  const t = UI_TEXTS[lang] || UI_TEXTS['ta-IN'];

  // Update ended screen message according to selected language
  const endedMsg = document.getElementById('ended-thanks-msg');
  if (endedMsg) {
    endedMsg.textContent = t.thanksEnded;
  }

  // Update service card texts if active
  const promptEl = document.getElementById('service-card-prompt');
  const siteBtn = document.getElementById('service-site-btn');
  if (promptEl) promptEl.textContent = t.visitPrompt;
  if (siteBtn) siteBtn.textContent = t.visitBtn;

  // Notify server if call WebSocket is active
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_language', language: lang }));
  }
}

// ── Playback state ──────────────────────────────────────────────────────────
let playbackQueue = [];          // [{decodePromise, text, gen}]
let isPlaying = false;           // audio is scheduled / currently audible
let hasReceivedAudioEnd = false; // server finished sending this turn
let pendingServerState = null;
let pendingAudioMeta = null;     // {format, rate, text}
let currentTurnAgentText = '';
let nextStartTime = 0;           // AudioContext clock time for next chunk
let playbackGeneration = 0;      // bumped on barge-in to cancel stale work
let finishTimer = null;
let playbackPumpRunning = false;
const liveSources = new Set();

// Barge-in: sustained mic energy while the agent is active interrupts it.
let noiseFloor = 0.005;
let speechFrames = 0;
let bargeInCooldownUntil = 0;
let bargeInArmedAt = 0;          // ignore the caller's own speech tail
let agentAudioStartedAt = 0;
const ENERGY_THRESHOLD = 0.03;   // minimum RMS; scaled by ambient noise floor

function hasPendingAgentAudio() {
  return isPlaying || playbackQueue.length > 0;
}

// ── UI helpers ──────────────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(el => {
    el.classList.remove('active');
    el.classList.add('hidden');
  });
  const s = document.getElementById(id);
  s.classList.remove('hidden');
  s.classList.add('active');
}

function setCallStatus(text, cls) {
  const el = document.getElementById('call-status');
  el.textContent = text;
  el.className = 'call-status' + (cls ? ' ' + cls : '');
}

function startTimer() {
  callStartTime = Date.now();
  timerInterval = setInterval(() => {
    const secs = Math.floor((Date.now() - callStartTime) / 1000);
    const m = Math.floor(secs / 60);
    const s = String(secs % 60).padStart(2, '0');
    document.getElementById('call-timer').textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ── Ringtone ───────────────────────────────────────────────────────────────
function playRing() {
  const ring = document.getElementById('ring-audio');
  if (ring) {
    ring.volume = 0.5;
    ring.play().catch(() => {});
  }
}

function stopRing() {
  const ring = document.getElementById('ring-audio');
  if (ring) {
    ring.pause();
    ring.currentTime = 0;
  }
}

let playbackPumpId = 0;

// ── Audio playback scheduler ────────────────────────────────────────────────
// Each incoming chunk is decoded immediately and scheduled back-to-back on the
// AudioContext clock.  Scheduling ahead (rather than waiting for the previous
// chunk's `onended` before decoding the next) removes the audible gaps that
// made the agent sound choppy.
function enqueueAudio(arrayBuffer, sampleRate, text) {
  if (!audioCtx) return;
  const gen = playbackGeneration;
  const decodePromise = audioCtx
    .decodeAudioData(arrayBuffer.slice(0))
    .catch((e) => {
      console.warn('Audio decode error:', e);
      return null;
    });

  if (!isPlaying) agentAudioStartedAt = performance.now();
  playbackQueue.push({ decodePromise, text: text || '', gen });
  isPlaying = true;
  setCallStatus('Speaking…', 'speaking');
  startPlaybackPump();
}

function appendAgentCaption(text) {
  if (!isCaptionsOn || !text) return;
  if (currentTurnAgentText.length > 0) {
    currentTurnAgentText += ' ' + text;
  } else {
    currentTurnAgentText = text;
  }
  document.getElementById('agent-caption-text').textContent = currentTurnAgentText;
  const area = document.getElementById('captions-area');
  if (area) {
    area.scrollTop = area.scrollHeight;
  }
}

function startPlaybackPump() {
  if (playbackPumpRunning) return;
  playbackPumpRunning = true;
  const myPump = ++playbackPumpId;

  void (async () => {
    try {
      while (playbackQueue.length > 0) {
        if (myPump !== playbackPumpId) break;
        const item = playbackQueue.shift();
        const decoded = await item.decodePromise;

        if (myPump !== playbackPumpId) break;
        if (!audioCtx) break;
        if (item.gen !== playbackGeneration) continue;
        if (!decoded || !decoded.duration) continue;

        if (audioCtx.state === 'suspended') {
          try { await audioCtx.resume(); } catch (_) {}
        }

        const src = audioCtx.createBufferSource();
        src.buffer = decoded;
        src.connect(audioCtx.destination);

        // Start right after the previous chunk, with a small lead-in so we
        // never schedule in the past (which would clip the first sample).
        const startAt = Math.max(audioCtx.currentTime + 0.04, nextStartTime);
        nextStartTime = startAt + decoded.duration;

        liveSources.add(src);
        src.onended = () => liveSources.delete(src);

        // Caption appears in sync with the audio being spoken.
        if (item.text) {
          const delay = Math.max(0, (startAt - audioCtx.currentTime) * 1000);
          setTimeout(() => {
            if (item.gen === playbackGeneration) appendAgentCaption(item.text);
          }, delay);
        }

        src.start(startAt);
      }
    } catch (e) {
      console.warn('Playback pump error:', e);
    } finally {
      if (myPump === playbackPumpId) {
        playbackPumpRunning = false;
        scheduleFinishCheck();
      }
    }
  })();
}

function scheduleFinishCheck(extraDelayMs) {
  if (finishTimer) {
    clearTimeout(finishTimer);
    finishTimer = null;
  }
  if (playbackQueue.length > 0 || playbackPumpRunning) return;

  const now = audioCtx ? audioCtx.currentTime : 0;
  const remainingMs = Math.max(0, (nextStartTime - now) * 1000);
  finishTimer = setTimeout(finalizePlaybackIfIdle, remainingMs + (extraDelayMs || 120));
}

function finalizePlaybackIfIdle() {
  finishTimer = null;
  if (playbackQueue.length > 0 || playbackPumpRunning) return;

  // The timer can fire a few milliseconds early; wait for the real tail.
  if (audioCtx && audioCtx.currentTime < nextStartTime - 0.02) {
    scheduleFinishCheck(40);
    return;
  }

  if (!hasReceivedAudioEnd) {
    // The server may still be synthesising the next segment.  Keep "Speaking…"
    // rather than briefly flashing back to Listening between segments.
    scheduleFinishCheck(250);
    return;
  }

  isPlaying = false;
  if (!isEnded) setCallStatus('Listening…', 'listening');
  sendPlaybackDone();
}

function sendPlaybackDone() {
  hasReceivedAudioEnd = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'playback_done' }));
  }
}

function flushPlayback() {
  // Cancel any in-flight decode/schedule work and stop audible chunks.
  playbackGeneration += 1;
  playbackPumpId += 1;
  playbackPumpRunning = false;
  playbackQueue = [];
  hasReceivedAudioEnd = false;
  pendingServerState = null;
  pendingAudioMeta = null;
  currentTurnAgentText = '';

  if (finishTimer) {
    clearTimeout(finishTimer);
    finishTimer = null;
  }

  for (const src of liveSources) {
    try { src.stop(); } catch (_) {}
  }
  liveSources.clear();

  isPlaying = false;
  agentAudioStartedAt = 0;
  nextStartTime = audioCtx ? audioCtx.currentTime : 0;

  if (!isEnded) setCallStatus('Listening…', 'listening');
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume().catch(() => {});
}

// ── Barge-in detection ──────────────────────────────────────────────────────
// Mic energy is measured by the worklet.  If the caller speaks with sustained
// energy while the agent is thinking or speaking, we stop playback locally and
// tell the server to cancel the in-flight turn.
function maybeDetectBargeIn(energy) {
  if (isMuted || isEnded || !Number.isFinite(energy)) {
    speechFrames = 0;
    return;
  }

  const agentActive =
    hasPendingAgentAudio() ||
    currentServerState === 'thinking' ||
    currentServerState === 'speaking';

  if (!agentActive) {
    // Adapt to ambient noise only while the agent is quiet, so the agent's own
    // voice (leakage) cannot raise the threshold.
    noiseFloor = noiseFloor * 0.97 + energy * 0.03;
    speechFrames = 0;
    return;
  }

  // Ignore the brief window right after the question, when the caller's own
  // voice tail is still being captured.
  if (performance.now() < bargeInArmedAt) {
    speechFrames = 0;
    return;
  }

  // Ignore the click/tail right after playback starts.
  if (performance.now() - agentAudioStartedAt < 150) {
    speechFrames = 0;
    return;
  }

  const threshold = Math.max(ENERGY_THRESHOLD, noiseFloor * 2.2);
  if (energy >= threshold) {
    speechFrames += 1;
  } else {
    speechFrames = Math.max(0, speechFrames - 1);
  }

  // ~3 energy frames (~150 ms) of sustained speech = deliberate interruption.
  if (speechFrames >= 3 && performance.now() >= bargeInCooldownUntil) {
    triggerBargeIn();
  }
}

function triggerBargeIn() {
  bargeInCooldownUntil = performance.now() + 800;
  speechFrames = 0;
  currentServerState = 'listening';
  flushPlayback();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'bargein' }));
  }
}

// ── Service card ────────────────────────────────────────────────────────────
function showServiceCard(entry) {
  const card = document.getElementById('service-card');
  const promptEl = document.getElementById('service-card-prompt');
  const siteBtn = document.getElementById('service-site-btn');
  if (!card || !siteBtn) return;

  const t = UI_TEXTS[currentLanguage] || UI_TEXTS['ta-IN'];
  if (promptEl) promptEl.textContent = t.visitPrompt;
  siteBtn.textContent = t.visitBtn;

  if (entry && entry.official_site) {
    siteBtn.href = entry.official_site;
    siteBtn.style.display = 'inline-flex';
    card.classList.remove('hidden');
  } else {
    card.classList.add('hidden');
  }
}

// ── Caption update ──────────────────────────────────────────────────────────
function updateCaption(who, text) {
  if (!isCaptionsOn) return;
  if (who === 'user') {
    const row = document.getElementById('user-caption');
    row.classList.remove('hidden');
    document.getElementById('user-caption-text').textContent = text;
  } else {
    document.getElementById('agent-caption-text').textContent = text;
  }
  const area = document.getElementById('captions-area');
  if (area) {
    area.scrollTop = area.scrollHeight;
  }
}

// ── WebSocket connection ────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/call`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    isStarting = false;
    stopRing();
    setCallStatus('Connected', 'listening');
    ws.send(JSON.stringify({ type: 'start', language: currentLanguage }));
    startTimer();
  };

  ws.onmessage = (evt) => {
    if (typeof evt.data === 'string') {
      handleJSONMessage(JSON.parse(evt.data));
    } else {
      // Binary audio frame
      if (pendingAudioMeta) {
        enqueueAudio(evt.data, pendingAudioMeta.rate, pendingAudioMeta.text);
        pendingAudioMeta = null;
      }
    }
  };

  ws.onclose = () => {
    isStarting = false;
    if (!isEnded) handleCallEnded();
  };

  ws.onerror = (e) => {
    console.error('WS error', e);
    isStarting = false;
    stopRing();
    setCallStatus('Connection error');
  };
}

function handleJSONMessage(msg) {
  switch (msg.type) {
    case 'state':
      if (msg.value === 'thinking') {
        currentTurnAgentText = '';
        document.getElementById('agent-caption-text').textContent = '…';
      }
      onStateChange(msg.value);
      break;
    case 'audio':
      pendingAudioMeta = { format: msg.format, rate: msg.rate, text: msg.text || '' };
      break;
    case 'audio_end':
      hasReceivedAudioEnd = true;
      scheduleFinishCheck();
      break;
    case 'stop_audio':
      flushPlayback();
      break;
    case 'user_text':
      currentTurnAgentText = '';
      if (msg.final) updateCaption('user', msg.text);
      else updateCaption('user', msg.text + ' …');
      break;
    case 'agent_text':
      if (!isPlaying && !currentTurnAgentText) {
        updateCaption('agent', msg.text);
      }
      break;
    case 'service':
      showServiceCard(msg.entry);
      break;
    case 'error':
      console.error('Server error:', msg.message);
      setCallStatus('Error: ' + msg.message);
      break;
  }
}

function onStateChange(state) {
  currentServerState = state;
  pendingServerState = state;
  if (state === 'thinking') {
    // Do not let the caller's own speech tail cancel the turn that just began.
    bargeInArmedAt = performance.now() + 600;
  }
  // If server indicates 'listening' while the browser is still playing queued
  // speech, hold in Speaking… mode until playback actually finishes.
  if (state === 'listening' && hasPendingAgentAudio()) {
    setCallStatus('Speaking…', 'speaking');
    return;
  }
  pendingServerState = null;
  const labels = {
    idle: 'Idle',
    listening: 'Listening…',
    thinking: 'Thinking…',
    speaking: 'Speaking…',
    ended: 'Call Ended',
  };
  setCallStatus(labels[state] || state, state);
  if (state === 'ended') handleCallEnded();
}

// ── Mic + AudioWorklet setup ────────────────────────────────────────────────
async function setupMic() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      sampleRate: 44100,
    },
    video: false,
  });

  audioCtx = new AudioContext({ sampleRate: 44100 });
  if (audioCtx.state === 'suspended') {
    try { await audioCtx.resume(); } catch (_) {}
  }
  await audioCtx.audioWorklet.addModule('/worklet.js');

  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
    processorOptions: { targetRate: 16000 },
  });
  source.connect(workletNode);

  // Keep the graph rendering without any audible loopback: the worklet feeds a
  // zero-gain node that reaches the destination.
  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;
  workletNode.connect(silentGain);
  silentGain.connect(audioCtx.destination);

  workletNode.port.onmessage = (e) => {
    const { type, buffer, value } = e.data;
    if (type === 'pcm') {
      // Stream mic audio to the server unless the agent's own voice is
      // actually playing (avoids feeding echo into STT).  During THINKING we
      // still stream so the server can transcribe a barge-in.
      if (ws && ws.readyState === WebSocket.OPEN && !isMuted) {
        if (!hasPendingAgentAudio() || currentServerState === 'thinking') {
          ws.send(buffer);
        }
      }
    } else if (type === 'energy') {
      maybeDetectBargeIn(value);
    }
  };
}

// ── Public call actions ─────────────────────────────────────────────────────
async function startCall() {
  if (isStarting || (ws && ws.readyState === WebSocket.OPEN)) return;

  const errEl = document.getElementById('mic-error');
  errEl.classList.add('hidden');
  isStarting = true;

  try {
    await setupMic();
  } catch (err) {
    isStarting = false;
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    if (audioCtx) {
      audioCtx.close().catch(() => {});
      audioCtx = null;
    }
    errEl.textContent = '🎙️ Microphone access denied. Please allow mic access and try again.';
    errEl.classList.remove('hidden');
    return;
  }

  isEnded = false;
  showScreen('screen-call');
  setCallStatus('Connecting…');
  playRing();

  // Small delay to let the ring play before WS connects
  setTimeout(connectWS, 800);
}

function toggleMute() {
  isMuted = !isMuted;
  const btn = document.getElementById('btn-mute');
  const icon = document.getElementById('mute-icon');
  if (isMuted) {
    btn.classList.add('muted');
    icon.textContent = '🔇';
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'mute', on: true }));
    }
  } else {
    btn.classList.remove('muted');
    icon.textContent = '🎙️';
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'mute', on: false }));
    }
  }
}

function toggleCaptions() {
  isCaptionsOn = !isCaptionsOn;
  const area = document.getElementById('captions-area');
  area.style.opacity = isCaptionsOn ? '1' : '0.2';
  document.getElementById('btn-captions').style.opacity = isCaptionsOn ? '1' : '0.5';
}

function endCall() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'end' }));
    ws.close();
  }
  handleCallEnded();
}

function handleCallEnded() {
  if (isEnded) return;
  isEnded = true;
  isStarting = false;
  stopTimer();
  stopRing();
  flushPlayback();

  if (workletNode) {
    workletNode.port.postMessage({ type: 'stop' });
    workletNode.disconnect();
    workletNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }

  setTimeout(() => showScreen('screen-ended'), 300);
}

function resetUI() {
  isEnded = false;
  isStarting = false;
  isMuted = false;
  isCaptionsOn = true;
  ws = null;
  currentServerState = 'idle';
  pendingAudioMeta = null;
  playbackQueue = [];
  playbackGeneration += 1;
  playbackPumpId += 1;
  playbackPumpRunning = false;
  if (finishTimer) {
    clearTimeout(finishTimer);
    finishTimer = null;
  }
  liveSources.clear();
  isPlaying = false;
  hasReceivedAudioEnd = false;
  pendingServerState = null;
  currentTurnAgentText = '';
  nextStartTime = 0;
  agentAudioStartedAt = 0;
  noiseFloor = 0.005;
  speechFrames = 0;
  bargeInCooldownUntil = 0;
  bargeInArmedAt = 0;

  document.getElementById('service-card').classList.add('hidden');
  document.getElementById('user-caption').classList.add('hidden');
  document.getElementById('agent-caption-text').textContent = '…';
  document.getElementById('call-timer').textContent = '0:00';

  showScreen('screen-idle');
}

