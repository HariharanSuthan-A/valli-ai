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

// Playback queue
let playbackQueue = [];
let isPlaying = false;
let hasReceivedAudioEnd = false;
let pendingServerState = null;
let pendingAudioMeta = null; // {format, rate, text}
let currentTurnAgentText = '';

// Energy VAD: track recent energy for barge-in hint
let recentEnergy = 0;
const ENERGY_THRESHOLD = 0.015; // tune to mic level

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

let currentSourceNode = null;

// ── Audio playback queue ────────────────────────────────────────────────────
async function enqueueAudio(arrayBuffer, sampleRate, text) {
  if (!audioCtx) return;
  playbackQueue.push({ arrayBuffer, sampleRate, text: text || '' });
  if (!isPlaying) playNextChunk();
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

async function playNextChunk() {
  if (playbackQueue.length === 0) {
    isPlaying = false;
    currentSourceNode = null;
    // Transition to Listening as soon as speech finishes playing
    if (pendingServerState === 'listening') {
      pendingServerState = null;
      setCallStatus('Listening…', 'listening');
    }
    if (hasReceivedAudioEnd) {
      sendPlaybackDone();
    }
    return;
  }
  isPlaying = true;
  setCallStatus('Speaking…', 'speaking');
  const { arrayBuffer, sampleRate, text } = playbackQueue.shift();

  // Display text in real time matching exactly what is currently being spoken
  if (text) {
    appendAgentCaption(text);
  }

  try {
    // Decode WAV
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const src = audioCtx.createBufferSource();
    src.buffer = decoded;
    src.connect(audioCtx.destination);
    currentSourceNode = src;
    src.onended = () => {
      if (currentSourceNode === src) {
        currentSourceNode = null;
        playNextChunk();
      }
    };
    src.start();
  } catch (e) {
    console.warn('Audio decode error:', e);
    playNextChunk();
  }
}

function sendPlaybackDone() {
  hasReceivedAudioEnd = false;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'playback_done' }));
  }
}

function flushPlayback() {
  playbackQueue = [];
  isPlaying = false;
  hasReceivedAudioEnd = false;
  pendingServerState = null;
  currentTurnAgentText = '';
  currentTurnAgentText = '';
  if (currentSourceNode) {
    try {
      currentSourceNode.stop();
    } catch (_) {}
    currentSourceNode = null;
  }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}

// ── Service card ────────────────────────────────────────────────────────────
function showServiceCard(entry) {
  const card = document.getElementById('service-card');
  document.getElementById('service-name-en').textContent = entry.name_en || '';
  document.getElementById('service-name-ta').textContent = entry.name_ta ? `(${entry.name_ta})` : '';

  const docsList = document.getElementById('docs-list');
  docsList.innerHTML = '';
  (entry.documents || []).slice(0, 6).forEach(doc => {
    const li = document.createElement('li');
    li.textContent = doc;
    docsList.appendChild(li);
  });

  const siteBtn = document.getElementById('service-site-btn');
  if (entry.official_site) {
    siteBtn.href = entry.official_site;
    siteBtn.style.display = '';
  } else {
    siteBtn.style.display = 'none';
  }

  card.classList.remove('hidden');
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
    stopRing();
    setCallStatus('Connected', 'listening');
    ws.send(JSON.stringify({ type: 'start' }));
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
    if (!isEnded) handleCallEnded();
  };

  ws.onerror = (e) => {
    console.error('WS error', e);
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
      if (!isPlaying && playbackQueue.length === 0) {
        sendPlaybackDone();
      }
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
  pendingServerState = state;
  // If server indicates 'listening', but browser is still playing queued speech,
  // hold in Speaking… mode until playback finishes.
  if (state === 'listening' && (isPlaying || playbackQueue.length > 0)) {
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
  await audioCtx.audioWorklet.addModule('/worklet.js');

  const source = audioCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture', {
    processorOptions: { targetRate: 16000 },
  });
  source.connect(workletNode);
  // Don't connect workletNode to destination – we don't want loopback

  workletNode.port.onmessage = (e) => {
    const { type, buffer, value } = e.data;
    if (type === 'pcm') {
      // Do not stream microphone audio while speech is playing
      if (ws && ws.readyState === WebSocket.OPEN && !isMuted && !isPlaying && playbackQueue.length === 0) {
        ws.send(buffer);
      }
    } else if (type === 'energy') {
      recentEnergy = value;
    }
  };
}

// ── Public call actions ─────────────────────────────────────────────────────
async function startCall() {
  const errEl = document.getElementById('mic-error');
  errEl.classList.add('hidden');

  try {
    await setupMic();
  } catch (err) {
    errEl.textContent = '🎙️ Microphone access denied. Please allow mic access and try again.';
    errEl.classList.remove('hidden');
    return;
  }

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
  isMuted = false;
  isCaptionsOn = true;
  ws = null;
  pendingAudioMeta = null;
  playbackQueue = [];
  isPlaying = false;
  hasReceivedAudioEnd = false;
  pendingServerState = null;
  currentTurnAgentText = '';

  document.getElementById('service-card').classList.add('hidden');
  document.getElementById('user-caption').classList.add('hidden');
  document.getElementById('agent-caption-text').textContent = '…';
  document.getElementById('call-timer').textContent = '0:00';

  showScreen('screen-idle');
}

