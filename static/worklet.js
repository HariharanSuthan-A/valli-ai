/**
 * worklet.js – AudioWorklet processor that captures PCM16 mono at 16 kHz.
 *
 * The worklet receives float32 samples at the browser's native sample rate.
 * It downsamples to 16 kHz, converts to PCM16 (little-endian int16), and
 * posts chunks of ~20 ms (320 samples at 16 kHz) to the main thread.
 *
 * Also performs a simple energy-based VAD: posts {type:"energy",value:N}
 * messages so the main thread can hint at barge-in.
 */
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Target sample rate (must match server: 16 kHz)
    this._targetRate = (options.processorOptions || {}).targetRate || 16000;
    // Resample ratio: how many input frames per output frame
    this._ratio = sampleRate / this._targetRate; // sampleRate = global in worklet
    this._accumulator = 0;
    this._outBuf = [];
    // Chunk size: 20 ms at 16 kHz = 320 samples
    this._chunkSize = Math.round(this._targetRate * 0.02);
    this._active = true;

    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'stop') {
        this._active = false;
      }
    };
  }

  process(inputs) {
    if (!this._active) return false;

    const channel = inputs[0] && inputs[0][0];
    if (!channel || channel.length === 0) return true;

    // Compute energy for VAD hint
    let sumSq = 0;
    for (let i = 0; i < channel.length; i++) sumSq += channel[i] * channel[i];
    const rms = Math.sqrt(sumSq / channel.length);
    this.port.postMessage({ type: 'energy', value: rms });

    // Nearest-neighbour downsampling + PCM16 conversion
    for (let i = 0; i < channel.length; i++) {
      this._accumulator += 1;
      if (this._accumulator >= this._ratio) {
        this._accumulator -= this._ratio;
        // Clamp and convert float32 to int16
        const s = Math.max(-1, Math.min(1, channel[i]));
        this._outBuf.push(Math.round(s < 0 ? s * 32768 : s * 32767));
      }

      if (this._outBuf.length >= this._chunkSize) {
        const pcm = new Int16Array(this._outBuf.splice(0, this._chunkSize));
        this.port.postMessage({ type: 'pcm', buffer: pcm.buffer }, [pcm.buffer]);
      }
    }

    return true;
  }
}

registerProcessor('pcm-capture', PcmCapture);

