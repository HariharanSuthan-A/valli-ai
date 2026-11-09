/**
 * worklet.js – AudioWorklet processor that captures PCM16 mono at 16 kHz.
 *
 * The worklet receives float32 samples at the browser's native sample rate.
 * It downsamples to 16 kHz (box-filtered to reduce aliasing), converts to
 * PCM16 (little-endian int16), and posts 32 ms chunks (512 samples / 1024
 * bytes) to the main thread – exactly the frame size Prisma expects.
 *
 * It also posts a throttled RMS energy value (~every 50 ms) used by the main
 * thread for barge-in detection.
 */
class PcmCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Target sample rate (must match server: 16 kHz)
    this._targetRate = (options.processorOptions || {}).targetRate || 16000;
    // Resample ratio: how many input frames per output frame
    this._ratio = sampleRate / this._targetRate; // sampleRate = global in worklet
    this._accumulator = 0;
    this._sum = 0;
    this._count = 0;
    this._outBuf = [];
    // Chunk size: 32 ms at 16 kHz = 512 samples = 1024 bytes (Prisma frame)
    this._chunkSize = Math.round(this._targetRate * 0.032);
    this._active = true;

    // Throttle energy messages to ~50 ms (the previous per-block rate was
    // ~340 messages/second and wasted main-thread time).
    this._blockCount = 0;
    this._energyEvery = Math.max(1, Math.round((sampleRate * 0.05) / 128));

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

    // Throttled energy for the barge-in hint
    this._blockCount += 1;
    if (this._blockCount >= this._energyEvery) {
      this._blockCount = 0;
      let sumSq = 0;
      for (let i = 0; i < channel.length; i++) sumSq += channel[i] * channel[i];
      this.port.postMessage({
        type: 'energy',
        value: Math.sqrt(sumSq / channel.length),
      });
    }

    // Box-filtered downsampling + PCM16 conversion
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this._sum += s;
      this._count += 1;
      this._accumulator += 1;

      if (this._accumulator >= this._ratio) {
        this._accumulator -= this._ratio;
        const avg = this._count ? this._sum / this._count : s;
        this._sum = 0;
        this._count = 0;
        this._outBuf.push(Math.round(avg < 0 ? avg * 32768 : avg * 32767));
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
