class VadProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = null;
  }

  // Merge all input channels to mono and post a transferable buffer
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const numChannels = input.length;
    const chan0 = input[0];
    const len = chan0.length;
    const out = new Float32Array(len);
    if (numChannels === 1) {
      out.set(chan0);
    } else {
      for (let i = 0; i < len; i++) {
        let acc = 0;
        for (let ch = 0; ch < numChannels; ch++) acc += input[ch][i];
        out[i] = acc / numChannels;
      }
    }
    this.port.postMessage({ type: 'audio', buffer: out.buffer }, [out.buffer]);
    return true;
  }
}

registerProcessor('vad-processor', VadProcessor);
