export class RingBuffer {
    constructor() { this.buf = new Float32Array(0); }
    push(arr) {
        if (!arr || !arr.length) return;
        const out = new Float32Array(this.buf.length + arr.length);
        out.set(this.buf, 0);
        out.set(arr, this.buf.length);
        this.buf = out;
    }
    shift(n) {
        if (n <= 0 || this.buf.length === 0) return new Float32Array(0);
        const take = Math.min(n, this.buf.length);
        const head = this.buf.subarray(0, take);
        const rest = this.buf.subarray(take);
        const out = new Float32Array(rest.length);
        out.set(rest, 0);
        this.buf = out;
        return head;
    }
    length() { return this.buf.length; }
}

const sinc = (x) => (x === 0 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x));
const designLowpassFIR = (cutoffHz, sampleRate, taps) => {
    const fc = cutoffHz / (sampleRate / 2);
    const M = taps - 1;
    const h = new Float32Array(taps);
    for (let n = 0; n < taps; n++) {
        const k = n - M / 2;
        const ideal = 2 * fc * sinc(2 * fc * k);
        const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / M);
        h[n] = ideal * w;
    }
    const sum = h.reduce((a, b) => a + b, 0);
    for (let n = 0; n < taps; n++) h[n] /= sum || 1;
    return h;
};

export class AudioResampler {
    constructor(inRate, outRate, withLPF = true) {
        this.taps = withLPF ? 64 : 0;
        const cutoff = withLPF ? Math.min(7900, 0.45 * outRate) : 0;
        this.h = withLPF ? designLowpassFIR(cutoff, inRate, this.taps) : null;
        this.inRate = inRate;
        this.outRate = outRate;
        this.withLPF = withLPF;
        this.hist = withLPF ? new Float32Array(this.taps) : null;
        this.pos = 0;
        this.prev = 0;
        this.ratio = inRate / outRate;
    }
    process(block) {
        let y;
        if (this.withLPF) {
            y = new Float32Array(block.length);
            for (let n = 0; n < block.length; n++) {
                for (let k = this.taps - 1; k > 0; k--) this.hist[k] = this.hist[k - 1];
                this.hist[0] = block[n];
                let acc = 0;
                for (let k = 0; k < this.taps; k++) acc += this.h[k] * this.hist[k];
                y[n] = acc;
            }
        } else {
            y = block;
        }
        const proc = new Float32Array(y.length + 1);
        proc[0] = this.prev;
        proc.set(y, 1);
        const out = [];
        while (this.pos <= proc.length - 2) {
            const i = Math.floor(this.pos);
            const frac = this.pos - i;
            const s0 = proc[i];
            const s1 = proc[i + 1];
            out.push(s0 + (s1 - s0) * frac);
            this.pos += this.ratio;
        }
        this.pos -= (proc.length - 1);
        this.prev = proc[proc.length - 1];
        return new Float32Array(out);
    }
}

