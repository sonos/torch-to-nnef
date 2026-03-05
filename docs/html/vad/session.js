import { AudioResampler, RingBuffer } from './audio.js';

export class VadSession {
    constructor(wasm, plot, modes, stats) {
        this.wasm = wasm;
        this.plot = plot;
        this.modes = modes;
        this.stats = stats;
    }
    setMode(m) { this.modes?.set(m); }
    async prepareForFile() {
        // stop mic handled by page’s live manager if needed; caller ensures it
        // Do not reset WASM here; keeping warm state from mic avoids cold-start issues.
        this.modes?.disable(true);
        try { this.plot.clearTitle?.(); } catch { }
        this.plot.reset();
        this.plot.show();
        // Ensure legend reflects current mode after reset
        try { this.plot.setMode?.(this.modes?.get?.() || 'pulsed'); } catch { }
    }
    async finalizeAfterFile() {
        this.modes?.disable(false);
    }
    async runFile(file, desiredSampleRate = 16000, chunk = 4 * 160) {
        if (!file) return;
        this.stats?.setFileText?.('decoding...');
        const arrayBuf = await file.arrayBuffer();
        const AC = window.OfflineAudioContext || window.webkitOfflineAudioContext || window.AudioContext;
        const tmp = new (window.AudioContext || window.webkitAudioContext)();
        let audioBuf;
        try { audioBuf = await tmp.decodeAudioData(arrayBuf.slice(0)); }
        finally { tmp.close(); }
        // mono merge
        const n = audioBuf.length;
        const chs = audioBuf.numberOfChannels;
        let mono = new Float32Array(n);
        for (let ch = 0; ch < chs; ch++) {
            const d = audioBuf.getChannelData(ch);
            for (let i = 0; i < n; i++) mono[i] += d[i] / chs;
        }
        // resample
        const rs = new AudioResampler(audioBuf.sampleRate, desiredSampleRate, true);
        const block = 4096;
        let out = [];
        for (let i = 0; i < mono.length; i += block) {
            const sl = mono.subarray(i, Math.min(i + block, mono.length));
            const y = rs.process(sl);
            if (y && y.length) out.push(y);
        }
        const resampled = out.length ? new Float32Array(out.reduce((a, b) => a + b.length, 0)) : new Float32Array(0);
        if (out.length) {
            let off = 0; for (const y of out) { resampled.set(y, off); off += y.length; }
        }
        // Keep raw resampled audio; live path also sends raw 16k frames by default.
        // Determine mode once and warm up pulsed with zeros (matches the known-good mic-prewarm effect)
        const mode = this.modes?.get?.() || 'pulsed';
        if (mode !== 'batch') {
            try {
                const zero = new Float32Array(chunk);
                for (let k = 0; k < 128; k++) this.wasm.predictPulsed(zero);
            } catch { }
        }

        // decode in chunks, starting after warmupOffset (if any)
        let probs = [];
        let t = 0;
        const total = resampled.length;
        let lastPaint = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        const paintEveryMs = 16; // ~60 FPS target for smoother progress
        for (let i = 0; i < resampled.length; i += chunk) {
            const sl = resampled.subarray(i, Math.min(i + chunk, resampled.length));
            let buf = new Float32Array(chunk);
            buf.set(sl, 0);
            let pP = NaN, pB = NaN;
            if (mode === 'both') {
                pP = this.wasm.predictPulsed(buf);
                pB = this.wasm.predictBatch(buf);
                probs.push(Number.isFinite(pP) ? pP : (Number.isFinite(pB) ? pB : NaN));
            } else if (mode === 'batch') {
                pB = this.wasm.predictBatch(buf); probs.push(pB);
            } else {
                pP = this.wasm.predictPulsed(buf); probs.push(pP);
            }
            t = (i / desiredSampleRate) * 1000;
            this.plot.push(t, pP, pB, mode);
            const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
            if (now - lastPaint >= paintEveryMs) {
                this.plot.render();
                const done = Math.min(i + chunk, total);
                const pct = Math.min(100, Math.round((done / total) * 100));
                this.stats?.setFileText?.(`progress: ${pct}%`);
                // Yield to the browser so it can paint updates
                await new Promise(requestAnimationFrame);
                lastPaint = (typeof performance !== 'undefined' ? performance.now() : Date.now());
            }
        }
        // stats tail (skip first few warmup emissions)
        const tail = probs.slice(Math.min(5, probs.length));
        const mean = tail.length ? (tail.reduce((a, b) => a + b, 0) / tail.length) : 0;
        const maxp = tail.length ? Math.max(...tail) : 0;
        this.stats?.updateFileStats({ mean, max: maxp, n: probs.length });
        this.plot.render();
    }

    handleRunFileClick() {
        let fi = document.getElementById('file-audio');
        if (!fi) return;
        const parent = fi.parentNode;
        const clone = fi.cloneNode(true);
        parent.replaceChild(clone, fi);
        fi = clone;
        fi.addEventListener('change', async () => {
            const file = fi.files && fi.files[0];
            if (!file) return;
            await this.prepareForFile();
            try { await this.runFile(file); }
            finally { await this.finalizeAfterFile(); }
        }, { once: true });
        fi.click();
    }
}
