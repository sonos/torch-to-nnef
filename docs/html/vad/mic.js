import { AudioResampler, RingBuffer } from './audio.js';

const DESIRED_SR = 16000;
const CHUNK_SAMPLES = 4 * 160; // 640 @16kHz

export class MicRunner {
    constructor(wasm, plot, modes, controls) {
        this.wasm = wasm;
        this.plot = plot;
        this.modes = modes;
        this.controls = controls;

        this.ac = null;
        this.source = null;
        this.worklet = null;
        this.resampler = null;
        this.fifo = new RingBuffer();
        this.running = false;
    }

    async start() {
        if (this.running) return;
        const AC = window.AudioContext || window.webkitAudioContext;
        try {
            // Request 16k if supported (browsers may ignore)
            this.ac = new AC({ sampleRate: DESIRED_SR });
        } catch {
            this.ac = new AC();
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: DESIRED_SR,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                googEchoCancellation: false,
                googAutoGainControl: false,
                googNoiseSuppression: false,
            }
        });

        const source = this.ac.createMediaStreamSource(stream);
        this.source = source;
        // Ensure plotting state is fresh
        const fps = 60;
        this.plot.init(fps, 200);
        this.plot.setTitle(fps);

        // Resampler to 16k with LPF
        this.resampler = new AudioResampler(this.ac.sampleRate, DESIRED_SR, true);
        await this.ac.audioWorklet.addModule('./vad_worklet.js');
        const worklet = new AudioWorkletNode(this.ac, 'vad-processor');
        this.worklet = worklet;
        source.connect(worklet);
        // pull the graph silently
        const silent = this.ac.createGain();
        silent.gain.value = 0;
        worklet.connect(silent).connect(this.ac.destination);

        let audioMs = 0;
        let nextPlotMs = 0;
        let lastScoreP = NaN, lastScoreB = NaN;
        let rafPending = false;
        this.modes?.disable(true);
        this.controls?.setMicRunning(true);
        this.running = true;

        worklet.port.onmessage = (ev) => {
            if (!this.running) return;
            const msg = ev.data || {};
            if (msg.type !== 'audio' || !msg.buffer) return;
            const merged = new Float32Array(msg.buffer);
            const resampled = (this.ac.sampleRate === DESIRED_SR) ? merged : this.resampler.process(merged);
            if (!resampled || !resampled.length) return;

            // Feed 16k FIFO and run fixed-size blocks
            this.fifo.push(resampled);
            const mode = this.modes?.get?.() || 'pulsed';
            while (this.fifo.length() >= CHUNK_SAMPLES) {
                const block = this.fifo.shift(CHUNK_SAMPLES);
                if (mode === 'both') {
                    lastScoreP = this.wasm.predictPulsed(block);
                    lastScoreB = this.wasm.predictBatch(block);
                } else if (mode === 'batch') {
                    lastScoreB = this.wasm.predictBatch(block);
                    lastScoreP = NaN;
                } else {
                    lastScoreP = this.wasm.predictPulsed(block);
                    lastScoreB = NaN;
                }
                audioMs += (CHUNK_SAMPLES / DESIRED_SR) * 1000;
                while (audioMs >= nextPlotMs) {
                    this.plot.push(nextPlotMs, lastScoreP, lastScoreB, mode);
                    nextPlotMs += (1000 / this.plot.hz);
                }
            }

            if (!rafPending) {
                rafPending = true;
                requestAnimationFrame(() => {
                    this.plot.render();
                    rafPending = false;
                });
            }
        };
    }

    async stop() {
        if (!this.running) return;
        try { this.worklet && this.worklet.port && (this.worklet.port.onmessage = null); } catch { }
        try { this.worklet && this.worklet.disconnect(); } catch { }
        try { this.source && this.source.disconnect(); } catch { }
        try { this.ac && this.ac.close && (await this.ac.close()); } catch { }
        this.ac = null; this.source = null; this.worklet = null;
        this.resampler = null; this.fifo = new RingBuffer();
        this.running = false;
        this.modes?.disable(false);
        this.controls?.setMicRunning(false);
    }

    async toggle() {
        if (this.running) return this.stop();
        try { await this.start(); }
        catch (err) {
            console.error('Mic start error', err);
            await this.stop();
            const ec = document.getElementById('error-container');
            const em = document.getElementById('error');
            if (ec) ec.style.display = 'block';
            if (em) em.innerHTML = `${err} If you are on mobile ensure that your browser app have access to microphone: <a target='_blank' rel='noopener noreferrer' href='https://support.proof.com/hc/en-us/articles/14237306788503-Enable-camera-and-microphone-for-mobile'>documentation</a>`;
        }
    }
}

