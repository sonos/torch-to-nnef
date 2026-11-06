import { VADPlot } from './plot.js';
import { WasmVAD } from './wasm.js';
import { ModeController, Controls, StatsPanel } from './ui.js';
import { VadSession } from './session.js';
import { MicRunner } from './mic.js';

function setVisible(id, on) {
    const el = document.getElementById(id);
    if (el) el.style.display = on ? 'block' : 'none';
}

export async function initVAD() {
    // Grab required DOM nodes from existing HTML
    const container = document.getElementById('vad-plot');
    const stats = new StatsPanel(document);
    // Show loading until wasm is ready
    setVisible('page', false);
    setVisible('loading', true);

    // Optional: allow overriding pulse via global (e.g., window.VAD_PULSE)
    const pulseOverride = Number(window?.VAD_PULSE ?? 0) || undefined;
    const wasm = await WasmVAD.load({ pulseFrames: pulseOverride });
    const modes = new ModeController(document);
    const controls = new Controls(document);
    // Build base uPlot opts from globals if present or define minimal
    const baseOpts = window.opts || {
        width: 640, // will be recalculated on reset
        height: 150,
        pxAlign: false,
        scales: { x: { time: false }, y: { range: [0, 1.1] } },
        axes: [{ space: 50, values: (self, val) => val.map(v => `${v / 1000}s`) }],
        series: [
            {},
            { label: 'score (pulsed)', stroke: '#1976d2', width: 2, fill: '#1976d220', spanGaps: false },
            { label: 'score (batch)', stroke: '#43a047', width: 2, fill: '#43a04720', spanGaps: false },
            { label: 'detection', stroke: '#e53935', width: 2, fill: '#e5393520', spanGaps: false },
        ],
    };
    const plot = new VADPlot(container, baseOpts, 256, 0.8);

    // Wire controls
    // Make page visible before initializing plot (to get correct width)
    setVisible('page', true);
    setVisible('loading', false);
    plot.init(60, 500);
    const sess = new VadSession(wasm, plot, modes, stats);
    const mic = new MicRunner(wasm, plot, modes, controls);
    controls.onRunFile(async () => { await mic.stop(); sess.handleRunFileClick(); });
    controls.onMicToggle(() => mic.toggle());

    // Sync legend visibility with current mode and on changes
    // Store mode for later; plot not initialized until first action
    plot.setMode(modes.get());
    // Pull pulsed delay from WASM and shade 0..delay
    try {
        const delayFrames = wasm.getPulseDelay() || 0;
        const poolFrames = wasm.getDecoderPoolLen() || 0; // usually 10
        const pulseFrames = wasm.getPulseFrames() || 4;
        const frameSize = wasm.getFrameSize() || 160; // samples per frame
        // Convert frame size to milliseconds: 160 samples at 16kHz -> 10ms
        const frameMs = Math.round(frameSize / 16);
        // Need frames before first valid pulsed decode
        const needFrames = delayFrames + poolFrames;
        // Quantize to pulse multiple: first decode occurs at ceil(need/pulse)*pulse frames
        const warmFrames = Math.ceil(needFrames / pulseFrames) * pulseFrames;
        plot.setPulseDelayMs(warmFrames * frameMs);
    } catch { }
    modes.onChange((m) => plot.setMode(m));

    // Resize: update plot size on window changes
    window.addEventListener('resize', () => plot.resizeToContainer());

    // Resize notifications for embedding
    const sendHeight = () => {
        const height = document.documentElement.scrollHeight;
        try { window.parent.postMessage({ type: 'resize', height, ref: 'vad' }, '*'); } catch {}
    };
    window.addEventListener('load', sendHeight);
    try { new ResizeObserver(sendHeight).observe(document.body); } catch { /* older browsers */ }
    window.vadPlot = plot;
    window.vadSession = sess;
}

// Auto-init when the module is imported by demo_vad.html
document.addEventListener('DOMContentLoaded', () => {
    initVAD().catch(err => console.error('VAD init error', err));
});
