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
    const container = document.getElementById('vad-preview');
    const stats = new StatsPanel(document);
    // Show loading until wasm is ready
    setVisible('page', false);
    setVisible('loading', true);

    const wasm = await WasmVAD.load();
    const modes = new ModeController(document);
    const controls = new Controls(document);
    // Build base uPlot opts from globals if present or define minimal
    const baseOpts = window.opts || {
        title: 'VAD detection with Nvidia MarbleNet',
        width: 640, // will be recalculated on reset
        height: 150,
        pxAlign: false,
        scales: { x: { time: false }, y: { range: [0, 1.1] } },
        axes: [{ space: 50, values: (self, val) => val.map(v => `${v / 1000}s`) }],
        series: [{}, { label: 'score (pulsed)', stroke: '#1976d2', width: 2, fill: '#1976d220' }, { label: 'score (batch)', stroke: '#43a047', width: 2, fill: '#43a04720' }, { label: 'detection', stroke: '#e53935', width: 2, fill: '#e5393520' }],
    };
    const plot = new VADPlot(container, baseOpts, 256, 0.95);

    // Wire controls
    // Make page visible before initializing plot (to get correct width)
    setVisible('page', true);
    setVisible('loading', false);
    plot.init(60, 200);
    const sess = new VadSession(wasm, plot, modes, stats);
    const mic = new MicRunner(wasm, plot, modes, controls);
    controls.onRunFile(async () => { await mic.stop(); sess.handleRunFileClick(); });
    controls.onMicToggle(() => mic.toggle());

    // Resize: update plot size on window changes
    window.addEventListener('resize', () => plot.resizeToContainer());

    // Resize notifications for embedding
    const sendHeight = () => {
        const height = document.documentElement.scrollHeight;
        window.parent.postMessage({ type: 'resize', height, ref: 'vad' }, '*');
    };
    window.addEventListener('load', sendHeight);
    new ResizeObserver(sendHeight).observe(document.body);
    window.vadPlot = plot;
    window.vadSession = sess;
}

// Auto-init when the module is imported by demo_vad.html
document.addEventListener('DOMContentLoaded', () => {
    initVAD().catch(err => console.error('VAD init error', err));
});
