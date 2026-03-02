export class VADPlot {
    constructor(container, opts, seriesLen = 256, threshold = 0.95) {
        this.container = container;
        this.opts = opts;
        this.seriesLen = seriesLen;
        this.threshold = threshold;
        this.u = null;
        this.plotMsStep = 1000 / 60;
        this.hz = 60;
        this.postProcWindowSize = Math.max(1, Math.round(200 / this.plotMsStep));
        this.times = null;
        this.scoresP = null;
        this.scoresB = null;
        this.det = null;
        this.data = null;
        this.currentMode = 'pulsed';
        this.pulseDelayMs = 0;
    }
    _ensurePreZeroShadePlugin() {
        const that = this;
        const preZeroShade = {
            hooks: {
                // Draw over series to mask any area-fill artifacts before 0s
                draw: (u) => {
                    const { ctx, bbox } = u;
                    const x0 = u.valToPos(0, 'x', true);
                    const left = bbox.left;
                    const right = bbox.left + bbox.width;
                    // Clamp shade region to chart bounds: shade [left, min(x0, right)]
                    let r = Math.min(Math.max(x0, left), right);
                    const w = r - left;
                    if (w <= 0) return; // zero or negative width → nothing to shade
                    ctx.save();
                    // Solid light gray (no opacity), to clearly mark < 0s
                    ctx.fillStyle = '#e0e0e0';
                    ctx.fillRect(left, bbox.top, w, bbox.height);
                    // Shade 0..pulseDelay if provided
                    if (that.pulseDelayMs && that.pulseDelayMs > 0) {
                        const xDelay = u.valToPos(that.pulseDelayMs, 'x', true);
                        const dl = Math.max(x0, left);
                        const dr = Math.min(xDelay, right);
                        const dw = dr - dl;
                        if (dw > 0) {
                            // Transparent light gray so series remain visible beneath
                            ctx.fillStyle = 'rgba(224,224,224,0.25)';
                            ctx.fillRect(dl, bbox.top, dw, bbox.height);
                        }
                    }
                    ctx.restore();
                },
            },
        };
        const hasPlugins = Array.isArray(this.opts.plugins);
        const already = hasPlugins && this.opts.plugins.some(p => p && p.hooks && (p.hooks.drawClear));
        if (!already) {
            this.opts = { ...this.opts, plugins: [...(this.opts.plugins || []), preZeroShade] };
        }
    }
    init(plotFps = 60, smoothMs = 200) {
        this.plotMsStep = 1000 / plotFps;
        this.hz = plotFps;
        this.postProcWindowSize = Math.max(1, Math.round(smoothMs / this.plotMsStep));
        this.reset();
    }
    _containerSize() {
        // Compute width from container, with sane floor to avoid 0 when hidden
        const w = Math.max(320, (this.container?.clientWidth || this.container?.getBoundingClientRect?.().width || 0) - 50);
        const h = this.opts?.height || 150;
        return { width: w, height: h };
    }
    reset() {
        try { if (this.u && this.u.destroy) this.u.destroy(); } catch { }
        this.u = null;
        // Recompute width from container each reset
        const sz = this._containerSize();
        this.opts = { ...this.opts, width: sz.width, height: sz.height };
        this._ensurePreZeroShadePlugin();
        const L = this.seriesLen;
        this.times = Array.from({ length: L }, (v, i) => i * this.plotMsStep - (L * this.plotMsStep));
        this.scoresP = Array.from({ length: L }, () => null);
        this.scoresB = Array.from({ length: L }, () => null);
        this.det = Array.from({ length: L }, () => null);
        this.data = [this.times.slice(), this.scoresP.slice(), this.scoresB.slice(), this.det.slice()];
        this.u = new uPlot(this.opts, this.data, this.container);
        // Re-apply legend visibility after rebuilding chart
        this.applyModeLegend(this.currentMode);
    }
    zero() {
        if (!this.times) this.reset();
        this.scoresP = Array.from({ length: this.seriesLen }, () => null);
        this.scoresB = Array.from({ length: this.seriesLen }, () => null);
        this.det = Array.from({ length: this.seriesLen }, () => null);
        this.data = [this.times.slice(), this.scoresP.slice(), this.scoresB.slice(), this.det.slice()];
        this.u ? this.u.setData(this.data) : (this.u = new uPlot(this.opts, this.data, this.container));
    }
    push(timeMs, pPulsed, pBatch, mode = 'pulsed') {
        if (!this.times) this.reset();
        const L = this.seriesLen;
        const cloneShift = (arr, v) => arr.slice(1).concat(v);
        // Gate pulsed values prior to warmup (pulseDelayMs): hide fill/points until ready
        if (mode !== 'batch' && this.pulseDelayMs && timeMs < this.pulseDelayMs) {
            pPulsed = null;
        }
        const scoresSel = mode === 'batch' ? this.scoresB : this.scoresP;
        const lastScoreSel = mode === 'batch' ? pBatch : pPulsed;
        const start = Math.max(0, scoresSel.length - this.postProcWindowSize);
        const tail = scoresSel.slice(start).filter((x) => x != null && Number.isFinite(x));
        const maxDet = tail.length ? (Math.max(...tail) > this.threshold ? 1.0 : 0.0) : null;
        const prevDet = this.det.length ? this.det[this.det.length - 1] : null;
        const isDet = (lastScoreSel != null && Number.isFinite(lastScoreSel))
            ? ((lastScoreSel > this.threshold) ? 1.0 : 0.0)
            : (prevDet != null ? prevDet : maxDet);

        this.times = cloneShift(this.times, timeMs);
        this.scoresP = cloneShift(this.scoresP, (pPulsed != null && Number.isFinite(pPulsed)) ? pPulsed : null);
        this.scoresB = cloneShift(this.scoresB, (pBatch != null && Number.isFinite(pBatch)) ? pBatch : null);
        this.det = cloneShift(this.det, (isDet != null && Number.isFinite(isDet)) ? isDet : null);
        this.data = [this.times, this.scoresP, this.scoresB, this.det];
        return this.data;
    }
    render() {
        if (!this.u) this.u = new uPlot(this.opts, this.data, this.container);
        else this.u.setData(this.data);
    }
    resizeToContainer() {
        if (!this.u) return;
        const sz = this._containerSize();
        this.u.setSize(sz);
    }
    applyModeLegend(mode = 'pulsed') {
        if (!this.u) return;
        this.currentMode = mode;
        const showP = mode !== 'batch';
        const showB = mode !== 'pulsed';
        // series indices: 0=time, 1=pulsed, 2=batch, 3=detection
        try { this.u.setSeries(1, { show: showP }); } catch { }
        try { this.u.setSeries(2, { show: showB }); } catch { }
    }
    setMode(mode) {
        this.applyModeLegend(mode);
    }
    show() {
        try { this.container.classList.add('show-plot'); } catch { }
    }
    setPulseDelayMs(ms) {
        this.pulseDelayMs = ms || 0;
        if (this.u) this.u.setData(this.data);
    }
    setTitle(hz) {
        this.opts = { ...this.opts, title: `VAD detection with Nvidia MarbleNet @ ${hz}hz:` };
        // Reapply title on next reset; uPlot doesn’t support live title updates directly without full opts rebuild.
    }
}
