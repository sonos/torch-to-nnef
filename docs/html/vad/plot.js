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
        const L = this.seriesLen;
        this.times = Array.from({ length: L }, (v, i) => i * this.plotMsStep - (L * this.plotMsStep));
        this.scoresP = Array.from({ length: L }, () => NaN);
        this.scoresB = Array.from({ length: L }, () => NaN);
        this.det = Array.from({ length: L }, () => NaN);
        this.data = [this.times.slice(), this.scoresP.slice(), this.scoresB.slice(), this.det.slice()];
        this.u = new uPlot(this.opts, this.data, this.container);
    }
    zero() {
        if (!this.times) this.reset();
        this.scoresP = Array.from({ length: this.seriesLen }, () => NaN);
        this.scoresB = Array.from({ length: this.seriesLen }, () => NaN);
        this.det = Array.from({ length: this.seriesLen }, () => NaN);
        this.data = [this.times.slice(), this.scoresP.slice(), this.scoresB.slice(), this.det.slice()];
        this.u ? this.u.setData(this.data) : (this.u = new uPlot(this.opts, this.data, this.container));
    }
    push(timeMs, pPulsed, pBatch, mode = 'pulsed') {
        if (!this.times) this.reset();
        const L = this.seriesLen;
        const cloneShift = (arr, v) => arr.slice(1).concat(v);
        const scoresSel = mode === 'batch' ? this.scoresB : this.scoresP;
        const lastScoreSel = mode === 'batch' ? pBatch : pPulsed;
        const start = Math.max(0, scoresSel.length - this.postProcWindowSize);
        const tail = scoresSel.slice(start).filter((x) => Number.isFinite(x));
        const maxDet = tail.length ? (Math.max(...tail) > this.threshold ? 1.0 : 0.0) : NaN;
        const prevDet = this.det.length ? this.det[this.det.length - 1] : NaN;
        const isDet = Number.isFinite(lastScoreSel)
            ? ((lastScoreSel > this.threshold) ? 1.0 : 0.0)
            : (Number.isFinite(prevDet) ? prevDet : maxDet);

        this.times = cloneShift(this.times, timeMs);
        this.scoresP = cloneShift(this.scoresP, Number.isFinite(pPulsed) ? pPulsed : NaN);
        this.scoresB = cloneShift(this.scoresB, Number.isFinite(pBatch) ? pBatch : NaN);
        this.det = cloneShift(this.det, isDet);
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
    setTitle(hz) {
        this.opts = { ...this.opts, title: `VAD detection with Nvidia MarbleNet @ ${hz}hz:` };
        // Reapply title on next reset; uPlot doesn’t support live title updates directly without full opts rebuild.
    }
}
