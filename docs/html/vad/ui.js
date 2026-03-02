export class ModeController {
    constructor(root = document) {
        this.root = root;
        this.mode = this._currentFromUI();
        this.listeners = [];
        const radios = this.root.getElementsByName('vad-mode');
        for (let i = 0; i < radios.length; i++) {
            radios[i].addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.mode = e.target.value;
                    this.listeners.forEach(cb => cb(this.mode));
                }
            });
        }
    }
    _currentFromUI() {
        const sel = this.root.querySelector('input[name="vad-mode"]:checked');
        return sel && sel.value ? sel.value : 'pulsed';
    }
    get() { return this.mode; }
    set(m) {
        this.mode = m;
        const el = this.root.querySelector(`input[name="vad-mode"][value="${m}"]`);
        if (el) el.checked = true;
        this.listeners.forEach(cb => cb(this.mode));
    }
    disable(b) {
        const radios = this.root.getElementsByName('vad-mode');
        radios.forEach?.(r => r.disabled = !!b);
    }
    onChange(cb) { this.listeners.push(cb); }
}

export class Controls {
    constructor(root = document) {
        this.root = root;
        this.micBtn = this.root.getElementById('vad-click');
        this.runBtn = this.root.getElementById('run-file');
    }
    onMicToggle(cb) {
        if (!this.micBtn) return;
        this.micBtn.addEventListener('click', async () => {
            await cb();
        });
    }
    onRunFile(cb) {
        if (!this.runBtn) return;
        this.runBtn.addEventListener('click', cb);
    }
    setMicRunning(running) {
        if (!this.micBtn) return;
        if (running) {
            this.micBtn.innerHTML = '<i class="material-icons left">stop</i>Stop Microphone';
            this.micBtn.classList.remove('blue', 'darken-3');
            this.micBtn.classList.add('red', 'darken-2');
        } else {
            this.micBtn.innerHTML = '<i class="material-icons left">mic</i>From Microphone';
            this.micBtn.classList.remove('red', 'darken-2');
            this.micBtn.classList.add('blue', 'darken-3');
        }
    }
}

export class StatsPanel {
    constructor(root = document) {
        this.fileStats = root.getElementById('file-stats');
        this.lastStats = null;
    }
    setFileText(text) {
        if (!this.fileStats) return;
        this.fileStats.textContent = text;
    }
    updateFileStats({ mean, max, n }) {
        if (!this.fileStats) return;
        this.fileStats.textContent = `mean: ${mean.toFixed(3)}  max: ${max.toFixed(3)}  n: ${n}`;
    }
}
