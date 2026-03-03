import initWasm, { VadClassifier } from '../vad_wasm.js';

export class WasmVAD {
    static async load(opts = {}) {
        await initWasm();
        const inst = new WasmVAD();
        if (opts.pulseFrames && typeof opts.pulseFrames === 'number' && opts.pulseFrames > 0) {
            inst.classifier = VadClassifier.load_with_pulse(opts.pulseFrames);
        } else {
            inst.classifier = VadClassifier.load();
        }
        return inst;
    }
    reset() {
        try { this.classifier.reset_sessions?.(); } catch { }
    }
    predictPulsed(block) { return this.classifier.predict_speech_presence(block); }
    predictBatch(block) { return this.classifier.predict_speech_presence_batch(block); }
    getPulseDelay() { return this.classifier.get_pulse_delay(); }
    getDecoderPoolLen() { return this.classifier.get_decoder_pool_len(); }
    isPulsedReady() { return this.classifier.is_pulsed_ready(); }
    getPulseFrames() { return this.classifier.get_pulse_frames?.(); }
    getFrameSize() { return this.classifier.get_frame_size?.() }
}
