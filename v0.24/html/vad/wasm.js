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
    // FSMN-VAD is causal (rorder=0) and has no decoder stage, so the old
    // marblenet-era pulse delay / decoder pool accessors are now constants.
    getPulseDelay() { return 0; }
    getDecoderPoolLen() { return this.getPulseFrames(); }
    isPulsedReady() { try { return this.classifier.is_pulsed_ready(); } catch { return false; } }
    getPulseFrames() { try { return this.classifier.get_pulse_frames?.(); } catch { return 4; } }
    getFrameSize() { try { return this.classifier.get_frame_size?.(); } catch { return 160; } }
}
