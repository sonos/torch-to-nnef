import initWasm, { VadClassifier } from '../vad_wasm.js';

export class WasmVAD {
    static async load(opts = {}) {
        await initWasm();
        const inst = new WasmVAD();
        try {
            if (opts.pulseFrames && typeof opts.pulseFrames === 'number' && opts.pulseFrames > 0) {
                inst.classifier = VadClassifier.load_with_pulse(opts.pulseFrames);
            } else {
                inst.classifier = VadClassifier.load();
            }
        } catch {
            inst.classifier = VadClassifier.load();
        }
        return inst;
    }
    reset() {
        try { this.classifier.reset_sessions?.(); } catch { }
    }
    predictPulsed(block) { return this.classifier.predict_speech_presence(block); }
    predictBatch(block) { return this.classifier.predict_speech_presence_batch(block); }
    getPulseDelay() { try { return this.classifier.get_pulse_delay(); } catch { return 0; } }
    getDecoderPoolLen() { try { return this.classifier.get_decoder_pool_len(); } catch { return 10; } }
    isPulsedReady() { try { return this.classifier.is_pulsed_ready(); } catch { return false; } }
    getPulseFrames() { try { return this.classifier.get_pulse_frames?.() ?? 4; } catch { return 4; } }
    getFrameSize() { try { return this.classifier.get_frame_size?.() ?? 160; } catch { return 160; } }
}
