import initWasm, { VadClassifier } from '../vad_wasm.js';

export class WasmVAD {
    static async load() {
        await initWasm();
        const inst = new WasmVAD();
        inst.classifier = VadClassifier.load();
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
}
