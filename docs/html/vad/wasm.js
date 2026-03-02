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
}
