import init, { LLM, LLMState } from '/html/llm_wasm.js';
let llm = null;
let llmState = null;



onmessage = (e) => {
    console.log("LLMWorker: Message received from main script");
    let msg = e.data;
    if (msg.kind == "initLLM") {
        init().then(() => {
            console.log("start wasm");
            try {
                postMessage({
                    kind: "llmStatus",
                    value: "loading",
                })
                llm = LLM.load();
                postMessage({
                    kind: "llmStatus",
                    value: "ready",
                })
            } catch (error) {
                postMessage({
                    kind: "llmStatus",
                    value: "fail",
                    message: error
                })
            }
            console.log("inited LLM");
        })
    } else if (msg.kind == "generatePoem" && llm !== null) {
        llmState = llm.new_state();
        postMessage({
            kind: "poemGen",
            value: llmState.process_prompt(msg.value)
        });
        let idx = 0;
        while (idx < 100) {
            postMessage({
                kind: "poemGen",
                value: llmState.process_next_token()
            });
            idx++;
        }
        postMessage({
            kind: "poemFinished",
            reason: "maxSize"
        });
    } else {
        console.log("Un-expected request in worker:", e);
    }
};
console.log("loaded from worker");
