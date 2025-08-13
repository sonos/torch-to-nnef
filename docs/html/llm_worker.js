import init, { LLM, LLMState } from '/html/llm_wasm.js';
var llm = null;
var llmState = null;

const processInitialMessage = (prompt) => {
    try {
        llmState = llm.new_state();
        console.log("start prompt processing: ", prompt)
        let nextToken = llmState.process_prompt(prompt);
        console.log("finished prompt processing")
        postMessage({
            kind: "poemGen",
            value: nextToken
        });
        console.log("prompt processing sent")
        return true;
    } catch (error) {
        console.log("newToken generation worker error", error);
        postMessage({
            kind: "poemFinished",
            reason: "error",
            message: error
        });
        llmState = null;
    }
    return false;

};

const processNextToken = () => {
    try {
        console.log("start next token processing");
        let newToken = llmState.process_next_token();
        console.log("finished next token processing");
        postMessage({
            kind: "poemGen",
            value: newToken
        });
        console.log("next token sent");
        return true;
    } catch (error) {
        console.log("newToken generation worker error", error);
        postMessage({
            kind: "poemFinished",
            reason: "error",
            message: error
        });
        llmState = null;
    }
    return false;
};

const generatePoemTask = (prompt) => {
    let idx = 0;
    if (!processInitialMessage(prompt)) return;
    while (idx < 100) {
        if (!processNextToken()) return;
        idx++;
    }
    postMessage({
        kind: "poemFinished",
        reason: "maxSize"
    });
    llmState = null;
};


onmessage = (e) => {
    let msg = e.data;
    console.log("LLMWorker: Message received from main script", msg);
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
        generatePoemTask(msg.value);
    } else {
        console.log("Un-expected request in worker:", e);
    }
};
console.log("loaded from worker");
