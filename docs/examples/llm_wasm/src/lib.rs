use anyhow::anyhow;
use causal_llm::{CausalLlmModel, CausalLlmState};
use tract_nnef::internal::TractErrorContext;
use tract_nnef::prelude::*;
use tract_transformers::WithTractTransformers;
use wasm_bindgen::prelude::*;

extern crate wee_alloc;

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

extern crate web_sys;

#[wasm_bindgen]
struct LLM {
    llm_model: Arc<CausalLlmModel>,
}

#[wasm_bindgen]
struct LLMState {
    state: CausalLlmState,
    prompt_processed: bool,
    last_token: u32,
}

#[wasm_bindgen]
impl LLM {
    fn load_internal() -> TractResult<LLM> {
        web_sys::console::log_1(&"> bytes get".into());
        let tokenizer_bytes = include_bytes!("../dump_model/tokenizer/tokenizer.json");
        let llm_model_bytes = include_bytes!("../dump_model/model/model.nnef.tgz");
        web_sys::console::log_1(&"> bytes ready".into());
        let llm_model = CausalLlmModel::from_bytes(tokenizer_bytes, llm_model_bytes)?;
        web_sys::console::log_1(&"> model loaded/optimized".into());
        Ok(LLM { llm_model })
    }

    pub fn new_state(&mut self) -> Result<LLMState, JsError> {
        let state = self
            .llm_model
            .spawn()
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))?;

        Ok(LLMState {
            state,
            last_token: 0,
            prompt_processed: false,
        })
    }

    pub fn load() -> Result<LLM, JsError> {
        web_sys::console::log_1(&"> Try loading".into());
        LLM::load_internal().map_err(|err| JsError::new(format!("{:?}", err).as_str()))
    }
}

#[wasm_bindgen]
impl LLMState {
    fn get_last_tok(&self) -> JsValue {
        self.state.decode(&[self.last_token], false).unwrap().into()
    }

    pub fn process_prompt(&mut self, prompt: String) -> Result<JsValue, JsError> {
        web_sys::console::log_1(&"> start process text".into());
        self.last_token = self.state.process_text(&prompt).unwrap();
        web_sys::console::log_1(&"> finished process text".into());
        self.prompt_processed = true;
        Ok(self.get_last_tok())
    }

    pub fn process_next_token(&mut self) -> Result<JsValue, JsError> {
        assert!(self.prompt_processed);
        self.last_token = self.state.process_tokens(&[self.last_token]).unwrap();
        Ok(self.get_last_tok())
    }
}
