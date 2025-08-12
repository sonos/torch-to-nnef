use causal_llm::{CausalLlmModel, CausalLlmState, CausalLlmStateConfig};
use tokenizers::Tokenizer;
use tract_nnef::internal::TractErrorContext;
use tract_nnef::internal::anyhow;
use tract_nnef::prelude::Framework;
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
        let llm_model_bytes = include_bytes!("../dump_model/model/model.nnef.tgz");
        let mut llm_read = std::io::Cursor::new(llm_model_bytes);

        let tokenizer_bytes = include_bytes!("../dump_model/tokenizer/tokenizer.json");

        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(|e| anyhow!(e))?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut nn = nnef.model_for_read(&mut llm_read)?.into_decluttered()?;

        let transform = nnef
            .get_transform("transformers-detect-all")?
            .context("transformers-detect-all not found")?;
        nn.transform(&*transform)?;
        nn.optimize()?; // no memory arena for wasm so simple optimize
        let nn = nn.into_runnable()?;
        let llm_model = Arc::new(CausalLlmModel {
            tokenizer,
            nn: Arc::new(nn),
        });
        web_sys::console::log_1(&"model loaded/optimized".into());
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

    pub fn load() -> LLM {
        web_sys::console::log_1(&"try loading".into());
        let result = LLM::load_internal()
            .map_err(|err| JsError::new(format!("{:?}", err).as_str()))
            .expect("unable to load");
        result
    }
}

#[wasm_bindgen]
impl LLMState {
    fn get_last_tok(&self) -> JsValue {
        self.state.decode(&[self.last_token], false).unwrap().into()
    }

    pub fn process_prompt(&mut self, prompt: &str) -> Result<JsValue, JsError> {
        self.last_token = self.state.process_text(prompt).unwrap();
        self.prompt_processed = true;
        Ok(self.get_last_tok())
    }

    pub fn process_next_token(&mut self) -> Result<JsValue, JsError> {
        assert!(self.prompt_processed);
        self.last_token = self.state.process_tokens(&[self.last_token]).unwrap();
        Ok(self.get_last_tok())
    }
}
