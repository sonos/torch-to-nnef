#[allow(clippy::missing_safety_doc)]

use nemo_asr::NemoAsrModel;

use anyhow::{Context, anyhow, ensure,Error, Result};
use libc::c_char;
use ffi_convert::{AsRust, RawBorrow, RawPointerConverter, CReprOf};
use log::LevelFilter;
use std::cell::RefCell;
use std::path::PathBuf;
use std::ffi::{CStr, CString};
use std::sync::Once;

static INIT_LOGGER: Once = Once::new();

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Debug, PartialEq, Eq)]
pub enum NEMO_ASR_FFI_RESULT {
    /// The function returned successfully
    NEMO_ASR_FFI_RESULT_OK = 0,
    /// The function returned an error
    NEMO_ASR_FFI_RESULT_KO = 1,
}

thread_local! {
    pub(crate) static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

pub fn wrap<F: FnOnce() -> Result<()>>(func: F) -> NEMO_ASR_FFI_RESULT {
    match func() {
        Ok(_) => NEMO_ASR_FFI_RESULT::NEMO_ASR_FFI_RESULT_OK,
        Err(e) => {
            let msg = format!("{:#?}", e);
            if std::env::var("NEMO_ASR_FFI_ERROR_STDERR").is_ok() {
                eprintln!("{}", msg);
            }
            LAST_ERROR.with(|p| *p.borrow_mut() = Some(msg));
            NEMO_ASR_FFI_RESULT::NEMO_ASR_FFI_RESULT_KO
        }
    }
}

/// Used to retrieve the last error that happened in this thread. A function encountered an error if
/// its return type is of type `NEMO_ASR_FFI_RESULT` and it returned `NEMO_ASR_FFI_RESULT_KO`.
///
/// # Arguments
///  - `error`: pointer to a string that will contain the error description, this should then be
///  destroyed properly using the `NEMO_ASR_destroy_string` function in this lib to prevent leaks
///
/// # Return type
/// Should return `NEMO_ASR_FFI_RESULT_OK`.
///
/// If `NEMO_ASR_FFI_RESULT_KO` is returned, then something very wrong happened in the lib.
///
/// # Safety
/// Make sure that the passed in pointer is safe to be dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_asr_get_last_error(error: *mut *mut c_char) -> NEMO_ASR_FFI_RESULT {
    wrap(move || {
        LAST_ERROR.with(|msg| {
            let string = msg
                .borrow_mut()
                .take()
                .unwrap_or_else(|| "No error message".to_string());
            let result: *const c_char = CString::c_repr_of(string)?.into_raw_pointer();
            unsafe { *error = result as _ };
            Ok(())
        })
    })
}

/// Used to destroy aa string created by the lib
///
/// # Arguments
///  - `ptr`: pointer to the string to destroy
///
/// # Return type
/// Returns `NEMO_ASR_FFI_RESULT_OK` if the string was destroyed properly.
///
/// If `NEMO_ASR_FFI_RESULT_KO` is returned, you can get more information on the error using the
/// `NEMO_ASR_get_last_error` function.
///
/// # Safety
/// Make sure that the passed in pointer is safe to be dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_asr_destroy_string(string: *mut c_char) -> NEMO_ASR_FFI_RESULT {
    wrap(|| unsafe { CString::drop_raw_pointer(string) }.map_err(Error::from))
}


/// Setup logger on Rust side
///
/// # Arguments
///  - `log_level`: string
///
/// # Return type
/// Returns `NEMO_ASR_FFI_RESULT_OK` if the logger has been setup properly.
///
/// If `NEMO_ASR_FFI_RESULT_KO` is returned, you can get more information on the error using the
/// `NEMO_ASR_get_last_error` function.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_asr_init_env_logger(verbosity: libc::size_t) -> NEMO_ASR_FFI_RESULT {
    wrap(|| {
        let mut log_result = None;
        INIT_LOGGER.call_once(|| log_result = Some(init_env_logger(verbosity)));
        log_result.transpose()?;
        Ok(())
    })
}

fn init_env_logger(verbosity: usize) -> Result<()> {
    let mut logger = env_logger::builder();
    // Setting up log level
    match verbosity {
        0 => {
            logger.filter_module("nemo_asr", LevelFilter::Info);
        }
        1 => {
            logger.filter_module("nemo_asr", LevelFilter::Debug);
            logger.filter_module("tract", LevelFilter::Info);
        }
        2 => {
            logger.filter_module("nemo_asr", LevelFilter::Trace);
            logger.filter_module("tract", LevelFilter::Debug);
        }
        _ => {
            logger.filter_module("nemo_asr", LevelFilter::Trace);
            logger.filter_module("tract", LevelFilter::Trace);
        }
    }

    logger
        .format_timestamp_millis()
        .try_init()
        .map_err(Error::from)?;
    Ok(())
}


#[derive(RawPointerConverter)]
pub struct FFINemoAsrModel(NemoAsrModel);

/// Load FFINemoAsrModel from a directory path. Destroy it using
/// `nemo_asr_model_destroy`.
///
/// # Arguments
///  - `model_dir`: path to model directory
///  - `asr`: Pointer to the NemoAsrModel to create
///
/// # Return type
/// Returns `ASR_FFI_RESULT_OK` if the function has run successfully.
///
/// If `SVC_LLM_FFI_RESULT_KO` is returned, you can get more information on the error using the
/// `svc_llm_get_last_error` function.
///
/// # Safety
/// Make sure that the passed in pointer is safe to be dereferenced.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_asr_from_dir(
    ptr: *mut *const FFINemoAsrModel,
    model_dir: *const c_char,
) -> NEMO_ASR_FFI_RESULT {
    crate::wrap(|| {
        let model_dir = unsafe { CStr::raw_borrow(model_dir) }?.as_rust()?;
        let asr = FFINemoAsrModel(NemoAsrModel::from_dir(model_dir.into())?);
        unsafe { *ptr = asr.into_raw_pointer_mut() };
        Ok(())
    })
}


/// Get answer from ASR batch of input wavs.
///
/// # Arguments
///  - `model_dir`: path to model directory
/// # Return type
/// Will return `NEMO_ASR_RESULT_OK` if the function was run sucessfully.
///
/// If something wrong happens, the function returns `NEMO_ASR_RESULT_KO` and additional information on
/// the error can be retrieved using the `sve_afe_get_last_error` function on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn infer_from_wav_paths(
    model: *const FFINemoAsrModel,
    wav_paths: *const *const c_char,
    wav_paths_len: libc::size_t,
    ptr: *mut *const c_char,
) -> NEMO_ASR_FFI_RESULT {
    crate::wrap(|| {
        let model = unsafe {&FFINemoAsrModel::raw_borrow(model)?.0 };
        let wav_pathbuf = (0..wav_paths_len)
            .map(|i| {
                let cstr = unsafe { CStr::raw_borrow(*wav_paths.add(i)) }?;
                let str_slice = cstr.as_rust()?;
                Ok(PathBuf::from(str_slice))
            })
            .collect::<Result<Vec<PathBuf>>>()?;

        let transcripts = model.infer_from_wav_paths(&wav_pathbuf)?;
        let transcripts_str = serde_json::to_string(&transcripts)?;
        let transcripts_cstring = CString::new(transcripts_str)?;
        unsafe { *ptr = transcripts_cstring.into_raw_pointer_mut() };
        Ok(())
    })
}


/// Destroys a FFINemoAsrModel.
///
/// # Arguments
///  - `ptr`: Pointer to the FFINemoAsrModel to destroy
///
/// # Return type
/// Will return `SVE_LLM_RESULT_OK` if the function was run sucessfully.
///
/// If something wrong happens, the function returns `NEMO_ASR_FFI_RESULT_KO` and additional information on
/// the error can be retrieved using the `nemo_asr_get_last_error` function on the same thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_asr_model_destroy(ptr: *mut FFINemoAsrModel) -> NEMO_ASR_FFI_RESULT {
    crate::wrap(move || Ok(unsafe { FFINemoAsrModel::drop_raw_pointer(ptr) }?))
}
