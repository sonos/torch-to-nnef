from datasets import Audio, load_dataset

from . import BasicMultilingualTextNormalizer, EnglishTextNormalizer


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            "Expected transcript column of either 'text', 'sentence', "
            "'normalized_text' or 'transcript'. Got sample of "
            f".join{sample.keys()}. Ensure a text column name "
            "is present in the dataset."
        )


normalizer = EnglishTextNormalizer()

ml_normalizer = BasicMultilingualTextNormalizer()


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


def load_data(args):
    dataset = load_dataset(
        args.hg_path,
        args.name,
        split=args.split,
        streaming=args.streaming,
        token=True,
    )

    return dataset


def prepare_data(dataset):
    # Re-sample to 16kHz and normalise transcriptions
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.map(normalize)
    dataset = dataset.filter(
        is_target_text_in_range, input_columns=["norm_text"]
    )

    return dataset
