"""
Fine-tuning starter.
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import random
from pathlib import Path

from transformers import AutoTokenizer, set_seed

from config.lab_settings import LabSettings, SFTParams
from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import (
    LLMPipeline,
    RawDataImporter,
    RawDataPreprocessor,
    SFTPipeline,
    TaskDataset,
    TaskEvaluator,
    TokenizedTaskDataset,
)


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """

    ## Inference constants
    max_length = 120
    batch_size = 64
    device = 'cpu'
    settings_path = Path(__file__).parent / 'settings.json'
    predictions_path = Path(__file__).parent / 'dist' / 'predictions.csv'

    settings = LabSettings(settings_path)

    # 1. Data preprocessing
    data_importer = RawDataImporter(settings.parameters.dataset)
    data_importer.obtain()

    if data_importer.raw_data is None:
        raise ValueError('No dataset created by obtain() method')

    data_preprocessor = RawDataPreprocessor(data_importer.raw_data)
    _dataset_properties = data_preprocessor.analyze()
    data_preprocessor.transform()

    # # 2. Pre-trained model inference
    # preprocessed_dataset = TaskDataset(data_preprocessor.data.head(100))
    # pipeline = LLMPipeline(settings.parameters.model,
    #                        preprocessed_dataset,
    #                        max_length, batch_size, device)
    # _model_params = pipeline.analyze_model()
    #
    # sample = preprocessed_dataset[random.randint(0, len(preprocessed_dataset) - 1)]
    # _sample_prediction = pipeline.infer_sample(sample)
    #
    # dataset_predictions = pipeline.infer_dataset()
    #
    # predictions_path.parent.mkdir(parents=True, exist_ok=True)
    # dataset_predictions.to_csv(predictions_path)
    #
    # evaluator = TaskEvaluator(predictions_path,
    #                           settings.parameters.metrics)
    # evaluation_result = evaluator.run()
    # print(f'Inference result: {evaluation_result}')

    # 3. Fine-tuning
    ## Fine-tuning constants
    max_length = 120
    batch_size = 64
    sft_batch_size = 3
    max_fine_tuning_steps = 300
    device = 'cpu'
    finetuned_model_path = Path(__file__).parent / 'dist' / f'{settings.parameters.model}-finetuned'
    learning_rate = 4e-3
    target_modules = [
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj"
    ]
    num_samples = 10

    set_seed(42)

    sft_parameters = SFTParams(
        max_length=max_length,
        batch_size=sft_batch_size,
        max_fine_tuning_steps=max_fine_tuning_steps,
        device=device,
        finetuned_model_path=str(finetuned_model_path),
        learning_rate=learning_rate,
        target_modules=target_modules
    )
    tokenized_dataset = TokenizedTaskDataset(
        data_preprocessor.data.loc[
            num_samples : num_samples + sft_batch_size * max_fine_tuning_steps
        ],
        AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length
    )

    sft_pipeline = SFTPipeline(settings.parameters.model, tokenized_dataset, sft_parameters)
    sft_pipeline.run()

    # 4. Fine-tuned model inference
    test_dataset = TaskDataset(data_preprocessor.data.head(num_samples))
    llm_pipeline_finetuned = LLMPipeline(
        str(finetuned_model_path),
        test_dataset,
        max_length, batch_size, device
    )

    _model_params_finetuned = llm_pipeline_finetuned.analyze_model()

    sample = test_dataset[random.randint(0, len(test_dataset) - 1)]
    _sample_prediction_finetuned = llm_pipeline_finetuned.infer_sample(sample)
    print(sample)
    print(_sample_prediction_finetuned)

    dataset_predictions_finetuned = llm_pipeline_finetuned.infer_dataset()

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_predictions_finetuned.to_csv(predictions_path)

    evaluator = TaskEvaluator(predictions_path,
                              settings.parameters.metrics)
    evaluation_result_finetuned = evaluator.run()
    print(f'Fine-tuning result: {evaluation_result_finetuned}')

    # "Helsinki-NLP/opus-mt-en-fr": {
    #     "enimai/MuST-C-fr": {
    #         "bleu": 0.45020
    #     }

    result = evaluation_result_finetuned
    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
