# """
# Fine-tuning starter.
# """
# # pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
# from pathlib import Path
#
# from transformers import AutoTokenizer
#
# from config.constants import PROJECT_ROOT
# from config.lab_settings import LabSettings, SFTParams
# from core_utils.llm.time_decorator import report_time
# from lab_8_sft.main import (
#     LLMPipeline,
#     RawDataImporter,
#     RawDataPreprocessor,
#     SFTPipeline,
#     TaskDataset,
#     TaskEvaluator,
#     TokenizedTaskDataset,
# )
#
#
# @report_time
# def main() -> None:
#     """
#     Run the translation pipeline.
#     """
#     settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
#
#     importer = RawDataImporter(settings.parameters.dataset)
#     importer.obtain()
#     if importer.raw_data is None:
#         return
#
#     preprocessor = RawDataPreprocessor(importer.raw_data)
#     _analysis = preprocessor.analyze()
#     preprocessor.transform()
#
#     num_samples = 10
#     sft_params = SFTParams(
#         batch_size=3,
#         max_length=120,
#         max_fine_tuning_steps=100,
#         learning_rate=1e-3,
#         finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model,
#         device="cpu",
#         # target_modules=["q", "v"]
#     )
#
#     fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps
#
#     sft_dataset = TokenizedTaskDataset(
#         preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
#         tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
#         max_length=sft_params.max_length
#     )
#
#     sft_pipeline = SFTPipeline(model_name=settings.parameters.model,
#                                dataset=sft_dataset,
#                                sft_params=sft_params)
#
#     sft_pipeline.run()
#
#     del sft_pipeline
#
#     dataset = TaskDataset(preprocessor.data.head(10))
#     pipeline = LLMPipeline(model_name=str(PROJECT_ROOT /
#     "lab_8_sft" / "dist" / settings.parameters.model),
#                            dataset=dataset,
#                            max_length=120,
#                            batch_size=64,
#                            device='cpu')
#
#     pipeline.analyze_model()
#
#     _sample_infer = pipeline.infer_sample(dataset[0])
#
#     predictions_path = Path(PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv')
#     if not predictions_path.parent.exists():
#         predictions_path.parent.mkdir()
#
#     pipeline.infer_dataset().to_csv(predictions_path, index=False)
#
#     evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
#     result = evaluator.run()
#     print(result)
#
#     assert result is not None, "Finetuning does not work correctly"
#
#
# if __name__ == "__main__":
#     main()
#
"""
Fine-tuning starter
"""
# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
from pathlib import Path

from transformers import AutoTokenizer

from config.constants import PROJECT_ROOT
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
    settings = LabSettings(PROJECT_ROOT / "lab_8_sft" / "settings.json")
    importer = RawDataImporter(settings.parameters.dataset)
    importer.obtain()
    if importer.raw_data is None:
        return
    preprocessor = RawDataPreprocessor(importer.raw_data)
    _analysis = preprocessor.analyze()
    preprocessor.transform()

    num_samples = 10
    sft_params = SFTParams(
        batch_size=3,
        max_length=120,
        max_fine_tuning_steps=5,
        learning_rate=1e-3,
        finetuned_model_path=PROJECT_ROOT / "lab_8_sft" / "dist" / settings.parameters.model,
        device="cpu"
    )

    fine_tune_samples = sft_params.batch_size * sft_params.max_fine_tuning_steps

    sft_dataset = TokenizedTaskDataset(
        preprocessor.data.loc[num_samples: num_samples + fine_tune_samples],
        tokenizer=AutoTokenizer.from_pretrained(settings.parameters.model),
        max_length=sft_params.max_length
    )

    sft_pipeline = SFTPipeline(model_name=settings.parameters.model,
                               dataset=sft_dataset,
                               sft_params=sft_params)

    sft_pipeline.run()

    # finetuned_tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT /
    #                                                         "lab_8_sft" /
    #                                                         "dist" / settings.parameters.model))

    dataset = TaskDataset(preprocessor.data.head(10))
    pipeline = LLMPipeline(str(sft_params.finetuned_model_path),
                           dataset, max_length=120,
                           batch_size=64,
                           device='cpu')

    # pipeline.tokenizer = finetuned_tokenizer

    pipeline.analyze_model()

    _sample_infer = pipeline.infer_sample(dataset[0])

    predictions_path = Path(PROJECT_ROOT / 'lab_8_sft' / 'dist' / 'predictions.csv')
    if not predictions_path.parent.exists():
        predictions_path.parent.mkdir()
    pipeline.infer_dataset().to_csv(predictions_path, index=False)
    evaluator = TaskEvaluator(predictions_path, settings.parameters.metrics)
    result = evaluator.run()
    print(result)

    assert result is not None, "Finetuning does not work correctly"


if __name__ == "__main__":
    main()
