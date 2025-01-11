set -x

echo $1
if [[ "$1" == "smoke" ]]; then
  DIRS_TO_CHECK=(
    "config"
    "seminars"
  )
else
  DIRS_TO_CHECK=(
    "config"
    "seminars"
    "core_utils"
    "lab_7_llm"
    "lab_8_llm"
    "reference_lab_classification"
    "reference_lab_generation"
    "reference_lab_nli"
    "reference_lab_nmt"
    "reference_lab_open_qa"
    "reference_lab_summarization"
  )
fi

export PYTHONPATH=$(pwd)

python -m black "${DIRS_TO_CHECK[@]}"

python -m pylint "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python config/static_checks/check_docstrings.py

python -m flake8 "${DIRS_TO_CHECK[@]}"

if [[ "$1" != "smoke" ]]; then
  python -m pytest -m "mark10 and lab_7_llm"
  python -m pytest -m "mark10 and lab_8_llm"
fi
