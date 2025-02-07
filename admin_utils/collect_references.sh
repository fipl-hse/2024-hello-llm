# Run: nohup bash admin_utils/collect_references.sh > references.log 2>&1 &
# Monitor: ps -ef | grep collect_references

set -ex

source venv/bin/activate

export PYTHONPATH=$(pwd)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# python admin_utils/get_datasets_analytics.py

# python admin_utils/get_model_analytics.py

# python admin_utils/get_inference_analytics.py

python admin_utils/get_references.py
