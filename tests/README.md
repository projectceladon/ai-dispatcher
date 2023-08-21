# Testing AI Dispatcher

## Steps to run test
export AID_DIR="$(dirname $PWD)"
export PYTHONPATH="$PYTHONPATH:$AID_DIR:$AID_DIR/services/rawTensor/"
python3 test.py
python3 fuzz.py -atheris_runs=5000
python3 -m coverage html
python3 -m coverage json
