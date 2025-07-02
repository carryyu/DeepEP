export WORLD_SIZE=2
export MASTER_ADDR="10.95.226.76"
export MASTER_PORT="8010"
export RANK=0
# python tests/test_low_latency.py
python tests/test_low_latency_two_stage.py