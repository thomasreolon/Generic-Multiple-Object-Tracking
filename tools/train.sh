set -x
set -o pipefail

num_nodes=${@:1}
args=$(cat tmp.args)

python -m torch.distributed.launch --nproc_per_node=${num_nodes} --use_env main.py ${args} --output_dir . |& tee -a output.log
