rm -rf build core.* deep_ep.egg-info/ dist/
rm -rf deep_ep/__pycache__ tests/__pycache__
NVSHMEM_DIR=/root/paddlejob/workspace/env_run/output/lzy/lidong/nvshmem python setup.py install

# rm -rf deep_ep_cpp.cpython-38-x86_64-linux-gnu.so
# ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so