# apex_example
Minimal example of NVIDIA Apex with Distributed Data Parallelism.

### run

```bash
sudo apt install python3-dev python3-pip virtualenv
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate

git clone https://github.com/enijkamp/apex_example.git
cd apex_example
pip3 install -r requirements.txt

cd ..
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd ../apex_example
python3 test_apex_distributed_spawn.py
```

### notes
* DistributedDataParallel seems to outperform DataParallel in general, even for local single-machine training without APEX. DataParallel relies on global interpreter lock sharing within a single process, which is slow.
* DeepSpeed relies on deprecated (and strongly discouraged) fp16 API, not supporting automatic mixed-precision (https://github.com/microsoft/DeepSpeed/issues/121).
* Apex fp16 O3 optimization level with DistributedDataParallel seems competitive with DeepSpeed throughput and memory allocation (only in single-machine configurations), while having the benenfit of being minimal, supporting the recommended unified API and automatic mixed precision.

### fixes
* Apex DataParallel issue: https://github.com/NVIDIA/apex/issues/227
* Apex C-compilation issue: https://github.com/NVIDIA/apex/issues/802
* Apex segfault gcc issue: https://github.com/NVIDIA/apex/issues/35