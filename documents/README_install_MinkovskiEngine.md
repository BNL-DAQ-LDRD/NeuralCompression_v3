# Install `MinkowskiEngine`
Let's face it, installing `MinkowskiEngine` is a pain in you-know-where. But
its performance is splendid (better than other alternatives I tried). And what
could be more fun than pulling some hair off and figuring something out.

So here I present how I made it work on a `Ubuntu2004 x86_64` server and
a `Ubuntu2204 x86_64` server that I have `sudo` privilege to.
I will provide my understanding of several specifities of `MinkowskiEngine`
installation according to my limited understanding of the trade, so feel
absolutely free to correct me if I say something stupid.

Also, please provide your successful experience on installing `MinkowskiEngine`
on your machine. I am sure someone out there will thank you wholeheartedly
(and may even worship you as a god).

### :boom::boom:Things that are NOT going to work without hacking the `MinkowskiEngine` source code :boom::boom:
With the current version of `MinkowskiEngine`,
- **Python version >= 3.12 does NOT work!**
- **NVCC version >= 12 does NOT work!**

I mean, don't even think about it, unless you are prepared and determined to
modify the source code (in both Python and C++)!

> Python version >= 3.12 doesn't work because the `numpy.distutils` is
discontinued for Python >= 3.12
(Ref. [https://numpy.org/doc/stable/reference/distutils.html](https://numpy.org/doc/stable/reference/distutils.html)),
however, the package is needed in `setup.py` of the project.

> `MinkowskiEngine` only supports `CUDA` 10.X and `CUDA` 11.X. Well, I
understand that sometimes, we will have luck with a slight different
version of things, but this is __NOT__ the case for `MinkowskiEngine` which
is found directly on `CUDA` programming. And trust me, I __tried__, it does
not work because of differences between versions of `CUDA` (C++) code.

### Something that may work
As I mentioned before, I only tested the following approach on my
`Ubuntu2004` and `Ubuntu2204` `x86_64` servers to which I have `sudo`
privilege. This means I cannot guarantee the following steps will to work
on your platform. But if you also make it work, please share your
experience!

I installed `MinkowskiEngine` in six steps. For each command I used below,
I will give a way to check whether it runs successfully. If any of the command
is not successfully executed, I think you are doomed. 

I am joking. Please let me know what problem you encounter,
what is your understanding of the problem, and more exciting,
how you solve the problem.

#### Step 1: Install `CUDA` toolkit version 11.8
> The "11.8" here can be any "10.X" or "11.X". But please change the
following code accordingly.

Run
```
sudo apt install cuda-11-8
```
Then run
```
ls /usr/local/
```
to check whether you have a folder called `/usr/local/cuda-11.8`, if so, continue.


#### Step 2: Set the correct version of NVCC
Here I will present a slightly overkilling solution to change `NVCC` version.
Since you are reading this document, I feel you might find this more
convoluted solution useful some time in the future.

Run
```
source scripts/use_cuda.sh 11.8
```
The version number (11.8 here) is provided as an argument. You can change it
to another number if a package need a specific version of `CUDA` toolkit.

To check whether the correct `NVCC` version is running, run
```
nvcc --version
```
This is my output
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
and you see the "11.8" there.

**IMPORTANT NOTE:** `NVCC` is not just changed for a conda environment. 
This means two things,
1. If you need to do something else on the same machine that needs another
   version of `NVCC`, run `source scripts/use_cuda.sh <version>` to change it.
1. Always make sure you have the currect version of `NVCC` before doing
   anything with `MinkowskiEngine`.

#### Step 3: Create a `conda` environment
Here I suppose you already have `conda` installed and running.
But if not, this website [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
may be helpful.

I'd like to called the environment `py311-me`, and if you are Okay with
my poor taste, run
```
conda create --name py311-me python=3.11
```
As explained before, don't try Python=3.12 or higher if you are not prepared
to hack the source code (`setup.py` in particular).

Activate the environment by running
```
conda activate py311-me
```
From now on, let us stay in this conda environment.

Finally let us install [`OpenBLAS`](https://github.com/OpenMathLib/OpenBLAS/wiki), 
an optimized Basic Linear Algebra Subprograms (BLAS) library.

> NOTE: This is NOT optional, MinkowskiEngine absolutely needs it!

```
conda install openblas-devel -c anaconda
```

#### Step 4: Install `PyTorch` with `CUDA` 11.8 as the compute platform
> According to `MinkowskiEngine`, `PyTorch` must be compiled with
the same version of `NVCC` that is running, but if my memory is reliable,
it might still work if your `PyTorch` is compiled by another version of NVCC.

The following image shows how I chose my version of `PyTorch` on 05/29/2024
from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
However, we all know that this website update really frequently.
So if you can no longer find `PyTorch` with Compute Platform = `CUDA` 11.8, you
may find your luck here [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).

<img src="https://github.com/pphuangyi/sparse_poi/assets/22546248/fcc77784-8905-47ad-af9c-647b8a92a288" alt="install_torch_on_2024-05-29_with_cuda-11-8" width="800">

To check whether `PyTorch` is successfully installed and compiled with
the correct version of NVCC, run
```
python -c "import torch; print(torch.version.cuda)"
```
The output should be "11.8".

**IMPORTANT NOTE:** You may have to reboot your machine before `torch` can
find the GPU(s). To check whther `torch` can find the GPU(s), run
```
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is anything other than "True", you may have to reboot.
If rebooting doesn't work, and you are sure there are GPU(s) on your machine,
Google or ask ChatGPT for solutions.

If `torch` cannot find GPU(s), `MinkowskiEngine` will be installed in
the CPU-only mode by default.

#### Step 5: Install `MinkowskiEngine`
Get a clone of the `MinkowskiEngine` by running
```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
```
Then
```
cd MinkowskiEngine
```
and run
```
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```
> I told you so, MinkowskiEngine needs `OpenBLAS` 😉

I recommend creating fork of the repo and clone the fork.
You may want to tweak it and make it better some time in the future. I surely do 😃

#### Step 6 (Optional): Unit tests
We got a problem here, the unit tests provided by the `MinkowskiEngine` repo
(stored in 'tests/python') are very very buggy. I am not sure whether they are
outdated and not supposed to be used, or the author put them there just to test
how  determined you are to use `MinkowskiEngine`. I think they are also good
learning material.

I am making correction to them in my own [fork of the repo](https://github.com/pphuangyi/MinkowskiEngine).
And I will document the work here
1. `tests/python/broadcast.py` needs a minor fix. `MinkowskiEngine/utils/gradcheck.py` also need a minor fix.
1. `tests/python/chwise_conv.py` needs a minor fix.
1. run `pip install open3d' before running 'tests/python/convolution.py'

To run a test, go back to the project root and run
```
python -m unittest ./tests/python/<test_name>.py
```

## Get prepared for Point-of-Interest Detection with Sparse Convolution

### Clone the repository
Get a clone of the `sparse_poi` by running
```
git clone https://github.com/pphuangyi/sparse_poi/tree/main
```
then
```
cd sparse_poi
```
Install the package by running
```
python setup.py develop
```
Again, please consider forking the repo and clone the fork.
Let us make it better together.

### Download the Time-Projection Chamber data
(to be completed)


### Set up environment variables


Set the root to the data by running
```
export DATAROOT=/path/to/your/data
```
Then you can check whether the data root is set correct by running the
unit test of the TPC dataset API
```
python -m unittest ./tests/test_dataset_tpc.py
```

This is optional, but if you want to develop your own code and
use `pylint` for disciplined coding as I do, please add the following
line so that `pylint` can locate your packages.
```
export PYTHONPATH=${CONDA_PREFIX}/lib/python3.11/site-packages:$PYTHONPATH
```
