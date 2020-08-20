# PyTorch binding for WarpRNNT

This package provides PyTorch kernels that wrap the WarpRNNT library. 

## Installation

Install [PyTorch](https://github.com/pytorch/pytorch#installation).

`WARP_RNNT_PATH` should be set to the location of a built WarpRNNT
(i.e. `libwarprnnt.so`).  This defaults to `../build`, so from within a
new warp-transducer clone you could build WarpRNNT like this:

```bash
git clone https://github.com/HawkAaron/warp-transducer
cd warp-transducer
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_RNNT_PATH` to wherever you have `libwarprnnt.so`
installed. If you have a GPU, you should also make sure that
`CUDA_HOME` is set to the home cuda directory (i.e. where
`include/cuda.h` and `lib/libcudart.so` live). For example:

```
export CUDA_HOME="/usr/local/cuda"
```

Now install the bindings: (Please make sure the GCC version >= 4.9)
```
cd pytorch_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 (as recommended by pytorch):
```
cd ../pytorch_binding
python setup.py install
cd ../build
cp libwarprnnt.dylib /Users/$WHOAMI/anaconda3/lib
```
This will resolve the library not loaded error. This can be easily modified to work with other python installs if needed.

Example to use the bindings below.

```python

import torch
from warprnnt_pytorch import RNNTLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu' # whether use GPU version

rnnt_loss = RNNTLoss()
rnnt_loss = rnnt_loss.to(device)

acts = torch.FloatTensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.6, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.8, 0.1]],
                            [[0.1, 0.6, 0.1, 0.1, 0.1],
                            [0.1, 0.1, 0.2, 0.1, 0.1],
                            [0.7, 0.1, 0.2, 0.1, 0.1]]]])
labels = torch.IntTensor([[1, 2]])
act_length = torch.IntTensor([2])
label_length = torch.IntTensor([2])
probs_sizes = torch.IntTensor([2])
label_sizes = torch.IntTensor([2])


acts = acts.to(device)

acts.requires_grad_(True)

labels = labels.to(device)
act_length = act_length.to(device)
label_length = label_length.to(device)

loss = rnnt_loss(acts, labels, act_length, label_length)
loss.backward()

```

## Documentation

```python
RNNTLoss(size_average=True, blank_label=0):
    """
    size_average (bool): normalize the loss by the batch size (default: True)
    blank_label (int): blank label index
    """

forward(acts, labels, act_lens, label_lens):
    """
    acts: Tensor of [batch x seqLength x (labelLength + 1) x outputDim] containing output from network
     (+1 means first blank label prediction)
    labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    label_lens: Tensor of (batch) containing label length of each example
    """
```
