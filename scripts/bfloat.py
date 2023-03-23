from pkg_resources import packaging
import torch
import torch.distributed as dist
import torch.cuda.nccl as nccl

# Check bfloat support.
bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
)
print(f'bfloat support: {bfloat_support}')
