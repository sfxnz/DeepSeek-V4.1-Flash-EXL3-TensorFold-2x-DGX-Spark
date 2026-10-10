# TensorFold with the deepseek_v41 CUDA family on NGC PyTorch for GB10 (sm_121, aarch64).
# The CUDA extensions JIT-compile at the first start on the GPU, not here; run.sh mounts a host cache per
# TF_SHA so only the first boot pays for them.
ARG BASE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE}
# The fork branch dsv41-recipe-engine5: the family of upstream PRs 390 and 391, the device keyed draw for top_k-off
# rows, the speed units C1-C5, C7, C8, concurrent requests (--parallel 1 to 4, units N0-N10), upstream's final Python
# line (v0.6.6, branch python-0.6), and the decode units P4, P2, P10, P1, P6b. Upstream's main is the Zig engine,
# which has no DeepSeek-V4 family (issue 286), so the recipe pins the fork. The package ships the family's .cu and
# .cpp sources (package data) and compiles none of them here.
ARG TF_REPO=https://github.com/sfxnz/TensorFold.git
ARG TF_SHA=19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb
# TensorFold's runtime deps at the versions expected in the base; pip may not change any installed package (checked below).
COPY constraints.txt /opt/tensorfold-constraints.txt
# Install from git at the full commit, then check the commit pip recorded (direct_url.json) and that every other
# package, torch and triton included, is still the base's. The family needs no extra: no grammar, no vision.
RUN pip freeze | grep -viE '^tensorfold[ =@]' | sort > /opt/base-freeze.txt \
 && env -u HF_TOKEN pip install --no-cache-dir -c /opt/tensorfold-constraints.txt "tensorfold @ git+${TF_REPO}@${TF_SHA}" \
 && pip freeze | grep -viE '^tensorfold[ =@]' | sort | diff /opt/base-freeze.txt - \
 && python -c "import json, importlib.metadata as md; \
d = json.loads(md.distribution('tensorfold').read_text('direct_url.json')); \
assert d['vcs_info']['commit_id'] == '${TF_SHA}', d; assert d['url'] == '${TF_REPO}', d; \
print('tensorfold', md.version('tensorfold'), d['url'], d['vcs_info']['commit_id'])" \
 && python -c "import tensorfold.families.deepseek_v41 as f; assert f.MODELS == ('sfxnz/DeepSeek-V4.1-Flash-EXL3',), f.MODELS; \
import torch, triton; print('deepseek_v41', f.__file__, 'torch', torch.__version__, 'triton', triton.__version__)" \
 && for b in tensorfold ninja nvcc; do command -v "$b" || exit 1; done \
 && printf '%s@%s\n' "${TF_REPO}" "${TF_SHA}" > /opt/tensorfold-sha
LABEL tensorfold.sha=${TF_SHA} tensorfold.repo=${TF_REPO}
ENV PYTHONUNBUFFERED=1 TENSORFOLD_NO_UPDATE_CHECK=1 HF_HUB_OFFLINE=1
