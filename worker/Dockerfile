# Hob Bridge serverless worker: Automatic1111 + ControlNet + RunPod handler
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pinned so image rebuilds stay reproducible — bump deliberately.
# A1111 v1.10.1: last tagged release of the (low-activity) upstream repo.
RUN git clone --depth 1 --branch v1.10.1 \
    https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui

WORKDIR /sd-webui
# github.com/Stability-AI/stablediffusion, v1.10.1's default for this dependency,
# no longer clones. A1111's dev branch moved to this fork at the same commit;
# launch.py checks out the pinned SHA (STABLE_DIFFUSION_COMMIT_HASH, unchanged),
# so the content is the original's.
ENV STABLE_DIFFUSION_REPO=https://github.com/w-e-w/stablediffusion.git
# Bake A1111's dependencies into the image so cold starts skip dependency
# resolution; --exit stops launch.py once prepare_environment() is done.
RUN python launch.py --skip-torch-cuda-test --exit

# ControlNet pinned to main @ 56cec5b (2024-07-25, "fix version parsing"),
# shallow-fetched by exact commit. Added after A1111's own setup on purpose:
# its requirements are loose, and installed unconstrained they pulled numpy 2
# (breaking scikit-image) and a mediapipe without `solutions` (breaking
# controlnet_aux), so ControlNet failed to load. They are installed under
# A1111's own pins (numpy==1.26.2, protobuf==3.20.0, ...) plus mediapipe pinned
# to 0.10.11: the newest that fits protobuf 3.20 and still ships `mp.solutions`.
RUN git init /sd-webui/extensions/sd-webui-controlnet \
    && git -C /sd-webui/extensions/sd-webui-controlnet remote add origin \
        https://github.com/Mikubill/sd-webui-controlnet.git \
    && git -C /sd-webui/extensions/sd-webui-controlnet fetch --depth 1 \
        origin 56cec5b2958edf3b1807b7e7b2b1b5186dbd2f81 \
    && git -C /sd-webui/extensions/sd-webui-controlnet checkout -B pinned FETCH_HEAD
RUN cp requirements_versions.txt /constraints.txt \
    && echo "mediapipe==0.10.11" >> /constraints.txt \
    && pip install --no-cache-dir -c /constraints.txt \
        -r extensions/sd-webui-controlnet/requirements.txt

# The handler (RunPod SDK: fastapi, aiohttp, websockets, ...) gets its own
# virtualenv so it cannot move A1111's pinned packages. It only starts A1111 as
# a subprocess (`python` = the conda interpreter on PATH), so they never share imports.
RUN python -m venv /handler-venv
COPY requirements.txt /handler/requirements.txt
RUN /handler-venv/bin/pip install --no-cache-dir -r /handler/requirements.txt

COPY handler.py /handler/handler.py
WORKDIR /handler

CMD ["/handler-venv/bin/python", "-u", "handler.py"]
