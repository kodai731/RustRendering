FROM ubuntu:24.04

ARG BLENDER_VERSION=5.1.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl xz-utils \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
        libgl1 libegl1 libxkbcommon0 libsm6 libice6 \
        libxcursor1 libxrandr2 libxinerama1 libxkbcommon-x11-0 \
        libglx-mesa0 libgl1-mesa-dri \
        libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN BLENDER_SERIES="$(echo "${BLENDER_VERSION}" | cut -d. -f1,2)" \
    && curl -L --fail --retry 3 --retry-delay 2 -o /tmp/blender.tar.xz \
        "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
    && mkdir -p /opt/blender \
    && tar -xJf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && /opt/blender/blender --version

ENV PATH="/opt/blender:${PATH}"
WORKDIR /workspace
