FROM jenkins/jenkins:2.555.2-jdk21
USER root

# ── System dependencies + Docker CLI ────────────────────────────────────────
RUN apt-get update && apt-get install -y \
        lsb-release ca-certificates curl git unzip python3 \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) \
       signed-by=/etc/apt/keyrings/docker.asc] \
       https://download.docker.com/linux/debian \
       $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
       | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── uv — installed system-wide ───────────────────────────────────────────────
# UV_PYTHON_INSTALL_DIR is set globally so the jenkins user shares the same
# Python installations that are pre-seeded here as root.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-pythons
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh \
    && uv python install 3.11 3.13 \
    && chmod -R a+rx /opt/uv-pythons

# ── gitleaks ─────────────────────────────────────────────────────────────────
ARG GITLEAKS_VERSION=8.27.2
RUN curl -sSfL \
    "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VERSION}/gitleaks_${GITLEAKS_VERSION}_linux_x64.tar.gz" \
    | tar -xz -C /usr/local/bin gitleaks \
    && chmod +x /usr/local/bin/gitleaks

# ── sonar-scanner CLI ─────────────────────────────────────────────────────────
ARG SONAR_SCANNER_VERSION=6.2.1.4610
RUN curl -sSfL \
    "https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-${SONAR_SCANNER_VERSION}-linux-x64.zip" \
    -o /tmp/sonar-scanner.zip \
    && unzip /tmp/sonar-scanner.zip -d /opt/ \
    && ln -sf "/opt/sonar-scanner-${SONAR_SCANNER_VERSION}-linux-x64/bin/sonar-scanner" \
       /usr/local/bin/sonar-scanner \
    && rm /tmp/sonar-scanner.zip

# ── Jenkins plugins ───────────────────────────────────────────────────────────
COPY plugins.txt /usr/share/jenkins/plugins.txt
USER jenkins
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/plugins.txt
