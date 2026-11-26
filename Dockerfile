# syntax=docker/dockerfile:1
# The service image, for linux/amd64 and linux/arm64. Every stage builds natively on
# the target architecture (release.yml runs one runner per architecture), so only the
# gitleaks download depends on it, through TARGETARCH.
FROM rust:1.100.0-alpine3.24 AS rust
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src
COPY migrations migrations
# target/ leaves in the same layer, so the registry cache never stores it.
RUN cargo build --release --locked --bin diffagent && mv target/release/diffagent / && rm -rf target
# gitleaks per architecture, each asset's sha256 from the release's checksums file.
FROM scratch AS gitleaks-amd64
ADD --checksum=sha256:551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb \
    https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz /gl.tgz
FROM scratch AS gitleaks-arm64
ADD --checksum=sha256:e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080 \
    https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_arm64.tar.gz /gl.tgz
# An architecture with no stage above fails here.
FROM gitleaks-${TARGETARCH} AS gitleaks
FROM golang:1.27.3-alpine3.24 AS go
COPY goctx /src/goctx
RUN CGO_ENABLED=0 GOTOOLCHAIN=local go build -C /src/goctx -trimpath -o /out/diffagent-goctx .
COPY --from=gitleaks /gl.tgz /tmp/gl.tgz
RUN tar -xzf /tmp/gl.tgz -C /out gitleaks
FROM alpine:3.24.3
# git and HTTPS trust the system CA store. No private CA is baked in: Postgres trusts
# webpki roots plus the sslrootcert that DATABASE_URL names, so mount a private CA.
# Alpine keeps only each package's current build, so a version pin breaks the build at its next update.
# hadolint ignore=DL3018
RUN apk add --no-cache git ca-certificates
# goctx reads this toolchain's export data, so both come from one stage.
COPY --from=go /usr/local/go /usr/local/go
COPY --from=go /out/ /usr/local/bin/
COPY --from=rust /diffagent /usr/local/bin/
ENV PATH=/usr/local/go/bin:$PATH HOME=/tmp TMPDIR=/tmp GOCACHE=/tmp/gocache GOMODCACHE=/tmp/gomodcache GOTMPDIR=/tmp
# Smoke without pipes (DL4006), on each architecture's own build: an empty request is
# goctx's exit 2; diffagent refuses an empty config.
RUN case "$(go version)" in *' go1.27.3 '*) ;; *) exit 1 ;; esac && gitleaks version && git --version \
 && { diffagent-goctx </dev/null 2>/dev/null; test $? -eq 2; } \
 && case "$(diffagent 2>&1)" in *'missing required config'*) ;; *) exit 1 ;; esac
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/diffagent"]
