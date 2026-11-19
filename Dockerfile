FROM rust:1-bookworm AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/bin && \
    echo "" > src/lib.rs && \
    for b in web fix sync spectro; do echo "fn main(){}" > src/bin/$b.rs; done && \
    cargo build --release && \
    rm -rf src
COPY src ./src
RUN touch src/lib.rs src/bin/*.rs && cargo build --release && \
    strip target/release/dubalign-web target/release/dubalign-fix target/release/dubalign-sync target/release/dubalign-spectro

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg mkvtoolnix fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/dubalign-web \
    /app/target/release/dubalign-fix \
    /app/target/release/dubalign-sync \
    /app/target/release/dubalign-spectro /usr/local/bin/
COPY static ./static
ENV MEDIA_DIR=/media OUTPUT_DIR=/output WORK_DIR=/work PORT=7341
CMD ["dubalign-web"]
