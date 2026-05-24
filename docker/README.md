# Docker

Optionally, you can run Philosopher's Stone inside Docker.

Build the image from the repository root:

    docker build -f docker/Dockerfile -t philosophers-stone .

Run the bundled sample files:

    docker run --rm philosophers-stone

Run your own files by mounting a host folder that contains your manifest and EEG data:

    docker run --rm \
      -v /path/to/data:/data \
      philosophers-stone \
      --manifest_csv /data/my_manifest.csv \
      --outdir /data/output

Important: file paths inside the manifest must use container paths such as `/data/file.edf`, not host-only paths such as `/home/user/file.edf`.
