# CenterNet (WATonomous version)

This repo is derived from the original [CenterNet repo](https://github.com/xingyizhou/CenterNet). The original README can be found in [README.original.md](README.original.md).

## Getting Started

1. Initialize submodules:

```bash
git submodule update --init --recursive
```

2. Make sure `/etc/docker/daemon.json` contains `"default-runtime": "nvidia"` as a top-level property. For example:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

If changes are made, restart the docker daemon:

```bash
sudo service docker restart
```

3. Start the container. Some assumptions have been made in `docker-compose.yml` (e.g. where the data is stored). Edit the file if needed.

```bash
./generate-dot-env.sh
docker-compose up -d
```

4. Open a shell

```bash
docker-compose exec dev /bin/bash
cd /project
```

