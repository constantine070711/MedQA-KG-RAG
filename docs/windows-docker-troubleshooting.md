# Windows Docker Desktop Socket Startup Failure

This note covers a Docker Desktop startup failure observed on Windows 11 with
Docker Desktop 4.87.0.

## Symptoms

Docker Desktop closes with an error similar to:

```text
starting services: initializing Ingest server: listening on
unix://C:/Users/<user>/AppData/Local/Docker/run/sailor-ingest.sock:
The file cannot be accessed by the system.
```

A later startup attempt may fail on the Secrets Engine instead:

```text
AppData/Local/docker-secrets-engine/engine.sock
```

Docker CLI may report:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

## What the C: Path Means

`%LOCALAPPDATA%\Docker\run` and
`%LOCALAPPDATA%\docker-secrets-engine` are runtime IPC directories. They are
not the Docker container data disk.

The container data location is controlled separately by Docker Desktop's disk
image location (`CustomWslDistroDir`). A runtime socket on C: does not mean that
containers or volumes were moved to C:.

## Cause

After Docker stops or restarts, Windows can leave Unix-domain socket entries as
inaccessible reparse points. Docker attempts to remove the old socket, receives
an error equivalent to `ERROR_CANT_ACCESS_FILE`, and aborts backend startup.

Changing the Docker disk image location while Docker was running was observed
immediately before the first failure. The exact filesystem-level reason for the
reparse point becoming inaccessible was not recorded by Docker, but the stale
runtime directories reproduce the failure consistently.

## Safe Recovery

Do not start with **Reset to factory defaults**. Factory reset removes
containers, images, and volumes.

1. Quit Docker Desktop.
2. Stop the WSL backend:

```powershell
wsl --shutdown
```

3. Move the stale runtime directories aside. Do not move the data disk:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$staleRoot = Join-Path $env:LOCALAPPDATA "Docker\stale-runtime\$stamp"
New-Item -ItemType Directory -Path $staleRoot -Force | Out-Null

$runPath = Join-Path $env:LOCALAPPDATA "Docker\run"
$secretsPath = Join-Path $env:LOCALAPPDATA "docker-secrets-engine"

if (Test-Path -LiteralPath $runPath) {
    Move-Item -LiteralPath $runPath -Destination (Join-Path $staleRoot "run")
}
if (Test-Path -LiteralPath $secretsPath) {
    Move-Item -LiteralPath $secretsPath -Destination (Join-Path $staleRoot "secrets-engine")
}

New-Item -ItemType Directory -Path $runPath -Force | Out-Null
```

4. Start Docker Desktop again and verify:

```powershell
docker info
docker ps -a
```

5. If the data disk location was changed, confirm it points to the directory
   containing the existing `disk\docker_data.vhdx`.

The stale directories contain inaccessible socket entries and may need to be
removed after a reboot or with a filesystem tool that can remove reparse points.
They do not contain container data.

## Recovering a Previous Data Disk

If a custom Docker data directory previously contained a
`disk\docker_data.vhdx`, preserve that file before changing any Docker setting.
Docker can reuse the existing data disk after the runtime socket cleanup and a
Docker restart.

Before probing an old data disk:

- Copy the `.vhdx` file to a separate location.
- Keep the current data disk untouched.
- Record the configured data directory in Docker Desktop settings.
- Run `docker ps -a` after startup to confirm which containers were restored.

## API Still Reports Neo4j Unavailable

Restarting Docker invalidates existing Neo4j driver connections held by a
long-running FastAPI process. A log entry such as
`ConnectionAbortedError(10053)` indicates a stale API connection, not a missing
graph.

Restart the FastAPI service after Docker is healthy, then verify a real query
returns `neo4j_unavailable: false` and graph evidence.

## Related

- [Deployment guide](deployment.md)
- [v2 architecture](v2-architecture.md)
