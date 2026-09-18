# Grafana Loki Logging Configuration

This directory contains the configuration files and components for centralized logging using Grafana and Loki in the Global Classifier project.

## Overview

We use **Grafana Loki** for centralized log aggregation and **Grafana** for log visualization and monitoring. This setup provides:

- **Centralized Logging**: All application logs are sent to Loki for storage and indexing
- **Real-time Monitoring**: Grafana dashboards provide real-time views of logs and metrics
- **Advanced Filtering**: Filter logs by service, log level, model ID, and other labels
- **Alerting**: Monitor error rates and system health
- **API-based Logging**: Direct HTTP API calls to Loki (no file dependencies)

## Architecture

```
┌─────────────────┐    HTTP API    ┌──────────┐    Query API    ┌─────────┐
│ Python Services │ ──────────────> │   Loki   │ <─────────────── │ Grafana │
│ (LokiLogger)    │                 │   :3100  │                  │  :4005  │
└─────────────────┘                 └──────────┘                  └─────────┘
                                          │
                                          ▼
                                    ┌──────────┐
                                    │ File     │
                                    │ Storage  │
                                    └──────────┘
```

## Components

### 1. **LokiLogger Class** (`loki_logger.py`)
- **Purpose**: Python logging class that sends logs directly to Loki API
- **Features**: 
  - Direct HTTP API calls (no file dependencies)
  - Automatic label generation (service, level, hostname, model_id)
  - Non-blocking, fire-and-forget logging
  - Console output for immediate feedback
  - Graceful error handling

### 2. **Loki Configuration** (`loki-config.yaml`)
- **Purpose**: Loki server configuration
- **Storage**: Filesystem-based with chunks and rules directories
- **Schema**: BoltDB shipper with filesystem object store
- **Port**: 3100 (HTTP), 9096 (gRPC)

### 3. **Grafana Datasources** (`grafana-datasources.yaml`)
- **Purpose**: Auto-provisioning of Loki as Grafana datasource
- **Connection**: Points to `http://loki:3100` (container network)
- **Default**: Set as the default datasource for new panels

### 4. **Grafana Dashboards** (`grafana-dashboards.yaml`)
- **Purpose**: Auto-provisioning of dashboards
- **Location**: Dashboards placed in `/etc/grafana/provisioning/dashboards`
- **Folder**: Organized under "RAG Module" folder

### 5. **RAG Module Dashboard** (`grafana-dashboard-deployment.json`)
- **Purpose**: Monitoring dashboard for RAG Module processes
- **Features**:
  - Time series graph showing log counts by level and service
  - Real-time log viewer with filtering
  - Template variables for service and log level filtering
  - 30-second auto-refresh

### 6. **Docker Compose** (`docker-compose-logging.yml`)
- **Purpose**: Container orchestration for logging stack
- **Services**: Grafana (port 4005) and Loki (port 3100)
- **Network**: Connected to `bykstack` network
- **Volumes**: Persistent storage for Grafana data and Loki chunks

## Usage

### Starting the Logging Stack
```bash
# Start logging services
docker-compose -f grafana-configs/docker-compose-logging.yml up -d

# Or include in development stack
docker-compose -f docker-compose-dev.yml up -d
```

### Using LokiLogger in Python
```python
from grafana_configs.loki_logger import LokiLogger

# Initialize logger with service name
logger = LokiLogger(service_name="model-deployment-orchestrator")

# Log with model context
logger.info(
    "Starting deployment",
    model_id="model123",
    current_env="testing",
    target_env="production",
)

# Log errors with extra context
logger.error(
    "Deployment failed", model_id="model123", error_code=500, step="model_loading"
)
```

### Accessing Grafana
1. **URL**: http://localhost:4005
2. **Credentials**: admin / admin123
3. **Dashboard**: Dashboards → RAG Module → "RAG Module Orchestrator"

### Log Filtering Examples
- **Service Filter**: `model-deployment-orchestrator`
- **Log Level Filter**: `ERROR`, `WARNING`, `INFO`, `DEBUG`
- **Model ID Filter**: Available in log content and labels
- **Time Range**: Last 1 hour (default), customizable

## Integration with CronManager

The `loki_logger.py` file is mounted into CronManager containers at the same location as the deployment scripts, allowing direct import:

```python
# In deployment_orchestrator.py
from loki_logger import LokiLogger

# Initialize with appropriate service name
logger = LokiLogger(service_name="model-deployment-orchestrator")
```

## Network Configuration

All services run on the `bykstack` Docker network:
- **Loki**: `loki:3100` (internal network)
- **Grafana**: `grafana:3000` → `localhost:4005` (external access)
- **Logger**: Uses `http://loki:3100` for container-to-container communication

## Log Retention

- **Storage**: Filesystem-based in container volumes
- **Retention**: Configurable in `loki-config.yaml`
- **Backup**: Consider backing up `/loki/chunks` for long-term retention

## Monitoring Best Practices

1. **Use Descriptive Service Names**: Help identify log sources
2. **Include Model Context**: Always pass `model_id` when available
3. **Structured Extra Fields**: Use consistent field names across services
4. **Error Context**: Include error codes, step information for debugging
5. **Log Levels**: Use appropriate levels (ERROR for failures, INFO for progress)

## Troubleshooting

### Loki Connection Issues
- Check if Loki container is running: `docker ps | grep loki`
- Verify network connectivity: `docker network inspect bykstack`
- Check Loki logs: `docker logs loki`

### Grafana Dashboard Issues
- Restart Grafana to reload dashboards: `docker restart grafana`
- Check provisioning: `docker logs grafana | grep provisioning`
- Validate JSON: `python -m json.tool grafana-dashboard-deployment.json`

### No Logs Appearing
- Verify LokiLogger is using correct URL (`http://loki:3100`)
- Check if services are on the same Docker network
- Test Loki API directly: `curl http://localhost:3100/ready`
