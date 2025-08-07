# Deployment Guide

This guide covers deployment options for the LLM-Powered Intelligent Query-Retrieval System.

## Prerequisites

- Python 3.8+
- 8GB+ RAM (recommended for FAISS indexing)
- OpenAI API key (optional - system has fallback)
- Docker (for containerized deployment)

## Quick Start (Local Development)

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment**:

   ```bash
   python start.py setup
   ```

3. **Configure API key** (optional):

   ```bash
   # Edit .env file
   OPENAI_API_KEY=your_actual_api_key_here
   ```

4. **Start server**:

   ```bash
   python start.py serve
   ```

5. **Test the system**:
   ```bash
   python start.py test
   ```

## Docker Deployment

### Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]
```

### Build and Run

```bash
# Build image
docker build -t hackrx-query-system .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key \
  hackrx-query-system
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

1. Launch EC2 instance (t3.large or larger)
2. Install Python and dependencies
3. Configure security groups (port 8000)
4. Use systemd service for auto-restart

#### ECS Deployment

```yaml
# ecs-task-definition.json
{
  "family": "hackrx-query-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions":
    [
      {
        "name": "hackrx-app",
        "image": "your-registry/hackrx-query-system:latest",
        "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
        "environment": [{ "name": "OPENAI_API_KEY", "value": "your_api_key" }],
        "logConfiguration":
          {
            "logDriver": "awslogs",
            "options":
              {
                "awslogs-group": "/ecs/hackrx-query-system",
                "awslogs-region": "us-east-1",
                "awslogs-stream-prefix": "ecs",
              },
          },
      },
    ],
}
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/hackrx-query-system

# Deploy to Cloud Run
gcloud run deploy hackrx-query-system \
  --image gcr.io/PROJECT_ID/hackrx-query-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --set-env-vars OPENAI_API_KEY=your_api_key
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name hackrx-rg --location eastus

# Deploy container
az container create \
  --resource-group hackrx-rg \
  --name hackrx-query-system \
  --image your-registry/hackrx-query-system:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your_api_key
```

## Production Configuration

### Environment Variables

```bash
# Production settings
DEBUG=False
LOG_LEVEL=WARNING
MAX_FILE_SIZE=100MB
CACHE_TTL=7200

# Performance tuning
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Faster model
MAX_CONTEXT_LENGTH=2000           # Reduce for speed
MAX_TOKENS=1000                   # Reduce for cost
```

### Systemd Service (Linux)

Create `/etc/systemd/system/hackrx-query-system.service`:

```ini
[Unit]
Description=HackRx Query System
After=network.target

[Service]
Type=simple
User=hackrx
WorkingDirectory=/opt/hackrx-query-system
Environment=PATH=/opt/hackrx-query-system/venv/bin
ExecStart=/opt/hackrx-query-system/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable hackrx-query-system
sudo systemctl start hackrx-query-system
```

## Monitoring and Logging

### Health Checks

The system provides several endpoints for monitoring:

- `GET /health` - Basic health check
- `GET /api/v1/status` - Detailed system status

### Logging Configuration

```python
# In main.py, configure structured logging
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_entry)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

### Metrics Collection

Consider integrating with:

- Prometheus + Grafana
- AWS CloudWatch
- Google Cloud Monitoring
- Azure Monitor

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Use HTTPS in production
- [ ] Implement rate limiting
- [ ] Validate input sizes
- [ ] Use secure API key storage (AWS Secrets Manager, etc.)
- [ ] Enable CORS properly
- [ ] Implement request logging
- [ ] Use container security scanning
- [ ] Regular dependency updates

### API Key Security

```bash
# Use AWS Secrets Manager
export OPENAI_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id hackrx/openai-key \
  --query SecretString --output text)
```

## Performance Optimization

### Scaling Considerations

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Caching**: Implement Redis for document caching
3. **Database**: Use PostgreSQL for metadata storage
4. **CDN**: Cache static responses

### Load Balancer Configuration (nginx)

```nginx
upstream hackrx_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://hackrx_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**:

   ```bash
   # Check memory usage
   free -h
   # Reduce model size or chunk size
   ```

2. **API Rate Limits**:

   ```bash
   # Monitor OpenAI API usage
   # Implement exponential backoff
   ```

3. **Performance Issues**:
   ```bash
   # Profile the application
   python -m cProfile main.py
   ```

### Log Analysis

```bash
# Monitor logs
tail -f /var/log/hackrx-query-system.log

# Search for errors
grep -i error /var/log/hackrx-query-system.log
```

## Support

For deployment issues:

1. Check the logs first
2. Verify environment variables
3. Test with curl commands
4. Check system resources
5. Review the troubleshooting section

---

This deployment guide covers most common scenarios. Adjust configurations based on your specific requirements and infrastructure.
