# Deployment Guide

This guide covers best practices for deploying DataQA in production or at scale-locally, with Docker, or in the cloud.

---

## 1. Local Deployment

For development, testing, or small-scale use, you can run DataQA directly on your machine:

```bash
python -m dataqa.examples.cib_mp.agent.cwd_agent
```
- Make sure all environment variables (e.g., LLM credentials) are set in your shell or `.env` file.
- For long-running services, consider using [supervisord](http://supervisord.org/) or [systemd](https://www.freedesktop.org/wiki/Software/systemd/) to manage your process.

---

## 2. Docker Deployment

**Recommended for reproducibility and easy scaling.**

**Sample `Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry install --no-dev

COPY your/agent/config and data assets
COPY agent.yaml /app/agent.yaml
COPY data/ /app/data/

# Set environment variables (or use Docker secrets)
ENV AZURE_OPENAI_API_KEY="your-azure-openai-api-key"
ENV OPENAI_API_BASE="https://your-azure-openai-endpoint.net/"

CMD ["python", "-m", "dataqa.examples.cib_mp.agent.cwd_agent"]
```

**Build and Run**
```bash
docker build -t dataqa-agent .
docker run --env-file .env dataqa-agent
```

---

## 3. Cloud Deployment

You can deploy DataQA on any cloud platform (AWS, Azure, GCP) using:
- **Containers:** (ECS, AKS, GKE, etc.)
- **Orchestration:** (Kubernetes, Helm charts, k8s manifests)
- **Serverless:** (for stateless batch jobs)

**Best Practices:**
- Use a secrets manager (AWS Secrets Manager, Azure Key Vault) for credentials.
- Use cloud storage (S3, Azure Blob, GCS) for large data assets.
- Set resource limits and autoscaling for production workloads.

---

## 4. Environment Variables & Secrets

**Never hard-code secrets in your code or YAML!**
- Use environment variables, `.env` files, or secret managers.
- In Docker/K8s, use `--env-file`, `envFrom`, or secret mounts.

**Example: Kubernetes Secret**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: dataqa-secrets
type: Opaque
data:
  AZURE_OPENAI_API_KEY: <base64-encoded-key>
```

---

## 5. Scaling & Performance

- Use async clients and batch processing for high throughput.
- Monitor memory and CPU usage; adjust container/pod resources as needed.
- For large workloads, use distributed task queues (e.g., Celery, Ray).

---

## 6. Monitoring & Logging

- Centralize logs using tools like ELK, Datadog, or cloud-native logging.
- Integrate with OpenTelemetry or Prometheus for metrics and tracing.

---

## 7. Updating & Maintenance

- Keep your dependencies up to date (`pip install --upgrade aicoelin-dataqa`).
- Use versioned configs and assets for reproducibility.
- Regularly rotate secrets and review access controls.

---

## Next Steps

- [Troubleshooting Guide](troubleshooting.md)
- [API Reference](../reference/agent.md)
- [FAQ](faq.md)
