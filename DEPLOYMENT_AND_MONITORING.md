# Deployment & Monitoring Guide (Steps 7–8)

## Prereqs
- Docker, Python 3.12+, kubectl, and either Minikube/Docker Desktop K8s or a cloud cluster.
- Trained model present at `models/log_reg_pipeline.joblib` (run `python -m src.model_train` as in README).

## Build & Run Locally (Docker)
```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"values":[63,1,3,145,233,1,0,150,0,2.3,0,0,1]}]}'
```
Health and metrics:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

## Kubernetes Deployment (Minikube or Docker Desktop)
1) Build image available to cluster. For Minikube:
```bash
eval "$(minikube docker-env)"
docker build -t heart-api .
```
For cloud K8s, push to a registry and update `k8s/deployment.yaml` image.

2) Apply manifests:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get svc heart-api
```

3) Access the service:
- If `LoadBalancer` on Minikube: `minikube service heart-api --url`
- Or port-forward: `kubectl port-forward svc/heart-api 8000:8000`

4) Test endpoints:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"values":[63,1,3,145,233,1,0,150,0,2.3,0,0,1]}]}'
```

5) Ingress (optional): enable an ingress controller (e.g., `minikube addons enable ingress`) and set DNS/hosts entry for `heart-api.local`; then hit `http://heart-api.local/`.

## Monitoring & Logging
- API logs: structured to STDOUT via FastAPI middleware; view with `kubectl logs -f deploy/heart-api`.
- Prometheus: scrape `http://heart-api:8000/metrics` (or via port-forward). Metrics include `api_requests_total` and `api_request_latency_seconds`.
- Quick local scrape example:
```bash
kubectl port-forward svc/heart-api 8000:8000 &
curl http://127.0.0.1:8000/metrics
```
- Grafana: add Prometheus as a data source pointing to the scrape endpoint; build a dashboard on the above metrics.

## Sharing/Export
- To share the repo as a zip:
```bash
zip -r mlops_assignment.zip .
```
- Artifact outputs: `models/` (saved pipeline), `reports/` (metrics), `mlruns/` (MLflow runs), `k8s/` (manifests).
