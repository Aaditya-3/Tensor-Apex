#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-business-policy-env:smoke}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
mkdir -p "$ARTIFACT_DIR"

docker build -t "$IMAGE_TAG" .
CONTAINER_ID="$(docker run -d -p 7860:7860 -p 7861:7861 "$IMAGE_TAG")"

cleanup() {
  docker logs "$CONTAINER_ID" > "$ARTIFACT_DIR/docker.log" 2>&1 || true
  docker rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in {1..30}; do
  if curl -fsS http://127.0.0.1:7860/health > "$ARTIFACT_DIR/health.json"; then
    break
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:7860/tasks > "$ARTIFACT_DIR/tasks.json"
curl -fsS -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"easy"}' > "$ARTIFACT_DIR/reset.json"
curl -fsS -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"categorize","reasoning":"Docker smoke route check.","category":"billing"}}' \
  > "$ARTIFACT_DIR/step.json"

cat > "$ARTIFACT_DIR/docker_runtime_proof.txt" <<EOF
docker_image=${IMAGE_TAG}
container_id=${CONTAINER_ID}
health_checked=true
tasks_checked=true
reset_checked=true
step_checked=true
EOF

echo "Docker smoke check completed."
