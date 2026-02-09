#!/bin/bash
# Deploy Heavy Runner to Cloud Run Jobs
#
# Usage:
#   ./deploy.sh                    # Uses defaults from .env or prompts
#   ./deploy.sh --project my-proj  # Override project
#   ./deploy.sh --timeout 120m     # Override Cloud Run task timeout
#   ./deploy.sh --create           # Create new job (instead of update)

set -e

# Defaults
PROJECT_ID="${HEAVY_RUNNER_PROJECT:-}"
REGION="${HEAVY_RUNNER_REGION:-us-central1}"
BUCKET="${HEAVY_RUNNER_BUCKET:-}"
JOB_NAME="${HEAVY_RUNNER_JOB:-heavy-runner}"
REPO="heavy-runner"
IMAGE="heavy-train"
MEMORY="32Gi"
CPU="8"
TIMEOUT="${HEAVY_RUNNER_TASK_TIMEOUT:-120m}"
CREATE_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --job)
            JOB_NAME="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --create)
            CREATE_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [[ -z "$PROJECT_ID" ]]; then
    echo "ERROR: PROJECT_ID not set. Use --project or set HEAVY_RUNNER_PROJECT"
    exit 1
fi

echo "========================================"
echo "Deploying Heavy Runner"
echo "========================================"
echo "Project:  $PROJECT_ID"
echo "Region:   $REGION"
echo "Job:      $JOB_NAME"
echo "Memory:   $MEMORY"
echo "CPU:      $CPU"
echo "Timeout:  $TIMEOUT"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build and push image
echo ""
echo "Building and pushing container image..."
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

gcloud builds submit \
    --project "$PROJECT_ID" \
    --tag "$IMAGE_URI" \
    --quiet

echo ""
echo "Image pushed: $IMAGE_URI"

# Create or update job
if $CREATE_MODE; then
    echo ""
    echo "Creating Cloud Run Job..."
    gcloud run jobs create "$JOB_NAME" \
        --project "$PROJECT_ID" \
        --image "$IMAGE_URI" \
        --region "$REGION" \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --max-retries 0 \
        --task-timeout "$TIMEOUT" \
        --set-env-vars "PYTHONUNBUFFERED=1"
else
    echo ""
    echo "Updating Cloud Run Job..."
    gcloud run jobs update "$JOB_NAME" \
        --project "$PROJECT_ID" \
        --image "$IMAGE_URI" \
        --region "$REGION" \
        --memory "$MEMORY" \
        --cpu "$CPU" \
        --task-timeout "$TIMEOUT"
fi

echo ""
echo "========================================"
echo "Deploy complete!"
echo "========================================"
echo ""
echo "Test with:"
echo "  gcloud run jobs execute $JOB_NAME --region $REGION \\"
echo "    --set-env-vars INPUT_URI=gs://bucket/inputs/test.json,OUTPUT_URI=gs://bucket/outputs/test/"
