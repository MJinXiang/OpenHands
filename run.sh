# Set required environment variables
export SANDBOX_VOLUMES="/netcache/mengjinxiang/Project/OpenHands:/workspace:rw"  # See SANDBOX_VOLUMES docs for details
export LLM_MODEL="gpt-4o"
export LLM_API_KEY="sk-q8onsOJmkNHJmcMWc5QHKfhN60cVISeOyaq1OVvt2aD2j1BO"
export BASE_URL="https://api2.aigcbest.top/v1"
# export SANDBOX_SELECTED_REPO="owner/repo-name"  # Optional: requires GITHUB_TOKEN
# export GITHUB_TOKEN="your-token"  # Required for repository operations

# Run OpenHands
docker run -it \
    --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.53-nikolaik \
    -e SANDBOX_USER_ID=$(id -u) \
    -e SANDBOX_VOLUMES=$SANDBOX_VOLUMES \
    -e LLM_API_KEY=$LLM_API_KEY \
    -e BASE_URL=$BASE_URL \
    -e LLM_MODEL=$LLM_MODEL \
    -e SANDBOX_SELECTED_REPO=$SANDBOX_SELECTED_REPO \
    -e GITHUB_TOKEN=$GITHUB_TOKEN \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app-$(date +%Y%m%d%H%M%S) \
    docker.all-hands.dev/all-hands-ai/openhands:0.53 \
    python -m openhands.core.main -t "write a bash script that prints hi"
