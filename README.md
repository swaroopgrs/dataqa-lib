uv add pre-commit
uv run pre-commit
uv run pre-commit autoupdate
uv tool install ruff@latest
uv run gitingest -i "dataqa/ tets/" 
uv sync
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
git rm -r --cached **/__pycache__

uv add duckdb gitingest faker pandas pydantic pytest pytest-asyncio langgraph langchain-openai pre-commit


find . -name "__pycache__" -type d -exec rm -rf {} +




docker pull docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik
docker run -it --rm --pull=always \
    -e SANDBOX_VOLUMES=/home/swaro/projects/dataqa-oh:/workspace:rw \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.38-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands-state:/.openhands-state \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.38