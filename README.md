# ai-unifier-assesment

## Run the service:

### Set your API credentials
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-api-key"

### Start the service
docker-compose up --build

### Test with curl:
curl -X POST http://localhost:8000/api/chat/stream \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello"}' \
    --no-buffer