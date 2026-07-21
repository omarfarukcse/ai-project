# scripts/canary-deploy.sh
#!/bin/bash
# Canary Deployment Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🦜 Starting Canary Deployment${NC}"

# Variables
NAMESPACE="cdss"
CANARY_VERSION=${1:-"v3.0.0"}
TRAFFIC_PERCENTAGE=${2:-5}
MONITOR_DURATION=${3:-300}  # 5 minutes
ROLLBACK_ON_ERROR=${4:-true}

echo -e "${BLUE}📋 Canary Configuration:${NC}"
echo "  Version: $CANARY_VERSION"
echo "  Traffic: $TRAFFIC_PERCENTAGE%"
echo "  Monitor Duration: ${MONITOR_DURATION}s"
echo "  Auto-Rollback: $ROLLBACK_ON_ERROR"

# Step 1: Build and push canary image
echo -e "${YELLOW}🏗️ Building canary image...${NC}"
docker build -t cdss-healthcare-api:canary .
docker tag cdss-healthcare-api:canary $REGISTRY/cdss-api:canary-$CANARY_VERSION
docker push $REGISTRY/cdss-api:canary-$CANARY_VERSION

# Step 2: Update canary deployment
echo -e "${YELLOW}🔄 Updating canary deployment...${NC}"
kubectl set image deployment/cdss-api-canary \
    api=$REGISTRY/cdss-api:canary-$CANARY_VERSION \
    -n $NAMESPACE

# Step 3: Wait for canary rollout
echo -e "${YELLOW}⏳ Waiting for canary rollout...${NC}"
kubectl rollout status deployment/cdss-api-canary -n $NAMESPACE --timeout=300s

# Step 4: Configure traffic splitting (Istio)
echo -e "${YELLOW}🌐 Configuring traffic splitting...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cdss-api-vs
  namespace: cdss
spec:
  hosts:
  - cdss-api-service
  http:
  - route:
    - destination:
        host: cdss-api-service
        subset: production
      weight: $((100 - $TRAFFIC_PERCENTAGE))
    - destination:
        host: cdss-api-canary-service
        subset: canary
      weight: $TRAFFIC_PERCENTAGE
EOF

# Step 5: Monitor canary
echo -e "${YELLOW}📊 Monitoring canary for ${MONITOR_DURATION}s...${NC}"
START_TIME=$(date +%s)
CANARY_FAILED=false

while [ $(($(date +%s) - $START_TIME)) -lt $MONITOR_DURATION ]; do
    # Check error rate
    ERROR_RATE=$(kubectl exec -n $NAMESPACE deployment/prometheus -- \
        curl -s "http://localhost:9090/api/v1/query?query=sum(rate(cdss_requests_total{track=canary,status=~\"5..\"}[1m]))/sum(rate(cdss_requests_total{track=canary}[1m]))" \
        | jq -r '.data.result[0].value[1] // 0')
    
    # Check latency
    LATENCY=$(kubectl exec -n $NAMESPACE deployment/prometheus -- \
        curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(cdss_request_latency_seconds_bucket{track=canary}[1m])) by (le))" \
        | jq -r '.data.result[0].value[1] // 0')
    
    echo -e "${BLUE}📈 Current Metrics:${NC}"
    echo "  Error Rate: ${ERROR_RATE}%"
    echo "  P95 Latency: ${LATENCY}s"
    
    # Check thresholds
    if (( $(echo "$ERROR_RATE > 5" | bc -l) )); then
        echo -e "${RED}❌ Error rate exceeded 5%${NC}"
        CANARY_FAILED=true
        break
    fi
    
    if (( $(echo "$LATENCY > 2.0" | bc -l) )); then
        echo -e "${RED}❌ Latency exceeded 2s${NC}"
        CANARY_FAILED=true
        break
    fi
    
    sleep 10
done

# Step 6: Handle result
if [ "$CANARY_FAILED" = true ] && [ "$ROLLBACK_ON_ERROR" = true ]; then
    echo -e "${RED}❌ Canary failed, rolling back...${NC}"
    
    # Rollback traffic
    cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cdss-api-vs
  namespace: cdss
spec:
  hosts:
  - cdss-api-service
  http:
  - route:
    - destination:
        host: cdss-api-service
        subset: production
      weight: 100
EOF
    
    # Scale down canary
    kubectl scale deployment cdss-api-canary --replicas=0 -n $NAMESPACE
    
    # Send alert
    echo -e "${RED}🚨 Canary rollback triggered!${NC}"
    
    # Send Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"🚨 *Canary Deployment Failed!*\nVersion: $CANARY_VERSION\nAction: Rolled back to production\nTime: $(date)\"
        }" \
        $SLACK_WEBHOOK_URL
    
elif [ "$CANARY_FAILED" = true ]; then
    echo -e "${RED}❌ Canary failed but auto-rollback disabled${NC}"
    echo "Manual intervention required"
else
    echo -e "${GREEN}✅ Canary deployment successful!${NC}"
    
    # Promote canary to production
    echo -e "${YELLOW}🚀 Promoting canary to production...${NC}"
    kubectl set image deployment/cdss-api \
        api=$REGISTRY/cdss-api:$CANARY_VERSION \
        -n $NAMESPACE
    
    # Wait for production rollout
    kubectl rollout status deployment/cdss-api -n $NAMESPACE --timeout=300s
    
    # Remove canary
    kubectl scale deployment cdss-api-canary --replicas=0 -n $NAMESPACE
    
    # Update VirtualService to 100% production
    cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cdss-api-vs
  namespace: cdss
spec:
  hosts:
  - cdss-api-service
  http:
  - route:
    - destination:
        host: cdss-api-service
        subset: production
      weight: 100
EOF
    
    # Send success notification
    curl -X POST -H 'Content-type: application/json' \
        --data "{
            \"text\": \"✅ *Canary Deployment Successful!*\nVersion: $CANARY_VERSION\nPromoted to: Production\nTime: $(date)\"
        }" \
        $SLACK_WEBHOOK_URL
fi

echo -e "${GREEN}🏁 Canary deployment complete!${NC}"