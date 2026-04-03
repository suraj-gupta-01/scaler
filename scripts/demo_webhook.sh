#!/bin/bash
# Demo webhook script - Simulates Datadog/Kafka alerts
#
# Usage:
#   bash scripts/demo_webhook.sh
#
# This script sends sample real alerts to the RL server to demonstrate
# the alert ingestion pipeline. The server prioritizes these real alerts
# over synthetic alerts.

BASE="http://localhost:8000"

echo "========================================"
echo "Alert Triage - Demo Webhook"
echo "========================================"
echo ""

# Check server health
echo "1. Checking server health..."
curl -s $BASE/health | python3 -m json.tool
echo ""

# Send real alerts
echo "2. Sending real alerts to triage server..."
echo ""

echo "   -> CPU spike alert (high severity)"
curl -s -X POST $BASE/ingest/alerts \
  -H "Content-Type: application/json" \
  -d '{"id":"real-cpu-001","visible_severity":0.9,"confidence":0.95,"type":"cpu_spike"}'
echo ""

echo "   -> Memory leak alert (medium severity)"
curl -s -X POST $BASE/ingest/alerts \
  -H "Content-Type: application/json" \
  -d '{"id":"real-mem-002","visible_severity":0.7,"confidence":0.85,"type":"memory_leak"}'
echo ""

echo "   -> Disk full alert (low severity)"
curl -s -X POST $BASE/ingest/alerts \
  -H "Content-Type: application/json" \
  -d '{"id":"real-disk-003","visible_severity":0.5,"confidence":0.75,"type":"disk_full"}'
echo ""

echo "   -> Network latency alert (critical)"
curl -s -X POST $BASE/ingest/alerts \
  -H "Content-Type: application/json" \
  -d '{"id":"real-net-004","visible_severity":0.95,"confidence":0.98,"type":"network_latency"}'
echo ""

echo "   -> Security alert (critical)"
curl -s -X POST $BASE/ingest/alerts \
  -H "Content-Type: application/json" \
  -d '{"id":"real-sec-005","visible_severity":0.99,"confidence":0.92,"type":"security_breach"}'
echo ""

# Show metrics
echo "3. Server metrics:"
curl -s $BASE/metrics | python3 -m json.tool
echo ""

# Show tasks
echo "4. Available tasks:"
curl -s $BASE/tasks | python3 -m json.tool
echo ""

echo "========================================"
echo "Demo complete!"
echo ""
echo "Next steps:"
echo "  - Run RL trainer: python train_external.py"
echo "  - View API docs:  http://localhost:8000/docs"
echo "========================================"
