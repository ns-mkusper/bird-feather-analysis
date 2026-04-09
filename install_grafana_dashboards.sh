#!/bin/bash
set -e

PLEX_HOST="10.0.0.246"
GRAFANA_URL="http://$PLEX_HOST:3000"
CREDS="${GRAFANA_CREDS:?GRAFANA_CREDS is required (format: user:pass)}"

# 1. Fetch Prometheus Datasource UID from Grafana
DS_UID=$(ssh -p 2204 -o StrictHostKeyChecking=no mkusper@$PLEX_HOST "curl -s -u $CREDS $GRAFANA_URL/api/datasources/name/Prometheus | grep -o '\"uid\":\"[^\"]*' | cut -d'\"' -f4")

if [ -z "$DS_UID" ]; then
    echo "ERROR: Could not find Prometheus datasource UID in Grafana."
    exit 1
fi

echo "Found Prometheus Datasource UID: $DS_UID"

# 2. Download the official Ray Grafana dashboard templates
DASH_DIR="/tmp/ray_dashboards"
mkdir -p $DASH_DIR
rm -rf $DASH_DIR/*

# Archive the templates on the head node
ssh -i ~/.ssh/ubuntu-mac-openteams-admin -o StrictHostKeyChecking=no openteams@10.0.0.148 "cd /Users/openteams/miniforge3/envs/feather_env/lib/python3.10/site-packages/ray/dashboard/modules/metrics/dashboards/ && tar -czf /tmp/ray_dashboards.tar.gz *_grafana_dashboard_base.json"
scp -i ~/.ssh/ubuntu-mac-openteams-admin -o StrictHostKeyChecking=no openteams@10.0.0.148:/tmp/ray_dashboards.tar.gz $DASH_DIR/
tar -xzf $DASH_DIR/ray_dashboards.tar.gz -C $DASH_DIR/

# 3. Patch the templates with your specific Datasource UID and upload them
for file in $DASH_DIR/*.json; do
    filename=$(basename "$file")
    echo "Uploading $filename..."
    
    # 1. Grafana uses a complex format for exporting/importing dashboards.
    # The Ray python templates use a templated variable "${DS_PROMETHEUS}"
    sed -i "s/\${DS_PROMETHEUS}/$DS_UID/g" "$file"
    
    # 2. Create the exact JSON payload Grafana API expects
    jq "{dashboard: ., overwrite: true, folderId: 0}" "$file" > "$file.payload"
    
    # 3. Push to Grafana securely over the network via SSH
    cat "$file.payload" | ssh -p 2204 -o StrictHostKeyChecking=no mkusper@$PLEX_HOST "curl -s -X POST -H 'Content-Type: application/json' -u $CREDS -d @- $GRAFANA_URL/api/dashboards/db"
    echo ""
done

echo "Successfully provisioned Ray Dashboards in Grafana!"
