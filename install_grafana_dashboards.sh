#!/bin/bash
set -e

PLEX_HOST="10.0.0.246"
GRAFANA_URL="http://$PLEX_HOST:3000"
CREDS="${GRAFANA_CREDS:?GRAFANA_CREDS is required (format: user:pass)}"

DS_UID=$(ssh -p 2204 -o StrictHostKeyChecking=no mkusper@$PLEX_HOST "curl -s -u $CREDS $GRAFANA_URL/api/datasources/name/Prometheus | grep -o '\"uid\":\"[^\"]*' | cut -d'\"' -f4")

if [ -z "$DS_UID" ]; then
    echo "ERROR: Could not find Prometheus datasource UID in Grafana."
    exit 1
fi
echo "Found Prometheus Datasource UID: $DS_UID"

# Generate fully populated dashboards using Ray's native python factory
DASH_DIR="/tmp/ray_dashboards"
mkdir -p $DASH_DIR
rm -rf $DASH_DIR/*

ssh -i ~/.ssh/ubuntu-mac-openteams-admin -o StrictHostKeyChecking=no openteams@10.0.0.148 "/Users/openteams/miniforge3/envs/feather_env/bin/python -c \"
import os, json
import ray.dashboard.modules.metrics.grafana_dashboard_factory as fac

out_dir = '/tmp/ray_generated_dashboards'
os.makedirs(out_dir, exist_ok=True)

# Generate the JSONs via the actual factory.
dashes = [
    ('default', fac.generate_default_grafana_dashboard()),
    ('serve', fac.generate_serve_grafana_dashboard()),
    ('serve_deploy', fac.generate_serve_deployment_grafana_dashboard()),
    ('train', fac.generate_train_grafana_dashboard()),
    ('data', fac.generate_data_grafana_dashboard()),
    ('serve_llm', fac.generate_serve_llm_grafana_dashboard())
]

for name, payload in dashes:
    with open(f'{out_dir}/{name}.json', 'w') as f:
        # payload[0] contains the actual JSON string! payload[1] is just the UID string.
        f.write(payload[0])

\" && cd /tmp/ray_generated_dashboards && tar -czf /tmp/ray_dashboards.tar.gz *.json"

scp -i ~/.ssh/ubuntu-mac-openteams-admin -o StrictHostKeyChecking=no openteams@10.0.0.148:/tmp/ray_dashboards.tar.gz $DASH_DIR/
tar -xzf $DASH_DIR/ray_dashboards.tar.gz -C $DASH_DIR/

# Patch and upload
for file in $DASH_DIR/*.json; do
    filename=$(basename "$file")
    echo "Uploading $filename..."
    sed -i "s/\${DS_PROMETHEUS}/$DS_UID/g" "$file"
    jq "{dashboard: ., overwrite: true, folderId: 0}" "$file" > "$file.payload"
    cat "$file.payload" | ssh -p 2204 -o StrictHostKeyChecking=no mkusper@$PLEX_HOST "curl -s -X POST -H 'Content-Type: application/json' -u $CREDS -d @- $GRAFANA_URL/api/dashboards/db"
    echo ""
done

echo "Successfully injected REAL Ray Dashboards in Grafana!"
