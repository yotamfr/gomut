#!/usr/bin/env bash

REMOTE_HOST=$1
LOCAL_PORT=$2
REMOTE_PORT=$3

ssh -o 'GatewayPorts yes' -L ${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT} yotamfr@trachel-srv.cs.haifa.ac.il -Nv


# sudo bash openport.sh trachel-srv2.cs.haifa.ac.il 8200 8888
# C%K9#bGx~ua
