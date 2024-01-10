#!/bin/bash

function kill_python() {
    kill $PYTHON_PID
}

trap kill_python EXIT

python3 task3/server.py &
PYTHON_PID=$!
cd mmsr-ui && npm install && npm start
