#!/bin/bash

msg=${date}
aws sqs send-message --queue-url ${QUEUE_URL} --message-body "$msg"
echo "sqs exit code="$?
