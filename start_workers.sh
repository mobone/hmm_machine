#!/usr/bin/env bash
for run in {1..24}
do
    rq worker --url redis://192.168.1.128 &
done