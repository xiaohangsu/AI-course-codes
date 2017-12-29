#!/usr/bin/env bash
>w.json

for i in {1..50}
do
    echo $i " time"
    python pacman.py -p CompetitionAgent
done
