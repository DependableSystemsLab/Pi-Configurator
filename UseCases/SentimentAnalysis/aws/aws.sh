#!/bin/bash

python3 a.textPreProcess.py 1 0 0 &
wait
echo "done-1"
python3 a.textPreProcess.py 0 1 0 &
wait
echo "done-2"
python3 a.textPreProcess.py 0 0 1 &
wait
echo "done-3"
python3 a.textPreProcess.py 1 1 0 &
wait
echo "done-4"
python3 a.textPreProcess.py 0 1 1 &
wait
echo "done-4"
python3 a.textPreProcess.py 1 0 1 &
wait
echo "done-5"
python3 a.textPreProcess.py 1 1 1 &
wait
echo "done-6"

