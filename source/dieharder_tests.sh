#!/bin/bash

SEED = 2366022249
DELTA = 0
ALGO = "lcg"

TEST_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 100 101 102 202 204 205 206 207 208 209"

for ID in $TEST_IDS; do
  echo "---"
  echo "### RUNNING DIEHARDER TEST $ID ###"
  echo "---"
  python3 testing_interface.py p $SEED $DELTA --algo $ALGO | dieharder -g 200 -d "$ID" > ${SEED}test"$ID".txt
done

echo "---"
echo "### All selected tests completed! ###"
echo "---"
