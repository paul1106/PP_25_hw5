#!/bin/bash

# Judge script for hw5 testcases
# Format: testcase_name time_seconds status

TIMEOUT=180  # 3 minutes time limit
EXECUTABLE="./hw5"
TESTCASE_DIR="testcases"
OUTPUT_DIR="judge_outputs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Testcases in order
TESTCASES=(b20 b30 b40 b50 b60 b70 b80 b90 b100 b200 b512 b1024)

echo "Running judge on all testcases..."
echo "=================================="
echo ""

# Summary arrays
declare -A times
declare -A statuses

for testcase in "${TESTCASES[@]}"; do
    input_file="${TESTCASE_DIR}/${testcase}.in"
    expected_file="${TESTCASE_DIR}/${testcase}.out"
    output_file="${OUTPUT_DIR}/${testcase}.out"
    
    # Check if input file exists
    if [ ! -f "$input_file" ]; then
        echo "Warning: $input_file not found, skipping..."
        continue
    fi
    
    # Run with timeout and measure time
    start_time=$(date +%s.%N)
    
    timeout $TIMEOUT srun -t 00:05:00 --gres=gpu:1 ./run_with_gpu.sh "$EXECUTABLE" "$input_file" "$output_file" > /dev/null 2>&1
    exit_code=$?
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Check status
    if [ $exit_code -eq 124 ]; then
        status="${RED}time limit exceeded${NC}"
        status_short="time limit exceeded"
    elif [ $exit_code -ne 0 ]; then
        status="${RED}runtime error${NC}"
        status_short="runtime error"
    elif [ ! -f "$output_file" ]; then
        status="${RED}no output${NC}"
        status_short="no output"
    else
        # Validate output
        if [ -f "$expected_file" ]; then
            python3 validate.py "$output_file" "$expected_file" > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                status="${GREEN}accepted${NC}"
                status_short="accepted"
            else
                status="${RED}wrong answer${NC}"
                status_short="wrong answer"
            fi
        else
            status="${YELLOW}no expected output${NC}"
            status_short="no expected output"
        fi
    fi
    
    # Store results
    times[$testcase]=$elapsed
    statuses[$testcase]=$status_short
    
    # Print result
    printf "%5s %7.2f   %b\n" "$testcase" "$elapsed" "$status"
done

echo ""
echo "=================================="
echo "Judge completed"
