#!/bin/bash

# Get the directory where the script is located (should be .../testing)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Move up one directory to reach the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT" || exit 1

# Run the unit tests
./testing/unit_tests/run_unit_tests.sh

# Run functional tests
./testing/functional_tests/run_functional_tests.sh

