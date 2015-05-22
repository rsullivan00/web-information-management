#!/usr/bin/env bash
# Check code guidelines
echo -e '\e[32mChecking for code style errors \e[0m'
if ! flake8 *.py unit_tests; then
    exit 1
fi

for f in ./unit_tests/*.py; do
    if [ "$(basename $f)" == "__init__.py" ]; then
        continue
    fi
    echo -e '\e[32mTesting '$(basename $f)'\e[0m'
    if ! python3 -m unit_tests.$(basename $f .py); then
        exit 1
    fi
done
