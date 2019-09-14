lcov --capture --directory ./src/  --output-file test.info --test-name test
genhtml test.info --output-directory output --title "dyna analysis" --show-details --legend
