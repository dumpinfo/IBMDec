./DynamicDecoder  -cfg asrmodel/configuration.txt -hyp ./outhyp.txt  -bat ./wvlist1.txt
gprof -b ./DynamicDecoder | gprof2dot | dot -Tpdf -o fib-gprof.pdf
# gprof -b ./DynamicDecoder |gprof2dot | dot -Tpng -o output.png
