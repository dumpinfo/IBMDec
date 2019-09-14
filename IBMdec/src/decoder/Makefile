CXX = g++ -g -pg  -fprofile-arcs -ftest-coverage
CFLAGS = -fPIC
MYFLAGS = -DTIME_INFO -Wno-deprecated
INC = -I./common/algebra  -I./common/audio -I./common/base -I./common/decoding \
	-I./common/dynamicdecoder  -I./common/config -I./common/hmm -I./common/io \
	-I./common/other -I./common/param   ${INCS_DIR_CBLAS} ${INCS_DIR_LAPACK}

CXXFLAGS = $(INC)
LDFLAGS =


SOURCES = \
	./common/param/ConfigurationFeatures.cpp \
	./common/param/Numeric.cpp \
	./common/param/FeatureExtractor.cpp \
	./common/audio/AudioFile.cpp \
	./common/config/CommandLineManager.cpp \
	./common/config/ParameterManager.cpp \
	./common/config/ConfigurationFile.cpp \
	./common/io/FileInput.cpp \
	./common/io/FileOutput.cpp \
	./common/io/IOBase.cpp \
	./common/hmm/HMMManager.cpp \
	./common/hmm/HMMStateDecoding.cpp \
	./common/hmm/PhoneticRulesManager.cpp \
	./common/hmm/ContextDecisionTree.cpp \
	./common/algebra/VectorStatic.cpp \
	./common/algebra/MatrixBase.cpp \
	./common/algebra/Vector.cpp \
	./common/base/ExceptionBase.cpp \
	./common/base/PhoneSet.cpp \
	./common/base/LexUnitsFile.cpp \
	./common/base/LexiconManager.cpp \
	./common/base/FileUtils.cpp \
	./common/base/BatchFile.cpp \
	./common/base/LogMessage.cpp \
	./common/base/TimeUtils.cpp \
	./common/base/LMManager.cpp \
	./common/base/LMARPA.cpp \
	./common/base/LMFSM.cpp \
	./common/decoding/FillerManager.cpp \
	./common/decoding/BestPath.cpp \
	./common/decoding/NBestList.cpp \
	./common/dynamicdecoder/ConfigurationDynamicDecoder.cpp \
	./common/dynamicdecoder/NetworkBuilderX.cpp \
	./common/dynamicdecoder/DynamicDecoderX.cpp \
	./common/dynamicdecoder/DynamicNetworkX.cpp \
	./mainDynamicDecoder.cpp


OBJ = \
	./common/param/ConfigurationFeatures.o \
	./common/param/Numeric.o \
	./common/param/FeatureExtractor.o \
	./common/audio/AudioFile.o \
	./common/config/CommandLineManager.o \
	./common/config/ParameterManager.o \
	./common/config/ConfigurationFile.o \
	./common/io/FileInput.o \
	./common/io/FileOutput.o \
	./common/io/IOBase.o \
	./common/hmm/HMMManager.o \
	./common/hmm/HMMStateDecoding.o \
	./common/hmm/PhoneticRulesManager.o \
	./common/hmm/ContextDecisionTree.o \
	./common/algebra/VectorStatic.o \
	./common/algebra/MatrixBase.o \
	./common/algebra/Vector.o \
	./common/base/ExceptionBase.o \
	./common/base/PhoneSet.o \
	./common/base/LexUnitsFile.o \
	./common/base/LexiconManager.o \
	./common/base/FileUtils.o \
	./common/base/BatchFile.o \
	./common/base/LogMessage.o \
	./common/base/TimeUtils.o \
	./common/base/LMManager.o \
	./common/base/LMARPA.o \
	./common/base/LMFSM.o \
	./common/decoding/FillerManager.o \
	./common/decoding/BestPath.o \
	./common/dynamicdecoder/ConfigurationDynamicDecoder.o \
	./common/dynamicdecoder/NetworkBuilderX.o \
	./common/dynamicdecoder/DynamicDecoderX.o \
	./common/dynamicdecoder/DynamicNetworkX.o \
	./mainDynamicDecoder.o

Decoder: $(OBJ) mainDynamicDecoder.o
	$(CXX) $(CXXFLAGS) -o DynamicDecoder $(OBJ) $(LDFLAGS)  -ldl -lpthread

clean:
	-rm DynamicDecoder $(OBJ)
