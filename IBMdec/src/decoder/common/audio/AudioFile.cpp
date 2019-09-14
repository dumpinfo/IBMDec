#include <stdexcept>

#include "AudioFile.h"
#include "FileInput.h"
#include "FileOutput.h"
#include "IOBase.h"
#include "LogMessage.h"

namespace Bavieca {

// constructor
AudioFile::AudioFile() {
}

// destructor
AudioFile::~AudioFile() {
}

// load the data
short *AudioFile::load(const char *strFile, int *iSamples) {
	
	short *sSamples = NULL;
	try {
	
		FileInput file(strFile,true);
		file.open();
		
		*iSamples = file.size()/2;
		assert(*iSamples > 0);
		sSamples = new short[*iSamples];
		IOBase::readBytes(file.getStream(),reinterpret_cast<char*>(sSamples),*iSamples*sizeof(short));
	
		file.close();

	} catch (std::runtime_error) {
		BVC_ERROR<< "unable to load the audio file: " << strFile;
	}

	return sSamples;
}

// store the data
void AudioFile::store(const char *strFile, short *sSamples, int iSamples) {

	assert((iSamples >= 0) && (sSamples));

	FileOutput file(strFile,true);
	IOBase::writeBytes(file.getStream(),reinterpret_cast<char*>(sSamples),iSamples*sizeof(short));
	file.close();
}

};	// end-of-namespace

