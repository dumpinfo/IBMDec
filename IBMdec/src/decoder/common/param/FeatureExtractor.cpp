#include <stdexcept>

#include "AudioFile.h"
#include "BatchFile.h"
#include "FeatureExtractor.h"
#include "Matrix.h"
#include "Vector.h"
//#include "MatrixStatic.h"
#include "Numeric.h"
//#include "Waveform.h"

#include "ConfigurationFeatures.h"

namespace Bavieca {

// note: the bank of filters used comprises N filters that are equally spaced on the mel scale
// all the filters have the same height and each filter bounday coincides with the center
// frequency of the adjacent filter

// constructor
FeatureExtractor::FeatureExtractor(ConfigurationFeatures *configurationFeatures, float fWarpFactor, int iCepstralBufferSize, int iCepstralNormalizationMode, int iCepstralNormalizationMethod) {

	m_iSamplingRate = configurationFeatures->getIntParameterValue("waveform.samplingRate");
	m_iSampleSize = configurationFeatures->getIntParameterValue("waveform.sampleSize");
	m_bDCRemoval = configurationFeatures->getBoolParameterValue("dcRemoval");
	m_bPreemphasis = configurationFeatures->getBoolParameterValue("preemphasis");
	m_iWindowWidth = configurationFeatures->getIntParameterValue("window.width");
	m_iWindowShift = configurationFeatures->getIntParameterValue("window.shift");
	m_iWindowTapering = getTaperingWindow(configurationFeatures->getStrParameterValue("window.tapering"));
	m_iType = getFeatureType(configurationFeatures->getStrParameterValue("features.type"));
	
	m_iFilterbankFrequencyMin = configurationFeatures->getIntParameterValue("filterbank.frequency.min");
	m_iFilterbankFrequencyMax = configurationFeatures->getIntParameterValue("filterbank.frequency.max");
	m_iFilterbankFilters = configurationFeatures->getIntParameterValue("filterbank.filters");
	m_iCepstralCoefficients = configurationFeatures->getIntParameterValue("cepstralCoefficients");
	m_bEnergy = configurationFeatures->getBoolParameterValue("energy");
	m_fWarpFactor = fWarpFactor; 
	m_iDerivativesOrder = configurationFeatures->getIntParameterValue("derivatives.order");
	m_iDerivativesDelta = configurationFeatures->getIntParameterValue("derivatives.delta");
	
	//m_iSplicedSize = configurationFeatures->getIntParameterValue("spliced.size");
	m_iSplicedSize = -1;
	
	m_iCepstralBufferSize = iCepstralBufferSize;
	m_iCepstralNormalizationMode = iCepstralNormalizationMode;
	m_iCepstralNormalizationMethod = iCepstralNormalizationMethod;	
	m_bCepstralBufferFull = false;
	
	if (m_bEnergy) {
		m_iCoefficients = m_iCepstralCoefficients+1;
	} else {
		m_iCoefficients = m_iCepstralCoefficients;
	}
	m_iCoefficientsTotal = m_iCoefficients;
	if (m_iDerivativesOrder > 0) {
		m_iCoefficientsTotal = m_iCoefficients*(m_iDerivativesOrder+1);
	} else if (m_iSplicedSize > 0) {
		m_iCoefficientsTotal = m_iCoefficients*m_iSplicedSize;
	}
		
	// filterbanks
	m_dCenterFrequencyMel = NULL;
		
	// fft
	m_iFFTPoints = 0;
	m_iFFTPointBin = NULL;
	
	// cepstral buffer	
	m_iCepstralBufferPointer = 0;
	m_mCepstralBuffer = NULL;
}

// destructor
FeatureExtractor::~FeatureExtractor()
{
	if (m_dCenterFrequencyMel) {
		delete [] m_dCenterFrequencyMel;
	}
	if (m_iFFTPointBin) {
		delete [] m_iFFTPointBin;
	}
	if (m_dFFTPointGain) {
		delete [] m_dFFTPointGain;
	}
	if (m_mCepstralBuffer) {
		delete m_mCepstralBuffer;
	}
	if (m_sSamplesUsefulPrev) {
		delete [] m_sSamplesUsefulPrev;
	}
}

// initialization
void FeatureExtractor::initialize() {

	// check parameters
	if (m_iCepstralNormalizationMode != CEPSTRAL_NORMALIZATION_MODE_NONE) {
		assert((m_iCepstralNormalizationMethod == CEPSTRAL_NORMALIZATION_METHOD_CMN) ||
			(m_iCepstralNormalizationMethod == CEPSTRAL_NORMALIZATION_METHOD_CMVN));		
	}
	
	if (m_iSampleSize != 16) {
		BVC_ERROR << "currently sample sizes different than 16 bits are not supported";
	}
	
	assert(m_iSamplingRate % 1000 == 0);
	m_iSamplesFrame = (m_iSamplingRate/1000)*m_iWindowWidth;
	m_iSamplesSkip = (m_iSamplingRate/1000)*m_iWindowShift;

	// compute the number of points in the FFT (smaller power of two above the number of samples in the window)
	m_iFFTPoints = 2;
	while(m_iFFTPoints < m_iSamplesFrame) {
		m_iFFTPoints *= 2; 
	}

	// build the filterbank
	buildFilterBank();
	
	// cepstral buffer
	if (m_iCepstralBufferSize != -1) {
		m_mCepstralBuffer = new Matrix<float>(m_iCepstralBufferSize,m_iCepstralCoefficients);
	} else {
		m_mCepstralBuffer = NULL;
	}
	
	// stream mode feature extraction (might not be used)
	m_sSamplesUsefulPrev = new short[m_iSamplesFrame];
	m_iSamplesUsefulPrev = 0;
	m_iSamplesStream = 0;		
}

// build the filter bank (it takes into account the warp factor)
void FeatureExtractor::buildFilterBank() {

	// (1) compute the center frequency of each filter in the filterbank
	
	// allocate memory to store the center frequencies
	m_dCenterFrequencyMel = new double[m_iFilterbankFilters+2];
	
	// min and max allowed frequencies (for band-limited feature extraction)
	double dFrequencyMinMel = linearToMel((double)m_iFilterbankFrequencyMin);
	double dFrequencyMaxMel = linearToMel((double)m_iFilterbankFrequencyMax);	
		
	// filterbank centers are equally spaced in the mel scale
	double dStepMel = (dFrequencyMaxMel-dFrequencyMinMel)/((double)(m_iFilterbankFilters+1));	
	
	// note: an extra center frequency is computed for an imaginary n+1 filter, this center 
	// is necessary for the computations during binning
	for(int i=1 ; i <= m_iFilterbankFilters+1 ; ++i) {
		m_dCenterFrequencyMel[i] = dFrequencyMinMel+(i*dStepMel);
	}
	
	// warped features
	if (m_fWarpFactor != 1.0) { 
	}
	
	// (2) precompute constants for the application of the filterbank	
	computeFFTPointBin();
}

 



// extract features in  batch mode
void FeatureExtractor::extractFeaturesSession(VUtteranceData &vUtteranceData, bool bHaltOnFailure) {

	// session-based normalization
	if (m_iCepstralNormalizationMode == CEPSTRAL_NORMALIZATION_MODE_SESSION) {	
		
		// compute the static coefficients
		int iUtterance = 0;
		for(VUtteranceData::iterator it = vUtteranceData.begin() ; it != vUtteranceData.end() ; ++it, ++iUtterance) {
			
			// extract static features
			it->mFeatures = extractStaticFeatures(it->samples.sSamples,it->samples.iSamples);
			if (it->mFeatures == NULL) {
				if (bHaltOnFailure) {
					BVC_ERROR << "unable to extract static features from utterance: " << iUtterance;
				} else {
					BVC_WARNING << "unable to extract static features from utterance: " << iUtterance;
				}
			}
		}
		
		// allocate memory for the static coefficients
		VUtteranceFeatures vUtteranceFeatures;
		iUtterance = 0;
		for(VUtteranceData::iterator it = vUtteranceData.begin() ; it != vUtteranceData.end() ; ++it, ++iUtterance) {
			vUtteranceFeatures.push_back(it->mFeatures);
		}	
	
		// CMN
		if (m_iCepstralNormalizationMethod == CEPSTRAL_NORMALIZATION_METHOD_CMN) {
			applyCMN(vUtteranceFeatures,m_iCepstralCoefficients);
		}
		// CMVN
		//else { 
		//	assert(m_iCepstralNormalizationMethod == CEPSTRAL_NORMALIZATION_METHOD_CMVN);
		//	applyCMVN(vUtteranceFeatures,m_iCepstralCoefficients);
		//}
		
		// extract derivatives and create the final features
		for(VUtteranceData::iterator it = vUtteranceData.begin() ; it != vUtteranceData.end() ; ++it) {
		
			if (it->mFeatures == NULL) {
				continue;
			}
			
			if (it->mFeatures->finite() == false) {
				BVC_ERROR << "invalid MFCC features";
			}
			Matrix<float> *mFeatures = computeDerivatives(*it->mFeatures);
			if (mFeatures == NULL) {
				if (bHaltOnFailure) {
					BVC_ERROR << "unable to compute derivatives";
				} else {
					BVC_WARNING << "unable to compute derivatives";		
				}
			}
			if (mFeatures->finite() == false) {
				BVC_ERROR << "invalid MFCC features";
			}
			
			delete it->mFeatures;
			it->mFeatures = mFeatures;
		}
	}
}


  
// extract static features
Matrix<float> *FeatureExtractor::extractStaticFeatures(short *sSamples, unsigned int iSamples) {

	// mfcc
	if (m_iType == FEATURE_TYPE_MFCC) {
		return extractStaticFeaturesMFCC(sSamples,iSamples);
	} 
	// plp
	else if (m_iType == FEATURE_TYPE_PLP) {
		return extractStaticFeaturesPLP(sSamples,iSamples);
	} 
	// not supported
	else {
		return NULL;
	}
}

// extract MFCC features (only the static coefficients)
Matrix<float> *FeatureExtractor::extractStaticFeaturesMFCC(short *sSamplesOriginal, unsigned int iSamples) {

	// (1) make a copy of the samples so they do not get modified
	short *sSamples = new short[iSamples];
	memcpy(sSamples,sSamplesOriginal,iSamples*sizeof(short));

	// (2) remove the DC mean
	if (m_bDCRemoval) {
		removeDC(sSamples,iSamples);
	}
	
	// (3) pre-emphasis
	if (m_bPreemphasis) {
		applyPreemphasis(sSamples,iSamples);
	}
	
	// check that the number of frames is positive, otherwise there are not enough samples available
	if (iSamples < m_iSamplesFrame) {
		BVC_WARNING << "insufficient number of audio samples (available: " << iSamples 
			<< " required: " << m_iSamplesFrame << ")";
		return NULL;
	}
	// compute the number of frames (feature vectors) to extract
	unsigned int iFrames = 1+(iSamples-m_iSamplesFrame)/m_iSamplesSkip;
	// allocate memory for the frames
	Matrix<float> *mMFCC = new Matrix<float>(iFrames,m_iCoefficients);	

	// allocate memory for the Fast Fourier Transform output (there has to be room for the real and the imaginary part)
	double *dFFT = new double[2*m_iFFTPoints+2];
	double *dFFTMagnitude = new double[m_iFFTPoints];
	
	// allocate memory for the bins
	double *dBins = new double[m_iFilterbankFilters+1];
	
	// auxiliar buffer
	double *dSamplesFrame = new double[m_iSamplesFrame];
	
	// process each window
	for(unsigned int i = 0 ; i < iFrames ; ++i) {
	
		for(unsigned int j=0 ; j < m_iSamplesFrame ; ++j) {
			dSamplesFrame[j] = (double)sSamples[(i*m_iSamplesSkip)+j];
		}
	
		// do window tapering if needed
		applyWindowTapering(dSamplesFrame,m_iSamplesFrame);
		
		// compute the frame energy
		double dFrameEnergy = computeLogEnergy(dSamplesFrame,m_iSamplesFrame);
		
		// apply the Fourier transform and get the frequencies
		for(unsigned int j = 0 ; j < m_iSamplesFrame ; ++j) {
			dFFT[j] = dSamplesFrame[j]; 
		}
		for(unsigned int j = m_iSamplesFrame ; j < m_iFFTPoints+2 ; ++j) {
			dFFT[j] = 0.0; 
		}	
		Numeric::fft(dFFT,m_iFFTPoints);
		
		// compute the magnitude from the real and imaginary parts
		int iIndex = 2;
		for(unsigned int j = 2 ; j < m_iFFTPoints/2 ; ++j, iIndex+=2) {
			double dReal = dFFT[iIndex];
			double dImaginary = dFFT[iIndex+1];
			dFFTMagnitude[j] = sqrt(dReal*dReal+dImaginary*dImaginary);
		}
		
		// apply the bank of filters (one bin per filter)
		for(int j=0 ; j < m_iFilterbankFilters+1 ; ++j) {
			dBins[j] = 0.0;
		}
		// multiply the filter gain by the magnitude
		// (every point in the FFT affects two bins (filters), since filters overlap)
		for(unsigned int j=2 ; j < m_iFFTPoints/2 ; ++j) {
			int iBin = m_iFFTPointBin[j];
			if (iBin == -1) {	// ignore frequencies outside the filterbank
				continue;
			}
			double dValue = m_dFFTPointGain[j]*dFFTMagnitude[j];
			if (iBin > 0) {
				dBins[iBin] += dValue;
			}
			if (iBin < m_iFilterbankFilters) {
				dBins[iBin+1] += dFFTMagnitude[j]-dValue;
			}	
		}
		
		// apply log compression
		for(int h=0 ; h < m_iFilterbankFilters+1 ; ++h) {
			if (dBins[h] <= 1.0) {
				dBins[h] = 0.0;
			} else {
				dBins[h] = log(dBins[h]);
			}
		}
		
		// Discrete Cosine Transform (DCT): compute cepstral features from the log filterbank amplitudes
		// note: we are ignoring c0
		float *fMFCCAux = mMFCC->getRow(i).getData();
		for(int j=1 ; j <= m_iCepstralCoefficients ; ++j) {
			double dAux = 0.0;
			for(int k=1 ; k < m_iFilterbankFilters+1 ; ++k) {	
				dAux += dBins[k]*cos((PI_NUMBER*j*(((double)k)-0.5))/((double)m_iFilterbankFilters));
			}
			dAux *= sqrt(2.0/((double)m_iFilterbankFilters));
			fMFCCAux[j-1] = (float)dAux;
		}
		fMFCCAux[m_iCepstralCoefficients] = (float)dFrameEnergy;
	}
	
	 
	
	// compute normalized log energy
	float fMinFrameEnergy = 10000.0;
	float fMaxFrameEnergy = 0.0;
	for(unsigned int i=0 ; i < mMFCC->getRows() ; ++i) {
		fMaxFrameEnergy = std::max((*mMFCC)(i,m_iCepstralCoefficients),fMaxFrameEnergy);
		fMinFrameEnergy = std::min((*mMFCC)(i,m_iCepstralCoefficients),fMinFrameEnergy);
	}
	
   for(unsigned int i=0 ; i < mMFCC->getRows() ; i++) {
      (*mMFCC)(i,m_iCepstralCoefficients) += 1.0f-fMaxFrameEnergy;
      (*mMFCC)(i,m_iCepstralCoefficients) = std::max((*mMFCC)(i,m_iCepstralCoefficients),-5.0f);
   }	
	
	// clean-up
	delete [] dBins;
	delete [] dFFT;
	delete [] dFFTMagnitude;
	delete [] dSamplesFrame;
	delete [] sSamples;
	
	if (mMFCC->finite() == false) {		
		BVC_ERROR << "invalid MFCC features";
	}

	return mMFCC;
}

// extract PLP features (only the static coefficients)
Matrix<float> *FeatureExtractor::extractStaticFeaturesPLP(short *sSamples, unsigned int iSamples) {

	return NULL;
}


// remove the DC-mean
void FeatureExtractor::removeDC(short *sSamples, int iSamples) {

	double *dSamples = new double[iSamples];

	// compute the mean
	double dAcc = 0.0;
	for(int i=0 ; i < iSamples ; ++i) {
		dAcc += (double)sSamples[i];
	}
	dAcc /= ((double)iSamples);
	
	// substract the mean
	for(int i=0 ; i < iSamples ; ++i) {
		dSamples[i] = ((double)sSamples[i])-dAcc;
	}
	
	// convert back to short
	convert(dSamples,sSamples,iSamples);
	
	delete [] dSamples;
}

// apply pre-emphasis to all the samples (note that some samples at the end may be discarded)
void FeatureExtractor::applyPreemphasis(short *sSamples, int iSamples) {

	for(int i=iSamples-1 ; i > 0 ; --i) {
		sSamples[i] = convert(((float)sSamples[i])-((float)PREEMPHASIS_COEFFICIENT*((float)sSamples[i-1])));
	}
	sSamples[0] = convert(((float)sSamples[0])-((float)PREEMPHASIS_COEFFICIENT*((float)sSamples[0])));
}

// apply Window-tapering
void FeatureExtractor::applyWindowTapering(double *dSamples, int iSamples) {

	// Hamming window
	if (m_iWindowTapering == WINDOW_TAPERING_METHOD_HAMMING) {
		applyHammingWindow(dSamples,iSamples);
	}
	 
	 
	// no window tapering
	else {
		assert(m_iWindowTapering == WINDOW_TAPERING_METHOD_NONE);
	}
}

// apply the Hamming window
void FeatureExtractor::applyHammingWindow(double *dSamples, int iSamples) {

	double dConst = (2.0*PI_NUMBER)/((double)(iSamples-1));

	for(int i = 0 ; i < iSamples ; ++i) {
		dSamples[i] = dSamples[i]*(0.54-0.46*cos(dConst*i));
	}
}

 
// computes the log energy of a speech frame
double FeatureExtractor::computeLogEnergy(double *dFrame, int iSamples){

   double dEnergy = 0.0; 
   for(int i = 0 ; i < iSamples ; i++) {
		dEnergy += dFrame[i]*dFrame[i];
   }

   dEnergy /= ((double)iSamples);
   if (dEnergy > 1.0) {
      dEnergy = log10(dEnergy);
   } else {
      dEnergy = 0.0;
   }

   return dEnergy;
}

// compute the lower bin connected to each FFT point
// note: all filters are equally spaced in the mel scale, a filter starts at the center 
//  freq of prev filter and ends at the center freq of the next filter so the right, then
//  right side and left side of contiguous filters completely overlap and are, thus, 
//  equivalent (same inclination)
void FeatureExtractor::computeFFTPointBin() {

	// generate a mapping: point in the FFT <-> filter in the filterbank (bin)
	m_iFFTPointBin = new int[m_iFFTPoints];
	double dStep = ((double)m_iSamplingRate)/((double)m_iFFTPoints);
	int iBin = 1;
	for(unsigned int i=1 ; i <= m_iFFTPoints/2 ; ++i) {
		double dLinearFreq = (i-1)*dStep;
		if ((dLinearFreq < m_iFilterbankFrequencyMin) || (dLinearFreq > m_iFilterbankFrequencyMax)) {
			m_iFFTPointBin[i] = -1;	
		} else {
			while((m_dCenterFrequencyMel[iBin] < linearToMel((i-1)*dStep)) && (iBin < (m_iFilterbankFilters+1)))  {
				++iBin;
			}
			m_iFFTPointBin[i] = iBin-1;
		}
	}
	
	// generate a mapping: filter <-> filter gain (right side of the triangular filter)
	m_dFFTPointGain = new double[m_iFFTPoints];
	for(unsigned int i=1 ; i <= m_iFFTPoints/2 ; ++i) {
		int iBin = m_iFFTPointBin[i];
		if (iBin == -1) {
			m_dFFTPointGain[i] = 0.0;
		} else if (iBin > 0) {
			m_dFFTPointGain[i] = (m_dCenterFrequencyMel[iBin+1]-linearToMel((i-1)*dStep))/(m_dCenterFrequencyMel[iBin+1]-m_dCenterFrequencyMel[iBin]);
		} else {
			m_dFFTPointGain[i] = (m_dCenterFrequencyMel[1]-linearToMel((i-1)*dStep))/m_dCenterFrequencyMel[1];
		}
	}	
}

// compute derivatives (each row is a feature vector)
Matrix<float> *FeatureExtractor::computeDerivatives(MatrixBase<float> &mStatic) {

	int iOrder = m_iDerivativesOrder;
	int iDelta = m_iDerivativesDelta;

	assert(iOrder >= 1);
	unsigned int iFeaturesMinimum = ((iDelta*2)+1);
	if (mStatic.getRows() < iFeaturesMinimum) {
		BVC_WARNING << "insufficient number of feature vectors to compute derivatives (available: " 
			<< mStatic.getRows() << " required: " << iFeaturesMinimum << ")";	
		return NULL;
	}

	// allocate memory for the features
	Matrix<float> *m = new Matrix<float>(mStatic.getRows(),mStatic.getCols()*(m_iDerivativesOrder+1));
	// copy static features
	for(unsigned int j=0 ; j < m->getRows() ; ++j) {
		m->getRow(j).copy(mStatic.getRow(j),0,mStatic.getCols());
	}	
	// generate derivatives
	Matrix<float> mAux(mStatic);
	Matrix<float> mDelta(mStatic.getRows(),mStatic.getCols());
	for(int i=1 ; i <= iOrder ; ++i) {
		mAux.delta(mDelta,iDelta);
		for(unsigned int j=0 ; j < m->getRows() ; ++j) {
			m->getRow(j).copy(mDelta.getRow(j),mDelta.getCols()*i,mDelta.getCols());
		}
		mDelta.swap(mAux);
	}
	
	return m;
}



// compute CMN over a set of utterances (session mode)
void FeatureExtractor::applyCMN(VUtteranceFeatures &vUtteranceFeatures, unsigned int iCoeffNorm) {

	assert(!vUtteranceFeatures.empty());	
	Vector<double> vObservation(m_iCoefficients);
	Vector<double> vMean(m_iCoefficients);	
 
 	// compute the mean
	vObservation.zero();
	int iFeatures = 0;
	for(VUtteranceFeatures::iterator it = vUtteranceFeatures.begin() ; it != vUtteranceFeatures.end() ; ++it) {
		if (*it) {
			vObservation.addRows(*(*it));
			iFeatures += (*it)->getRows();
		}
	}	
	vMean.mul(1.0/((double)iFeatures),vObservation);
	
	// not all coeff will be normalized
	for(int i=iCoeffNorm ; i < m_iCoefficients ; ++i) {
		vMean(i) = 0.0;
	}
	
	// substract the mean
	for(VUtteranceFeatures::iterator it = vUtteranceFeatures.begin() ; it != vUtteranceFeatures.end() ; ++it) {
		if (*it) {
			for(unsigned int i=0 ; i < (*it)->getRows() ; ++i) {	
				(*it)->getRow(i).add(-1.0,vMean);
			}
		}
	}
}

// compute CMVN over a set of utterances (session mode)
/*void FeatureExtractor::applyCMVN(VUtteranceFeatures &vUtteranceFeatures, unsigned int iCoeffNorm) {

	assert(!vUtteranceFeatures.empty());	
	Vector<double> vObservation(m_iCoefficients);
	Vector<double> vObservationSquare(m_iCoefficients);
	Vector<double> vMean(m_iCoefficients);	
	Vector<double> vStandardDeviation(m_iCoefficients);	
 
 	// compute the mean
	vObservation.zero();
	int iFeatures = 0;
	for(VUtteranceFeatures::iterator it = vUtteranceFeatures.begin() ; it != vUtteranceFeatures.end() ; ++it) {
		if (*it) {
			vObservation.addRows(*(*it));
			vObservationSquare.addRowsSquare(*(*it));
			iFeatures += (*it)->getRows();
		}
	}	

	// compute the mean and standard deviation
	vMean.mul(1.0/((double)iFeatures),vObservation);	
	vStandardDeviation.mul(1.0/((double)iFeatures),vObservationSquare);
	vStandardDeviation.addSquare(-1.0,vMean);	
	vStandardDeviation.sqrt();	
	
	// not all coeff will be normalized
	assert((iCoeffNorm >= 0) && (iCoeffNorm <= m_iCoefficients));
	for(unsigned int i=iCoeffNorm ; i < (unsigned int)m_iCoefficients ; ++i) {
		vMean(i) = 0.0;
		vStandardDeviation(i) = 1.0;
	}

	// substract the mean and divide by the standard deviation
	Vector<float> vSD(vStandardDeviation);
	for(VUtteranceFeatures::iterator it = vUtteranceFeatures.begin() ; it != vUtteranceFeatures.end() ; ++it) {
		if (*it) {
			for(unsigned int i=0 ; i < (*it)->getRows() ; ++i) {	
				(*it)->getRow(i).add(-1.0,vMean);
				(*it)->getRow(i).divide(vSD);
			}
		}
	}
}

// splice features (concatenates static coefficients)
Matrix<float> *FeatureExtractor::spliceFeatures(MatrixBase<float> &mFeatures, unsigned int iElements) {

	// it must be an odd number
	assert(iElements/2 == 1);
	int iContextSize = iElements%2;
	
	// create the stacked vectors
	Matrix<float> *m = new Matrix<float>(mFeatures.getRows(),mFeatures.getCols()*iElements);
	for(unsigned int i=0 ; i < mFeatures.getRows() ; ++i) {
		for(int j=i-iContextSize ; j <= (int)(i+iContextSize) ; ++j) {
			m->getRow(i).copy(mFeatures.getRow(std::min(std::max(0,j),(int)mFeatures.getCols())),	(j-i+iContextSize)*mFeatures.getCols(),mFeatures.getCols());
		}
	}

	return m;
}*/

}; // end-of-namespace



