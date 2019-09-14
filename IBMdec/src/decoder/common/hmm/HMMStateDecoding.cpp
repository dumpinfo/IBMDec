#include "FileInput.h"
#include "FileOutput.h"
#include "HMMStateDecoding.h"
#include "IOBase.h"
#include "PhoneSet.h"

#include <limits.h>

namespace Bavieca {

 

// default constructor
HMMStateDecoding::HMMStateDecoding()
{
	m_iTimestamp = -1;
	m_phoneSet = NULL;
	m_iId = -1;
	m_gaussians = NULL;
}

// set initial parameters
void HMMStateDecoding::setInitialParameters(int iDim, PhoneSet *phoneSet, int iId) {
	m_iDim = iDim;
	m_iTimestamp = -1;
	m_phoneSet = phoneSet;
	m_iId = iId;
	m_gaussians = NULL;
}

// destructor
HMMStateDecoding::~HMMStateDecoding()
{
	free(m_gaussians);
}

// initialize the estimation by precomputing constants and invariant terms
void HMMStateDecoding::initialize() {
	m_fConstant = powf(((float)PI_NUMBER)*2.0f,((float)DIMENSIONALITY)/2.0f);
	// compute the covariance determinantof every gaussian
	for(int iGaussian = 0 ; iGaussian < m_iGaussianComponents ; ++iGaussian) {
		double dDeterminant = 1.0;
		for(int i = 0 ; i < DIMENSIONALITY ; ++i) {
			// compute the determinant of the covariance matrix
			dDeterminant *= m_gaussians[iGaussian].fCovariance[i];
			assert(m_gaussians[iGaussian].fCovariance[i] != 0);
		}
		m_gaussians[iGaussian].fConstant = (float)log(m_gaussians[iGaussian].fWeight/(m_fConstant*sqrt(dDeterminant)));
		assert(finite(m_gaussians[iGaussian].fConstant));
		// invert covariance and divide it by two
		for(int i = 0 ; i < DIMENSIONALITY ; ++i) {
			m_gaussians[iGaussian].fCovariance[i] = (float)(1.0/(2.0*m_gaussians[iGaussian].fCovariance[i]));
		}
	}
	m_bCovarianceOriginal = false;
	//printf("%x %d %d\n",this,m_iId,m_iGaussianComponents);
}

 

// load the HMM from a file (binary format)
void HMMStateDecoding::load(FileInput &file) {
	// phonetic symbol
	char strPhone[MAX_PHONETIC_SYMBOL_LENGTH+1];	
	IOBase::readBytes(file.getStream(),reinterpret_cast<char*>(strPhone),MAX_PHONETIC_SYMBOL_LENGTH+1);
	m_iPhone = m_phoneSet->getPhoneIndex(strPhone);
	assert(m_iPhone != UCHAR_MAX);
	// state
	IOBase::read(file.getStream(),&m_iState);
	assert(m_iState < NUMBER_HMM_STATES);
	// within word position (DEPRECATED)
	IOBase::read(file.getStream(),&m_iPosition);
	// Gaussian components
	IOBase::read(file.getStream(),&m_iGaussianComponents);
	assert(m_iGaussianComponents > 0);
	// allocate memory
	m_gaussians = new GaussianDecoding[m_iGaussianComponents];
	// load
	for(int iGaussian = 0 ; iGaussian < m_iGaussianComponents ; ++iGaussian) {
	
		m_gaussians[iGaussian].iId = -1;
		IOBase::read(file.getStream(),&m_gaussians[iGaussian].fWeight);
		IOBase::readBytes(file.getStream(),reinterpret_cast<char*>(m_gaussians[iGaussian].fMean),m_iDim*sizeof(float));
		IOBase::readBytes(file.getStream(),reinterpret_cast<char*>(m_gaussians[iGaussian].fCovariance),
			m_iDim*sizeof(float));
	
		//m_gaussians[iGaussian].iBaseClass = -1;
		//m_gaussians[iGaussian].accumulator = NULL;	
	}
	
	m_bCovarianceOriginal = true;
}

float HMMStateDecoding::computeEmissionProbabilityNearestNeighborSSE(float *fFeatures, int iTime) {

	if (iTime == m_iTimestamp) {	
		return m_fProbabilityCached;
	}
	
	float *fMean,*fCovariance,fAcc;
	float fLogLikelihood = LOG_LIKELIHOOD_FLOOR;
	
	for(int iGaussian = 0 ; iGaussian < m_iGaussianComponents ; ++iGaussian) {
	
		fMean = m_gaussians[iGaussian].fMean;
		fCovariance = m_gaussians[iGaussian].fCovariance;	
		fAcc = m_gaussians[iGaussian].fConstant;
	
		/*for(int i=0 ; i<39 ; ++i) {
			fAcc -= (fFeatures[i]-fMean[i])*(fFeatures[i]-fMean[i])*fCovariance[i];
		}*/
      fAcc -= (fFeatures[0]-fMean[0])*(fFeatures[0]-fMean[0])*fCovariance[0];
      fAcc -= (fFeatures[1]-fMean[1])*(fFeatures[1]-fMean[1])*fCovariance[1];
      fAcc -= (fFeatures[2]-fMean[2])*(fFeatures[2]-fMean[2])*fCovariance[2];
      fAcc -= (fFeatures[3]-fMean[3])*(fFeatures[3]-fMean[3])*fCovariance[3];
      fAcc -= (fFeatures[4]-fMean[4])*(fFeatures[4]-fMean[4])*fCovariance[4];
      fAcc -= (fFeatures[5]-fMean[5])*(fFeatures[5]-fMean[5])*fCovariance[5];
      fAcc -= (fFeatures[6]-fMean[6])*(fFeatures[6]-fMean[6])*fCovariance[6];
      fAcc -= (fFeatures[7]-fMean[7])*(fFeatures[7]-fMean[7])*fCovariance[7];
      fAcc -= (fFeatures[8]-fMean[8])*(fFeatures[8]-fMean[8])*fCovariance[8];
      fAcc -= (fFeatures[9]-fMean[9])*(fFeatures[9]-fMean[9])*fCovariance[9];
      fAcc -= (fFeatures[10]-fMean[10])*(fFeatures[10]-fMean[10])*fCovariance[10];
      fAcc -= (fFeatures[11]-fMean[11])*(fFeatures[11]-fMean[11])*fCovariance[11];
      fAcc -= (fFeatures[12]-fMean[12])*(fFeatures[12]-fMean[12])*fCovariance[12];
      
      fAcc -= (fFeatures[13]-fMean[13])*(fFeatures[13]-fMean[13])*fCovariance[13];
      fAcc -= (fFeatures[14]-fMean[14])*(fFeatures[14]-fMean[14])*fCovariance[14];
      fAcc -= (fFeatures[15]-fMean[15])*(fFeatures[15]-fMean[15])*fCovariance[15];
      fAcc -= (fFeatures[16]-fMean[16])*(fFeatures[16]-fMean[16])*fCovariance[16];
      fAcc -= (fFeatures[17]-fMean[17])*(fFeatures[17]-fMean[17])*fCovariance[17];
      fAcc -= (fFeatures[18]-fMean[18])*(fFeatures[18]-fMean[18])*fCovariance[18];
      fAcc -= (fFeatures[19]-fMean[19])*(fFeatures[19]-fMean[19])*fCovariance[19];
      fAcc -= (fFeatures[20]-fMean[20])*(fFeatures[20]-fMean[20])*fCovariance[20];
      fAcc -= (fFeatures[21]-fMean[21])*(fFeatures[21]-fMean[21])*fCovariance[21];
      fAcc -= (fFeatures[22]-fMean[22])*(fFeatures[22]-fMean[22])*fCovariance[22];
      fAcc -= (fFeatures[23]-fMean[23])*(fFeatures[23]-fMean[23])*fCovariance[23];
      fAcc -= (fFeatures[24]-fMean[24])*(fFeatures[24]-fMean[24])*fCovariance[24];
      fAcc -= (fFeatures[25]-fMean[25])*(fFeatures[25]-fMean[25])*fCovariance[25];
      
      fAcc -= (fFeatures[26]-fMean[26])*(fFeatures[26]-fMean[26])*fCovariance[26];
      fAcc -= (fFeatures[27]-fMean[27])*(fFeatures[27]-fMean[27])*fCovariance[27];
      fAcc -= (fFeatures[28]-fMean[28])*(fFeatures[28]-fMean[28])*fCovariance[28];
      fAcc -= (fFeatures[29]-fMean[29])*(fFeatures[29]-fMean[29])*fCovariance[29];
      fAcc -= (fFeatures[30]-fMean[30])*(fFeatures[30]-fMean[30])*fCovariance[30];
      fAcc -= (fFeatures[31]-fMean[31])*(fFeatures[31]-fMean[31])*fCovariance[31];
      fAcc -= (fFeatures[32]-fMean[32])*(fFeatures[32]-fMean[32])*fCovariance[32];
      fAcc -= (fFeatures[33]-fMean[33])*(fFeatures[33]-fMean[33])*fCovariance[33];
      fAcc -= (fFeatures[34]-fMean[34])*(fFeatures[34]-fMean[34])*fCovariance[34];
      fAcc -= (fFeatures[35]-fMean[35])*(fFeatures[35]-fMean[35])*fCovariance[35];
      fAcc -= (fFeatures[36]-fMean[36])*(fFeatures[36]-fMean[36])*fCovariance[36];
      fAcc -= (fFeatures[37]-fMean[37])*(fFeatures[37]-fMean[37])*fCovariance[37];
      fAcc -= (fFeatures[38]-fMean[38])*(fFeatures[38]-fMean[38])*fCovariance[38];
 		
		fLogLikelihood = max(fAcc,fLogLikelihood);
	}
	
	assert(finite(fLogLikelihood) != 0);

	// cache the probability
	m_iTimestamp = iTime;
	m_fProbabilityCached = fLogLikelihood;
	
	return fLogLikelihood;
}
 
  
 
 


};	// end-of-namespace
