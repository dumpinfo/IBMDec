#ifndef HMMSTATEDECODING_H
#define HMMSTATEDECODING_H

using namespace std;

#include <vector>
#include <list>

#include <stdio.h>
#include <string.h>
#include <math.h>

 
 

// for gcc #include <x86intrin.h> can be used instead, it includes whatever is needed

#if defined __linux__ || defined __MINGW32__ || defined _MSC_VER
#include <malloc.h>
#elif __APPLE__
#include <sys/malloc.h>
#else 
	#error "unsupported platform"
#endif

#include "Global.h"

namespace Bavieca {

class FileInput;
class FileOutput;
class IOBase;
class PhoneSet;

// state's within-word position
#define WITHIN_WORD_POSITION_START			0
#define WITHIN_WORD_POSITION_INTERNAL		1
#define WITHIN_WORD_POSITION_END				2
#define WITHIN_WORD_POSITION_MONOPHONE		3

#define STR_WITHIN_WORD_POSITION_START			"start"
#define STR_WITHIN_WORD_POSITION_INTERNAL		"internal"
#define STR_WITHIN_WORD_POSITION_END			"end"
#define STR_WITHIN_WORD_POSITION_MONOPHONE	"monophone"

class HMMStateDecoding;

typedef vector<HMMStateDecoding*> VHMMStateDecoding;

#define DIMENSIONALITY 39

typedef struct {
	int iId;											// unique identifier (across all Gaussian component in the system)
	float fWeight;											// gaussian weight
	float fConstant;
 
	float fMean[DIMENSIONALITY];				// gaussian mean
	float fCovariance[DIMENSIONALITY];		// gaussian covariance matrix (diagonal)	
 	
} GaussianDecoding;

typedef vector<GaussianDecoding*> VGaussianDecoding;
typedef list<GaussianDecoding*> LGaussianDecoding;

/**
	@author daniel <dani.bolanos@gmail.com>
*/
class HMMStateDecoding {
	private:
		int m_iDim;
		PhoneSet *m_phoneSet;
		// state identity
		unsigned char m_iPhone;						// basephone
		unsigned char m_iState;						// HMM-state 
		unsigned char m_iPosition;					// HMM-state within-word position
		int m_iId;										// HMM-state unique identifier
		
		// gaussians
		int m_iGaussianComponents;
		GaussianDecoding *m_gaussians;			// block of memory containing all gaussians
		
		// constants
		float m_fConstant;
		
		// caching computations
		int m_iTimestamp;							// time-stamp (for caching emission probability computations)
		float m_fProbabilityCached;			// cached probability
		
		// whether the covariance was modified to accelerate the computation of emission probabilities
		bool m_bCovarianceOriginal;
		// computes the emission probability of the state given the feature vector 
		// uses nearest-neighbor approximation
		// uses SIMD instructions (SSE) (sse support must be enabled during compilation!)
		float computeEmissionProbabilityNearestNeighborSSE(float *fFeatures, int iTime);	
	public:
		// default constructor
		HMMStateDecoding();
		// set initial parameters
		void setInitialParameters(int iDim, PhoneSet *phoneSet, int iId);
		// destructor
		~HMMStateDecoding();
		// initialize the estimation by precomputing constants and invariant terms
		void initialize();
		// load the HMM from a file
		void load(FileInput &file);
		
		inline void resetTimeStamp() { m_iTimestamp = -1;}

		inline int getId() { return m_iId;}
		
		// computes the emission probability of the state given the feature vector
		inline float computeEmissionProbability(float *fFeatures, int iTime) {
			return computeEmissionProbabilityNearestNeighborSSE(fFeatures,iTime);
		}
		
		// return the Gaussian likelihood for the given feature vector
		// uses nearest-neighbor approximation
		// uses Partial Distance Elimination (PDE)
		//double computeGaussianProbability(int iGaussian, float *fFeatures);
		
		inline void setGaussianIds(int *iId) {
		
			for(int i=0 ; i < m_iGaussianComponents ; ++i) {
				m_gaussians[i].iId = (*iId)++;
			}	
		}		
};

};	// end-of-namespace

#endif
