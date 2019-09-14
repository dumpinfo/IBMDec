#ifndef HMMMANAGER_H
#define HMMMANAGER_H

#include <limits.h>

using namespace std;

#include <list>

//#include "HMMState.h"
#include "HMMStateDecoding.h"

#define HMM_CONTEXT_MODELING_MONOPHONES		1
#define ACCUMULATOR_TYPE_PHYSICAL			1
#define MAX_IDENTITY_LENGTH		32

namespace Bavieca {

class ContextDecisionTree;
class PhoneSet;
class PhoneticRulesManager;
class Transform;

// models purpose
#define HMM_PURPOSE_ESTIMATION			0
#define HMM_PURPOSE_EVALUATION			1

 
class HMMManager {

	private:
	
		PhoneSet *m_phoneSet;						// phonetic symbol set
		unsigned char m_iPurpose;					// whether HMMs will be used to estimation or just evaluation
		
		
		// HMM-states (evaluation)
		HMMStateDecoding *m_hmmStatesDecoding;	// monophones: [iBasePhone x iState x iPosition] 
												// triphones: array of physical clustered context dependent HMM-states
		int m_iDim;								// feature dimensionality
		int m_iCovarianceModeling;				// covariance modeling type (diagonal/full)
		int m_iCovarianceElements;
		int m_iHMMStates;							// number of unique HMM-states
		bool m_bInitialized;						// whether the models are already initialized
		int m_iBasePhones;							// number of basephones
		unsigned char m_iContextModelingOrderHMM;					// HMMs context modeling order (within-word)
		unsigned char m_iContextModelingOrderHMMCW;				// HMMs context modeling order (cross-word)
		unsigned char m_iContextModelingOrderAccumulators;		// Accumulator's context modeling order (within-word)
		unsigned char m_iContextModelingOrderAccumulatorsCW;	// Accumulator context modeling order (cross-word)
		unsigned char m_iIdentityAux[MAX_IDENTITY_LENGTH];
		bool m_bSingleGaussian;						// whether all the HMM-states are single gaussian
		
		// accumulators
		unsigned char m_iAccumulatorType;		// whether to use logical accumulators (logical vs physical)
												// logical acc: one accumulator per context modeling unit
												// physical acc: one accumulator per Gaussian component
		// phonetic rules
		PhoneticRulesManager *m_phoneticRulesManager;
		// context decision trees
		ContextDecisionTree **m_contextDecisionTrees;
		int m_iContextDecisionTrees;		
		// destroy
		void destroy();	
		// covariance floor
		float *m_fCovarianceFloor;
		// version of the hmmsystem that created the models
		int m_iSystemVersion;
	public:
		// constructor
		HMMManager(PhoneSet *phoneSet, unsigned char iPurpose);
		// destructor
		~HMMManager();

		
		unsigned int getHMMStateIndex(unsigned char *iPhoneLeft, unsigned char iPhone, 
		unsigned char *iPhoneRight, unsigned char iPosition, unsigned char iState);
		void load(const char *strFile);
		HMMStateDecoding * getHMMStateDecoding(unsigned char *iPhoneLeft, unsigned char iPhone, 
		unsigned char *iPhoneRight, unsigned char iPosition, unsigned char iState);
		
		// get the covariance elements (dimensionality)
		inline static int getCovarianceElements(int iDim, int iCovarianceModelling) 
		{ 
			return iDim;
		}
		
		// return the number of physical HMM-states
		inline int getNumberHMMStatesPhysical() {
			return m_iHMMStates;
		}
		
		void  resetHMMEmissionProbabilityComputation();
		// initialize HMM-states for decoding
		void  initializeDecoding() ;
		
		// return the context modeling size
		inline unsigned char getContextSizeHMM() {
			return (m_iContextModelingOrderHMM-1)/2;
		}
		
		// return the context modeling size
		inline unsigned char getContextSizeHMMCW() {
			return (m_iContextModelingOrderHMMCW-1)/2;
		}
		// return the feature dimensionality
		inline unsigned int getFeatureDim() {
			return m_iDim;
		}
		
};

};	// end-of-namespace

#endif
