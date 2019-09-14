
#include "ContextDecisionTree.h"
#include "HMMManager.h"
#include "PhoneSet.h"
#include "PhoneticRulesManager.h"

#include <iomanip>
#include <stdexcept>
#include <limits.h>

namespace Bavieca {

// return a pointer to a HMM-state given its identity
HMMStateDecoding *HMMManager::getHMMStateDecoding(unsigned char *iPhoneLeft, unsigned char iPhone, 
	unsigned char *iPhoneRight, unsigned char iPosition, unsigned char iState) {
	int iIndex = getHMMStateIndex(iPhoneLeft,iPhone,iPhoneRight,iPosition,iState);
	return &m_hmmStatesDecoding[iIndex];	
}

// return a pointer to a HMM-state given its identity
unsigned int HMMManager::getHMMStateIndex(unsigned char *iPhoneLeft, unsigned char iPhone, 
	unsigned char *iPhoneRight, unsigned char iPosition, unsigned char iState) {

	int iIndex = m_contextDecisionTrees[iPhone*NUMBER_HMM_STATES+iState]->getHMMIndex(iPhoneLeft,
				iPhone,iPhoneRight,iPosition,iState);
	 
	return iIndex;
	 
}

// reset the emission probability computation
void HMMManager::resetHMMEmissionProbabilityComputation() {
       {
		for(int i=0 ; i<m_iHMMStates ; ++i) {
			m_hmmStatesDecoding[i].resetTimeStamp();
		}	
	}
}


// initialize HMM-states for decoding
void HMMManager::initializeDecoding() {
	
	m_iAccumulatorType = ACCUMULATOR_TYPE_PHYSICAL;
	for(int i=0 ; i<m_iHMMStates ; ++i) {
		m_hmmStatesDecoding[i].initialize();
	}
}

// constructor
HMMManager::HMMManager(PhoneSet *phoneSet, unsigned char iPurpose)
{
	assert((iPurpose == HMM_PURPOSE_ESTIMATION) || (iPurpose == HMM_PURPOSE_EVALUATION));

	m_phoneSet = phoneSet;
	m_iPurpose = iPurpose;
	m_bInitialized = false;
	
	// HMMs (estimation)
	//m_hmmStates = NULL;
	// HMMs (evaluation)
	m_hmmStatesDecoding = NULL;	
	
	// get the number of basephones
	m_iBasePhones = m_phoneSet->size();	
	
	// phonetic rules
	m_phoneticRulesManager = NULL;
	
	// context decision trees
	m_contextDecisionTrees = NULL;
	m_iContextDecisionTrees = -1;	
	
	// covariance
	m_iCovarianceModeling = -1;
	m_iCovarianceElements = -1;
	m_fCovarianceFloor = NULL;
	
	// feature dimensionality
	m_iDim = -1;
}


// destroy
void HMMManager::destroy() {

	// delete the memory used to store the HMM-states
	// (estimation)
	if (m_iPurpose == HMM_PURPOSE_ESTIMATION) {
	} 
	// (evaluation)
	else if (m_iPurpose == HMM_PURPOSE_EVALUATION) {	
		if (m_hmmStatesDecoding != NULL) {
			delete [] m_hmmStatesDecoding;
			m_hmmStatesDecoding = NULL;
		}	
	} else {
		assert(0);
	}
	
	if (m_phoneticRulesManager != NULL) {
		delete m_phoneticRulesManager;
	}
	if (m_contextDecisionTrees != NULL) {
		for(int i=0 ; i < m_iContextDecisionTrees ; ++i) {
			delete m_contextDecisionTrees[i];
		}
		delete [] m_contextDecisionTrees;	
	}
}

// destructor
HMMManager::~HMMManager(){ destroy();}

// load models from file
void HMMManager::load(const char *strFile) {

	try {

		FileInput file(strFile,true);
		file.open();
		
		// load the version of the system that created the models
		char strVersion[SYSTEM_VERSION_FIELD_WIDTH+1];
		IOBase::readBytes(file.getStream(),strVersion,SYSTEM_VERSION_FIELD_WIDTH);
		m_iSystemVersion = atoi(strVersion);
		
		// load the dimensionality and the covariance modeling type
		IOBase::read(file.getStream(),&m_iDim);
		IOBase::read(file.getStream(),&m_iCovarianceModeling);
		m_iCovarianceElements = getCovarianceElements(m_iDim,m_iCovarianceModeling);
		
		// load the number of phonemes
		int iPhones = -1;
		IOBase::read(file.getStream(),&iPhones);
		assert(iPhones == (int)m_phoneSet->size());
		
		// load the context modeling type
		IOBase::read(file.getStream(),&m_iContextModelingOrderHMM);
		IOBase::read(file.getStream(),&m_iContextModelingOrderHMMCW);
		
		// load whether the models are single or multiple gaussian
		m_bSingleGaussian = false;
		unsigned char iGaussianMixtureSize = UCHAR_MAX;
		IOBase::read(file.getStream(),&iGaussianMixtureSize);
		//if (iGaussianMixtureSize == HMM_MIXTURE_SIZE_SINGLE) {
		//	m_bSingleGaussian = true;
		//} else {
		//	assert(iGaussianMixtureSize == HMM_MIXTURE_SIZE_MULTIPLE);
			m_bSingleGaussian = false;
		//}
		
		// load the number of different HMM-states (monophones or triphones)
		IOBase::read(file.getStream(),&m_iHMMStates);
		
		// HMM-estimation
		if (m_iPurpose == HMM_PURPOSE_ESTIMATION) { 
		
		} 
		// HMM-evaluation
		else {
			
			assert(m_iPurpose == HMM_PURPOSE_EVALUATION);
			
			// allocate memory for the states
			m_hmmStatesDecoding = new HMMStateDecoding[m_iHMMStates];
			
			// set initial parameters and load the states
			int iGaussianId = 0;
			for(int i=0 ; i < m_iHMMStates; ++i) {
				m_hmmStatesDecoding[i].setInitialParameters(m_iDim,m_phoneSet,i);
				m_hmmStatesDecoding[i].load(file);
				m_hmmStatesDecoding[i].setGaussianIds(&iGaussianId);
			}
		}
		
		// context dependent models (it is necessary to load the decision trees)
		if (m_iContextModelingOrderHMM > HMM_CONTEXT_MODELING_MONOPHONES) {
		
			// local accumulators have to be used
			m_iAccumulatorType = ACCUMULATOR_TYPE_PHYSICAL;
			
			// load the phonetic rules 
			m_phoneticRulesManager = PhoneticRulesManager::load(file,m_phoneSet);
			
			// load the # of context decision tree(s)
			IOBase::read(file.getStream(),&m_iContextDecisionTrees);
			// load the context decision tree(s)
			if (m_iContextDecisionTrees == 1) {
				m_contextDecisionTrees = new ContextDecisionTree*[1];
				m_contextDecisionTrees[0] = ContextDecisionTree::load(file,m_iDim,m_iCovarianceModeling,
					m_phoneSet,m_phoneticRulesManager,m_iContextModelingOrderHMM);
				assert(m_contextDecisionTrees[0]);
			} else {
				assert(m_iContextDecisionTrees == (int)m_phoneSet->size()*NUMBER_HMM_STATES);
				m_contextDecisionTrees = new ContextDecisionTree*[m_iContextDecisionTrees];
				for(int i=0 ; i < m_iContextDecisionTrees ; ++i) {
					m_contextDecisionTrees[i] = ContextDecisionTree::load(file,m_iDim,m_iCovarianceModeling,
						m_phoneSet,m_phoneticRulesManager,m_iContextModelingOrderHMM);
				}
			}
		}
		
		// mark the HMMs as initialized
		m_bInitialized = true;
		
	} catch(std::runtime_error) {
		BVC_ERROR << "unable to load the acoustic models from file: " << strFile;
	}
}

 
 

};	// end-of-namespace

