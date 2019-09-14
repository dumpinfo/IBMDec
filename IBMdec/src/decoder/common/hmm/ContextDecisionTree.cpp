#include "ContextDecisionTree.h"
#include "VectorBase.h"
#include <limits.h>

namespace Bavieca {

// return the HMM-state associated to the given context-dependent phone
int ContextDecisionTree::getHMMIndex(unsigned char *iPhoneLeft, unsigned char iPhone, 
	unsigned char *iPhoneRight, unsigned char iPosition, 
	unsigned char iState) {
		
	DTNode *node = m_dtnodeRoot;
	while(node->rule != NULL) {
		bool bAnswer = true;	
		switch(node->rule->iType) {	
			case RULE_TYPE_PHONETIC_PHONE_LEFT: {
				assert(iPhoneLeft[node->rule->iContextPosition] != UCHAR_MAX);
				bAnswer = node->rule->phoneticRule->bPhone[iPhoneLeft[node->rule->iContextPosition]];
				break;
			}
			case RULE_TYPE_PHONETIC_PHONE: {
				bAnswer = node->rule->phoneticRule->bPhone[iPhone];
				break;
			}
			case RULE_TYPE_PHONETIC_PHONE_RIGHT: {
				assert(iPhoneRight[node->rule->iContextPosition] != UCHAR_MAX);
				bAnswer = node->rule->phoneticRule->bPhone[iPhoneRight[node->rule->iContextPosition]];
				break;
			}
			case RULE_TYPE_POSITION: {	
				bAnswer = (node->rule->iPosition == iPosition);
				break;
			}
			case RULE_TYPE_STATE: {
				bAnswer = (node->rule->iState == iState);
				break;
			}
			default: {
				assert(0);	
				break;
			}
		}
		if (bAnswer) {
			node = node->dtnodeYes;
		} else {
			node = node->dtnodeNo;
		}	
	}
	
	// make sure we got to a leaf
	assert(node->iHMMState != -1);
	
	return node->iHMMState;	
} 

// constructor
ContextDecisionTree::ContextDecisionTree(int iFeatureDimensionality, int iCovarianceModelling, PhoneSet *phoneSet, unsigned char iContextModelingOrder) {

	m_iDim = iFeatureDimensionality;
	m_iCovarianceModelling = iCovarianceModelling;	
	m_phoneSet = phoneSet;
	m_iContextModelingOrder = iContextModelingOrder;
	m_dtnodeRoot = NULL;
	m_dtnodes = NULL;
	m_iNodes = 0;	
	m_iLeaves = 0;
	m_iDim = iFeatureDimensionality;	
	
	m_bEstimationDataAllocated = false;
}

 

// destructor
ContextDecisionTree::~ContextDecisionTree() {

	// tree loaded from disk
	if (m_dtnodes != NULL) {
		for(int i=0 ; i < m_iNodes ; ++i) {
			delete m_dtnodes[i].rule;
		}
		delete [] m_dtnodes;
	} 
	 
}

// load from file
ContextDecisionTree *ContextDecisionTree::load(FileInput &file, int iFeatureDimensionality, 
	int iCovarianceModelling, PhoneSet *phoneSet, PhoneticRulesManager *phoneticRulesManager, 
	unsigned char iContextModelingOrder) {

	ContextDecisionTree *contextDecisionTree = new ContextDecisionTree(iFeatureDimensionality,iCovarianceModelling,phoneSet,iContextModelingOrder);

	// read the context modeling order
	unsigned char iOrder;
 	IOBase::read(file.getStream(),&iOrder);	
	assert(iContextModelingOrder == iOrder);

	// read the number of nodes
 	IOBase::read(file.getStream(),&contextDecisionTree->m_iNodes);	
	assert(contextDecisionTree->m_iNodes >= 1);
	
	contextDecisionTree->m_iLeaves = 0;
	
	// allocate memory for the nodes
	contextDecisionTree->m_dtnodes = new DTNode[contextDecisionTree->m_iNodes];
	
	// read the nodes one by one
	int iIndexChildYes = -1;
	int iIndexChildNo = -1;	
	int iHMMState = -1;
	for(int i=0 ; i < contextDecisionTree->m_iNodes ; ++i) {
		
		// load the children indices
	 	IOBase::read(file.getStream(),&iIndexChildYes);	
 		IOBase::read(file.getStream(),&iIndexChildNo);	
		// leaf node
		if (iIndexChildYes == -1) {
			assert(iIndexChildNo == -1);
			
			// load the HMM-state index
	 		IOBase::read(file.getStream(),&iHMMState);	
	 		assert(iHMMState >= 0);
			
			contextDecisionTree->m_dtnodes[i].dtnodeYes = NULL;	
			contextDecisionTree->m_dtnodes[i].dtnodeNo = NULL;
			contextDecisionTree->m_dtnodes[i].iHMMState = iHMMState;
			contextDecisionTree->m_dtnodes[i].rule = NULL;
			
			contextDecisionTree->m_iLeaves++;
		} 
		// internal node
		else {

			assert(iIndexChildNo != -1);
			assert((iIndexChildYes >= 0) && (iIndexChildYes < contextDecisionTree->m_iNodes));
			assert((iIndexChildNo >= 0) && (iIndexChildNo < contextDecisionTree->m_iNodes));
			
			contextDecisionTree->m_dtnodes[i].dtnodeYes = &contextDecisionTree->m_dtnodes[iIndexChildYes];	
			contextDecisionTree->m_dtnodes[i].dtnodeNo = &contextDecisionTree->m_dtnodes[iIndexChildNo];
			contextDecisionTree->m_dtnodes[i].iHMMState = -1; 
			contextDecisionTree->m_dtnodes[i].rule = new Rule;
			
			// load the rule	
			
			// rule type
			IOBase::read(file.getStream(),&contextDecisionTree->m_dtnodes[i].rule->iType);
			// rule identity
			switch(contextDecisionTree->m_dtnodes[i].rule->iType) {
				// left context
				case RULE_TYPE_PHONETIC_PHONE_LEFT:
				// central phone
				case RULE_TYPE_PHONETIC_PHONE: 
				// right context
				case RULE_TYPE_PHONETIC_PHONE_RIGHT: {
					if (contextDecisionTree->m_dtnodes[i].rule->iType != RULE_TYPE_PHONETIC_PHONE) {
						// read the context position
						IOBase::read(file.getStream(),&contextDecisionTree->m_dtnodes[i].rule->iContextPosition);
					}
					// read the phonetic rule index
					int iRule = -1;
					IOBase::read(file.getStream(),&iRule);
					assert((iRule >= 0) && (iRule < (int)phoneticRulesManager->getRules()->size()));
					// get the rule
					contextDecisionTree->m_dtnodes[i].rule->phoneticRule = phoneticRulesManager->getRule(iRule);
					if (contextDecisionTree->m_dtnodes[i].rule->phoneticRule == NULL) {
						return NULL;
					}
					break;
				}
				// within-word position
				case RULE_TYPE_POSITION: {
					contextDecisionTree->m_dtnodes[i].rule->phoneticRule = NULL;
					// read the position
					IOBase::read(file.getStream(),&contextDecisionTree->m_dtnodes[i].rule->iPosition);
					//assert(HMMState::isPositionValid(contextDecisionTree->m_dtnodes[i].rule->iPosition));
					break;
				}
				// HMM-state
				case RULE_TYPE_STATE: {
					contextDecisionTree->m_dtnodes[i].rule->phoneticRule = NULL;
					// read the state
					IOBase::read(file.getStream(),&contextDecisionTree->m_dtnodes[i].rule->iState);
					//assert(HMMState::isStateValid(contextDecisionTree->m_dtnodes[i].rule->iState));
					break;
				}
				default: {
					assert(0);
				}
			}			
		}
	}	
	// root node?
	contextDecisionTree->m_dtnodeRoot = &contextDecisionTree->m_dtnodes[0];

	return contextDecisionTree;
}




//================================================================================
//================================================================================
 
// store to file
void ContextDecisionTree::store(FileOutput &file) { 
}

// cluster a node (if possible)
void ContextDecisionTree::clusterNode(DTNode *node, float fMinimumClusterOccupation, float fMinimumLikelihoodGainClustering) { 
}

// compute the likelihood of a cluster of context dependent units 
void ContextDecisionTree::computeLikelihoodCluster(DTNode *node, double *dLikelihoodCluster, double *dOccupationCluster, Vector<float> &vCovarianceGlobal) { 
}

// compute the cluster likelihood
double ContextDecisionTree::computeLikelihoodCluster(double dOccupation, Vector<double> &vCovariance) { 

	return 0.0;
}

// compute the cluster likelihood
double ContextDecisionTree::computeLikelihoodCluster(double dOccupation, Vector<float> &vCovariance) { 

	return 0.0;
}


// compute the likelihood of a cluster of nphones given the parent cluster, the rule and the decission made
// note: return false if after applying the rule:
// - there is not enough data to robustly estimate the parameters (minimum occupation count)
// - there is too much data resulting from the answer to the question, so the opposite answer wont produce enough data
bool ContextDecisionTree::computeLikelihoodRule(DTNode *node, Rule *rule, bool bAnswer, double *dLikelihoodCluster, double *dOccupationCluster, float fMinimumClusterOccupation) { 
	
	return true;
}


// compute the likelihood of the clusters resulting from applying a question to the given cluster
// note: return false if after applying the rule:
// - there is not enough data to robustly estimate the parameters (minimum occupation count)
// - there is too much data resulting from the answer to the question, so the opposite answer wont produce enough data
bool ContextDecisionTree::computeLikelihoodRule(DTNode *node, Rule *rule, double *dLikelihoodClusterYes, double *dLikelihoodClusterNo, double *dOccupationClusterYes, double *dOccupationClusterNo, float fMinimumClusterOccupation) {
	
	return true;
}


// compute the likelihood of a decission tree by summing up the likelihood in the leaves
double ContextDecisionTree::computeTreeLikelihood(DTNode *node) { 
  return 0.0;
}

// return the number of leaves in the tree (including those leaves merged to another leaf)
int ContextDecisionTree::countTreeLeaves(DTNode *node) { 
   return 0.0; 
}

// return the number of leaves in the tree (excluding those leaves merged to another leaf)
int ContextDecisionTree::countTreeLeavesUnique(DTNode *node) {
    return 0.0;
}


// get the tree leaves
void ContextDecisionTree::getTreeLeaves(DTNode *node, LDTNode &lLeaves) { }

// print tree leaves
void ContextDecisionTree::printTreeLeaves(DTNode *node) { }

 

// compute the likelihood of a decission tree by summing up the likelihood in the leaves
int ContextDecisionTree::countTreeLeavesOccupancy(DTNode *node) { 
return 0.0;
}

// destroy a decission tree
void ContextDecisionTree::destroyTree(DTNode *node) { 
}

// compute the likelihood of the cluster resulting from mergining two leaves
double ContextDecisionTree::computeLikelihoodFromMerging(DTNode *nodeA, DTNode *nodeB, Vector<float> &vCovarianceGlobal) { return 0.0;
}

// compact the tree leaves by merging those leaves that when merged the likelihood decrease is below the minimum splitting gain used for clustering (return the resulting number of leaves)
int ContextDecisionTree::compactTreeLeaves(float fMinimumLikelihoodGainClustering, Vector<float> &vCovarianceGlobal) { 
  return 0.0;
}

 
 

};	// end-of-namespace

