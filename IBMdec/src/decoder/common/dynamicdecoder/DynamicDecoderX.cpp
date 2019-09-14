#include "DynamicDecoderX.h"

#include "BestPath.h"
#include "LMFSM.h"
#include "HMMManager.h"
#include "LMManager.h"
//#include "LMLookAhead.h"
#include "PhoneSet.h"
#include "TimeUtils.h"
#include <iostream>     // std::cout, std::hex, std::endl
#include <iomanip>      // std::setiosflags

namespace Bavieca {

// constructor
DynamicDecoderX::DynamicDecoderX(PhoneSet *phoneSet, HMMManager *hmmManager, 
			LexiconManager *lexiconManager, LMManager *lmManager, float fLMScalingFactor, 
			DynamicNetworkX *dynamicNetwork, int iMaxActiveNodes, 
			int iMaxActiveNodesWE, int iMaxActiveTokensNode, float fBeamWidthNodes, 
			float fBeamWidthNodesWE, float fBeamWidthTokensNode, bool bWordGraphGeneration, 
			int iMaxWordSequencesState)
{
	m_phoneSet = phoneSet;
	m_hmmManager = hmmManager;
	m_lexiconManager = lexiconManager;
	m_lmManager = lmManager;
	m_lmFSM = lmManager->getFSM();
	m_fLMScalingFactor = fLMScalingFactor;
	m_iNGram = m_lmFSM->getNGramOrder();
	m_dynamicNetwork = dynamicNetwork;
	// network properties
	m_arcs = m_dynamicNetwork->getArcs(&m_iArcs);
	m_nodes = m_dynamicNetwork->getNodes(&m_iNodes);
	m_iTimeCurrent = -1;
	// pruning parameters
	m_iMaxActiveNodes = iMaxActiveNodes;
	m_iMaxActiveNodesWE = iMaxActiveNodesWE;
	m_iMaxActiveTokensNode = iMaxActiveTokensNode;	
	m_fBeamWidthNodes = fBeamWidthNodes;	
	m_fBeamWidthNodesWE = fBeamWidthNodesWE;	
	m_fBeamWidthTokensNode = fBeamWidthTokensNode;
	// active nodes
	m_nodesActiveCurrent = NULL;
	m_nodesActiveNext = NULL;
	// history items
	m_iHistoryItems = 0;
	m_historyItems = NULL;
	m_iHistoryItemBegSentence = -1;
	m_iHistoryItemAvailable = -1;
	m_iTimeGarbageCollectionLast = -1;
	// word-graph generation
	m_iMaxWordSequencesState = iMaxWordSequencesState;
	m_iWSHashBuckets = UINT_MAX;
	m_iWSHashEntries = UINT_MAX;	
	m_wshashEntries = NULL;
	m_iWSHashEntryCollisionAvailable = -1;
	// wrod-graph tokens
	m_iLexUnitPronUnknown = m_lexiconManager->m_lexUnitUnknown->iLexUnitPron;
	// mark it as uninitialized
	m_bInitialized = false;
}



// initialization
void DynamicDecoderX::initialize() {
	// max tokens per active arc
	m_iTokensNodeMax = m_iMaxActiveTokensNode*2;
	// allocate memory for the tables of active arcs
	m_iNodesActiveCurrentMax = m_iMaxActiveNodes*5;
	m_iNodesActiveNextMax = m_iMaxActiveNodes*5;	
	m_nodesActiveCurrent = new DNode*[m_iNodesActiveCurrentMax];
	m_nodesActiveNext = new DNode*[m_iNodesActiveNextMax];
	
	// allocate memory for the tokens
	m_iTokensMax = m_iNodesActiveCurrentMax*10;
	m_tokensCurrent = new Token[m_iTokensMax];
	m_tokensNext = new Token[m_iTokensMax];
	m_iTokensNext = 0;
	
	// pre-allocate memory for the active tokens 
	m_iActiveTokenMax = m_iNodesActiveCurrentMax*m_iTokensNodeMax;
	m_activeTokenCurrent = new ActiveToken[m_iActiveTokenMax];
	m_activeTokenNext = new ActiveToken[m_iActiveTokenMax];
	m_iActiveTokenTables = 0;
	// history item management (there exists garbage collection for history items)
	m_iHistoryItems = 10000;
	m_historyItems = new HistoryItem[m_iHistoryItems];
	m_iHistoryItemsAuxBuffer = new int[m_iTokensNodeMax];
	m_iHistoryItemsAux = NULL;
	m_iHistoryItemsAuxSize = -1;
	// language model look-ahead
	//m_lmLookAhead = new LMLookAhead(m_lexiconManager,m_lmManager,m_dynamicNetwork,this,m_iTokensMax);
	//m_lmLookAhead->initialize();
	m_bInitialized = true;	
}


// begin utterance
void DynamicDecoderX::beginUtterance() {
	assert(m_bInitialized);
	// mark all the nodes as inactive
	for(int i=0 ; i < m_iNodes; ++i) {
		m_nodes[i].iActiveTokensCurrent = 0;
		m_nodes[i].iActiveTokensCurrentBase = -1;
		m_nodes[i].iActiveTokensNext = 0;
		m_nodes[i].iActiveTokensNextBase = -1;
	}
	
	// reset history items
	for(unsigned int i=0 ; i < m_iHistoryItems-1 ; ++i) {
		m_historyItems[i].iActive = -1;
		m_historyItems[i].iPrev = i+1;
	}
	m_historyItems[m_iHistoryItems-1].iActive = -1;
	m_historyItems[m_iHistoryItems-1].iPrev = -1;
	m_iHistoryItemAvailable = 0;
	m_iTimeGarbageCollectionLast = -1;	
	m_iHistoryItemBegSentence = -1;
	
	// active tokens
	m_iActiveTokenTables = 0;
	m_iTokensNext = 0;
	m_iNodesActiveCurrent = 0;
	m_iNodesActiveNext = 0;
	m_fScoreBest = -FLT_MAX;
	m_fScoreBestWE = -FLT_MAX;
	// utterance information
	m_iFeatureVectorsUtterance = 0;
	m_hmmManager->resetHMMEmissionProbabilityComputation();
	
}
		
// process input feature vectors
void DynamicDecoderX::process(MatrixBase<float> &mFeatures) {
	assert(m_bInitialized);
	assert(mFeatures.getRows() > 0);
	// Viterbi search
	double dTimeBegin = TimeUtils::getTimeMilliseconds();
	unsigned int t = 0;
	if (m_iFeatureVectorsUtterance == 0) {
		m_iTimeCurrent = 0;
		// root node expansion	
		VectorStatic<float> vFeatureVector = mFeatures.getRow(0);
		expandRoot(vFeatureVector);
		// output search status
		BVC_VERB << "t= " << setw(5) << m_iTimeCurrent << " nodes= " << setw(6) << m_iNodesActiveCurrent << 
			" bestScore= " << FLT(12,4) << m_fScoreBest << " we: " << FLT(12,4) << m_fScoreBestWE;
		++m_iTimeCurrent;
		++t;
	}
	
	// regular expansion
	for( ; t < mFeatures.getRows() ; ++t, ++m_iTimeCurrent) {	
	
		// prune active nodes/tokens
		pruning();
		
		// next frame expansion
		VectorStatic<float> vFeatureVector = mFeatures.getRow(t);
		expand(vFeatureVector,m_iTimeCurrent);	
		
		// output search status
		if (m_iTimeCurrent % 100 == 0) {
			BVC_VERB << "t= " << setw(5) << m_iTimeCurrent << " nodes= " << setw(6) << m_iNodesActiveCurrent << 
				" bestScore= " << FLT(12,4) << m_fScoreBest << " we: " << FLT(12,4) << m_fScoreBestWE;
			//printHashsStats();
			//printHashContents();
		}
	}
	
	m_iFeatureVectorsUtterance += mFeatures.getRows();	
	
	double dTimeEnd = TimeUtils::getTimeMilliseconds();
	double dTimeSeconds = (dTimeEnd-dTimeBegin)/1000.0;
	
	BVC_VERB << "decoding time: " << FLT(8,2) << dTimeSeconds << " seconds (RTF: " <<  
		FLT(5,2) << dTimeSeconds/(((float)mFeatures.getRows())/100.0) << ")";	
}

// root-node expansion
void DynamicDecoderX::expandRoot(VectorBase<float> &vFeatureVector) {
	DNode *nodeRoot = m_dynamicNetwork->getRootNode();
	float fScore;
	int iLMStateInitial = m_lmFSM->getInitialState();
	// create the <s> history item
	m_iHistoryItemBegSentence = newHistoryItem();
	HistoryItem *historyItemBegSentence = m_historyItems+m_iHistoryItemBegSentence;
	historyItemBegSentence->iLexUnitPron = m_lexiconManager->m_lexUnitBegSentence->iLexUnitPron;
	historyItemBegSentence->iEndFrame = INT_MIN;
	historyItemBegSentence->fScore = 0.0;
	historyItemBegSentence->iPrev = -1;
	historyItemBegSentence->iActive = -1;
	historyItemBegSentence->iWGToken = -1;	
	
	// expand the root node
	DArc *arcEnd = m_arcs+(nodeRoot+1)->iArcNext;
	for(DArc *arc = m_arcs+nodeRoot->iArcNext ; arc != arcEnd ; ++arc) {
		assert(arc->iType == ARC_TYPE_HMM);
		DNode *nodeDest = m_nodes+arc->iNodeDest;
		// compute emission probability
		fScore = arc->state->computeEmissionProbability(vFeatureVector.getData(),0);	
		// apply insertion-penalty
		fScore += m_dynamicNetwork->getIP((m_nodes+arc->iNodeDest)->iIPIndex);	
		
		if (fScore > m_fScoreBest-m_fBeamWidthNodes) {
			if (fScore > m_fScoreBest) {m_fScoreBest = fScore;}	
		
			int iToken = newToken();
			Token *token = m_tokensNext+iToken;
			token->fScore = fScore;
			token->state = arc->state;
			token->iLMState = iLMStateInitial;
			token->iLexUnitPron = m_iLexUnitPronUnknown;
			token->iNode = arc->iNodeDest;
			token->iHistoryItem = (int)(historyItemBegSentence-m_historyItems);
			
			// activate the token
			assert(nodeDest->iActiveTokensNextBase == -1);
			nodeDest->iActiveTokensNextBase = newActiveTokenTable();
			(m_activeTokenNext+nodeDest->iActiveTokensNextBase)[0].iLMState = token->iLMState;
			(m_activeTokenNext+nodeDest->iActiveTokensNextBase)[0].iToken = iToken;
			nodeDest->iActiveTokensNext = 1;	
			m_nodesActiveNext[m_iNodesActiveNext++] = nodeDest;
			assert(m_iNodesActiveNext < m_iNodesActiveNextMax);
		}
	}	
}

// regular expansion
void DynamicDecoderX::expand(VectorBase<float> &vFeatureVector, int t) {
	m_fScoreBest = -FLT_MAX;
	m_fScoreBestWE = -FLT_MAX;
	float fScore , fScoreToken;
	// expand nodes in the active states
	// (1) self-loop (hmm is in the token) (no token recombination is needed since all LM-states in the arc are different)
	for(int l=0 ; l < m_iNodesActiveCurrent ; ++l) {
		DNode *node = m_nodesActiveCurrent[l];
		ActiveToken *activeTokensCurrent = m_activeTokenCurrent+node->iActiveTokensCurrentBase;
		HMMStateDecoding *state = (m_tokensCurrent+activeTokensCurrent[0].iToken)->state;
		// (1) self loop (the hmm-state is in the token)
		// compute emission probability
		fScore = state->computeEmissionProbability(vFeatureVector.getData(),t);	
		// regular-node
		float *fScoreBest = &m_fScoreBest;
		float fBeamWidth = m_fBeamWidthNodes;
		// propagate tokens within the arc
		for(int i=0 ; i < node->iActiveTokensCurrent ; ++i) {	
			Token *token = m_tokensCurrent+(activeTokensCurrent+i)->iToken;
			fScoreToken = token->fScore+fScore;
			if (fScoreToken > (*fScoreBest-fBeamWidth)) {
				// keep higher score
				if (fScoreToken > *fScoreBest) {
					*fScoreBest = fScoreToken;
				}
				// create expanded token
				int iToken = newToken();
				Token *tokenAux = m_tokensNext+iToken;
				tokenAux->fScore = fScoreToken;
				tokenAux->state = token->state;
				tokenAux->iLMState = token->iLMState;
				tokenAux->iLexUnitPron = token->iLexUnitPron;
				tokenAux->iNode = (int)(node-m_nodes);
				tokenAux->iHistoryItem = token->iHistoryItem;
				//tokenAux->fLAScores = token->fLAScores;
				//tokenAux->iWGToken = -1;	
				
				// activate the arc
				if (node->iActiveTokensNextBase == -1) {
					node->iActiveTokensNextBase = newActiveTokenTable();
					node->iActiveTokensNext = 0;
					m_nodesActiveNext[m_iNodesActiveNext++] = node;
					assert(m_iNodesActiveNext < m_iNodesActiveNextMax);
				}
				(m_activeTokenNext+node->iActiveTokensNextBase)[node->iActiveTokensNext].iLMState = token->iLMState;
				(m_activeTokenNext+node->iActiveTokensNextBase)[node->iActiveTokensNext].iToken = iToken;
				node->iActiveTokensNext++;
				assert(node->iActiveTokensNext < m_iTokensNodeMax);
			}
		}
	}
		
	// (2) outgoing arcs
	for(int i=0 ; i < m_iNodesActiveCurrent ; ++i) {
		DNode *node = m_nodesActiveCurrent[i];
		// word-end? if so, get ready to extend word history
		m_iHistoryItemsAux = NULL;
		if (node->bWordEnd) {
			m_iHistoryItemsAuxSize = node->iActiveTokensCurrent;
			assert(node->iActiveTokensCurrent < m_iTokensNodeMax);
			m_iHistoryItemsAux = m_iHistoryItemsAuxBuffer;
			for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
				m_iHistoryItemsAux[j] = -1;
			}
		}	
			
		DArc *arcEnd = m_arcs+(node+1)->iArcNext;
		for(DArc *arcNext = m_arcs+node->iArcNext ; arcNext != arcEnd ; ++arcNext) {
			// hmm-arc
			if (arcNext->iType == ARC_TYPE_HMM) {
				expandToHMM(node,arcNext,vFeatureVector,t);	
			} 
			// word-arc
			else if (arcNext->iType == ARC_TYPE_WORD) {	
				// lm-transition
				LMTransition *lmTransition = new LMTransition[node->iActiveTokensCurrent];
				for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
					lmTransition[j].iLMState = -1;
				}	
							
				DNode *node2 = m_nodes+arcNext->iNodeDest;
				DArc *arcEnd2 = m_arcs+(node2+1)->iArcNext;
				for(DArc *arcNext2 = m_arcs+node2->iArcNext ; arcNext2 != arcEnd2 ; ++arcNext2) {
					// hmm-arc
					if (arcNext2->iType == ARC_TYPE_HMM) {	
						expandToHMMNewWord(node,arcNext2,arcNext->lexUnit,lmTransition,vFeatureVector,t);
					} 
					// null-arc
					else {
						assert(arcNext2->iType == ARC_TYPE_NULL);
						
						DNode *node3 = m_nodes+arcNext2->iNodeDest;	
						DArc *arcEnd3 = m_arcs+(node3+1)->iArcNext;
						for(DArc *arcNext3 = m_arcs+node3->iArcNext ; arcNext3 != arcEnd3 ; ++arcNext3) {
							// hmm-arc
							assert(arcNext3->iType == ARC_TYPE_HMM);
							expandToHMMNewWord(node,arcNext3,arcNext->lexUnit,lmTransition,vFeatureVector,t);
						}	
					}
				}
				
				// there can be multiple words (homophones) getting to a starting node
				if (node->bWordEnd) {
					for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
						m_iHistoryItemsAux[j] = -1;
					}
				}	
				
				delete [] lmTransition;
			}
			// null-arc
			else {
				assert(arcNext->iType == ARC_TYPE_NULL);
				DNode *node2 = m_nodes+arcNext->iNodeDest;	
				DArc *arcEnd2 = m_arcs+(node2+1)->iArcNext;
				for(DArc *arcNext2 = m_arcs+node2->iArcNext ; arcNext2 != arcEnd2 ; ++arcNext2) {
					// hmm-arc
					if (arcNext2->iType == ARC_TYPE_HMM) {	
						expandToHMM(node,arcNext2,vFeatureVector,t);
					} 
					// word-arc
					else {
						assert(arcNext2->iType == ARC_TYPE_WORD);	
						// lm-transition
						LMTransition *lmTransition = new LMTransition[node->iActiveTokensCurrent];
						for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
							lmTransition[j].iLMState = -1;
						}	
						DNode *node3 = m_nodes+arcNext2->iNodeDest;	
						DArc *arcEnd3 = m_arcs+(node3+1)->iArcNext;
						for(DArc *arcNext3 = m_arcs+node3->iArcNext ; arcNext3 != arcEnd3 ; ++arcNext3) {
							// hmm-arc
							if (arcNext3->iType == ARC_TYPE_HMM) {
								expandToHMMNewWord(node,arcNext3,arcNext2->lexUnit,lmTransition,vFeatureVector,t);	
							} 
							// null-arc
							else {
								assert(arcNext3->iType == ARC_TYPE_NULL);
							
								DNode *node4 = m_nodes+arcNext3->iNodeDest;	
								DArc *arcEnd4 = m_arcs+(node4+1)->iArcNext;
								for(DArc *arcNext4 = m_arcs+node4->iArcNext ; arcNext4 != arcEnd4 ; ++arcNext4) {
									// hmm-arc
									assert(arcNext3->iType == ARC_TYPE_HMM);
									expandToHMMNewWord(node,arcNext4,arcNext2->lexUnit,lmTransition,vFeatureVector,t);	
								}
							}	
						}
						// there can be multiple words (homophones) getting to a starting node
						if (node->bWordEnd) {
							for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
								m_iHistoryItemsAux[j] = -1;
							}
						}	
						
						delete [] lmTransition;
					}
				}	
			}
		}
		
		node->iActiveTokensCurrent = 0;
		node->iActiveTokensCurrentBase = -1;
	}
}

// expand a series of tokens to a hmm-state
void DynamicDecoderX::expandToHMM(DNode *node, DArc *arcNext, VectorBase<float> &vFeatureVector, int t) {
	float fScore;
	float fScoreToken;
	ActiveToken *activeTokensCurrent = m_activeTokenCurrent+node->iActiveTokensCurrentBase;
	DNode *nodeNext = m_nodes+arcNext->iNodeDest;
	ActiveToken *activeTokensNext = m_activeTokenNext+nodeNext->iActiveTokensNextBase;

	// compute emission probability
	fScore = arcNext->state->computeEmissionProbability(vFeatureVector.getData(),t);	
	bool bWordEnd = (nodeNext->iIPIndex != -1);

	// apply insertion-penalty?
	if (bWordEnd) {fScore += m_dynamicNetwork->getIP(m_nodes[arcNext->iNodeDest].iIPIndex);}

	// regular-node
	float *fScoreBest = &m_fScoreBest;
	float fBeamWidth = m_fBeamWidthNodes;
	// we-node
	// propagate tokens within the arc
	for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
		Token *token = m_tokensCurrent+(activeTokensCurrent+j)->iToken;
		fScoreToken = token->fScore+fScore;
		
		
		if (fScoreToken > (*fScoreBest-fBeamWidth)) {
			// keep higher score
			if (fScoreToken > *fScoreBest) {
				*fScoreBest = fScoreToken;
			}
			
			// token recombination?
			bool bFound = false;
			for(int k=0 ; k < nodeNext->iActiveTokensNext ; ++k) {
				if (activeTokensNext[k].iLMState == token->iLMState) {
					Token *tokenRec = m_tokensNext+activeTokensNext[k].iToken;
						if (fScoreToken > tokenRec->fScore) {
							// recombine
							tokenRec->fScore = fScoreToken;
							tokenRec->iLexUnitPron = token->iLexUnitPron;
							//assert(tokenRec->state == arcNext->state);
							// (a) not an end-of-word
							//if (m_iHistoryItemsAux == NULL) {
							if (bWordEnd == false) {
								tokenRec->iHistoryItem = token->iHistoryItem;
								 	
							} 
							// (b) end-of-word
							else {
								assert(m_iHistoryItemsAux != NULL);
								if (m_iHistoryItemsAux[j] == -1) {
									m_iHistoryItemsAux[j] = newHistoryItem();
									HistoryItem &historyItem = m_historyItems[m_iHistoryItemsAux[j]];
									historyItem.iLexUnitPron = token->iLexUnitPron;
									historyItem.iEndFrame = t-1;
									historyItem.fScore = token->fScore;
									historyItem.iPrev = token->iHistoryItem;
									assert(m_historyItems[historyItem.iPrev].iEndFrame < historyItem.iEndFrame);
									historyItem.iActive = -1;
									historyItem.iWGToken = -1;
								}
								tokenRec->iHistoryItem = m_iHistoryItemsAux[j];
								 
							}
						}
					bFound = true;
					break;
				}
			}
			if (bFound) { continue; }
		
			// create expanded token (no recombination)
			int iToken = newToken();
			Token *tokenAux = m_tokensNext+iToken;
			tokenAux->fScore = fScoreToken;
			tokenAux->state = arcNext->state;
			tokenAux->iLMState = token->iLMState;
			tokenAux->iLexUnitPron = token->iLexUnitPron;
			tokenAux->iNode = arcNext->iNodeDest;
			
			// (a) not an end-of-word
			if (bWordEnd == false) {
				tokenAux->iHistoryItem = token->iHistoryItem;
				// word-graph generation
				 
			} 
			// (b) end-of-word
			else {
				assert(m_iHistoryItemsAux != NULL);	
				if (m_iHistoryItemsAux[j] == -1) {
					m_iHistoryItemsAux[j] = newHistoryItem();
					HistoryItem &historyItem = m_historyItems[m_iHistoryItemsAux[j]];
					historyItem.iLexUnitPron = token->iLexUnitPron;
					historyItem.iEndFrame = t-1;
					historyItem.fScore = token->fScore;
					historyItem.iPrev = token->iHistoryItem;
					assert(m_historyItems[historyItem.iPrev].iEndFrame < historyItem.iEndFrame);
					historyItem.iActive = -1;
					// word-graph generation?
					historyItem.iWGToken = -1;
					
				}
				tokenAux->iHistoryItem = m_iHistoryItemsAux[j];
				// word-graph generation?
				 
				
			}
			
			// activate the node
			if (nodeNext->iActiveTokensNextBase == -1) {
				nodeNext->iActiveTokensNextBase = newActiveTokenTable();
				nodeNext->iActiveTokensNext = 0;
				activeTokensNext = m_activeTokenNext+nodeNext->iActiveTokensNextBase;
				m_nodesActiveNext[m_iNodesActiveNext++] = nodeNext;	
				assert(m_iNodesActiveNext < m_iNodesActiveNextMax);
			}
			activeTokensNext[nodeNext->iActiveTokensNext].iLMState = token->iLMState;
			activeTokensNext[nodeNext->iActiveTokensNext].iToken = iToken;
			nodeNext->iActiveTokensNext++;
			if (nodeNext->iActiveTokensNext >= m_iTokensNodeMax) {
				pruneExtraTokens(nodeNext);
			}
			assert(nodeNext->iActiveTokensNext < m_iTokensNodeMax);
		}
	}
}

// expand a series of tokens to a hmm-state after obsering a new word
void DynamicDecoderX::expandToHMMNewWord(DNode *node, DArc *arcNext, LexUnit *lexUnit, LMTransition *lmTransition, 
	VectorBase<float> &vFeatureVector, int t) {
	float fScore;
	float fScoreLM;
	float fScoreToken;
	ActiveToken *activeTokensCurrent = m_activeTokenCurrent+node->iActiveTokensCurrentBase;
	DNode *nodeNext = m_nodes+arcNext->iNodeDest;
	ActiveToken *activeTokensNext = m_activeTokenNext+nodeNext->iActiveTokensNextBase;
	
	// compute emission probability
	fScore = arcNext->state->computeEmissionProbability(vFeatureVector.getData(),t);	
	bool bWordEnd = (nodeNext->iIPIndex != -1);
	// apply insertion-penalty?
	if (bWordEnd) {
		fScore += m_dynamicNetwork->getIP(m_nodes[arcNext->iNodeDest].iIPIndex);
	}
	// regular-arc
	float *fScoreBest = &m_fScoreBest;
	float fBeamWidth = m_fBeamWidthNodes;
 
	// propagate tokens within the node
	for(int j=0 ; j < node->iActiveTokensCurrent ; ++j) {
		Token *token = m_tokensCurrent+activeTokensCurrent[j].iToken;
		int iLMState = -1;
		// standard lexical unit: apply LM-score and insertion penalty
		if (m_lexiconManager->isStandard(lexUnit)) {
			// compute LM-score if needed
			if (lmTransition[j].iLMState == -1) {
				lmTransition[j].iLMState = m_lmFSM->updateLMState(token->iLMState,
					lexUnit->iLexUnit,&lmTransition[j].fScoreLM);
				lmTransition[j].fScoreLM *= m_fLMScalingFactor;
			}
			iLMState = lmTransition[j].iLMState;
			fScoreLM = lmTransition[j].fScoreLM;
		} 
		// filler lexical unit: keep original lm state 
		else {
			iLMState = token->iLMState;
			fScoreLM = 0.0;
		}
		fScoreToken = token->fScore+fScore+fScoreLM;
		if (fScoreToken > (*fScoreBest-fBeamWidth)) {
			// keep higher score
			if (fScoreToken > *fScoreBest) {
				*fScoreBest = fScoreToken;
			}
			// token recombination?
			bool bFound = false;
			for(int k=0 ; k < nodeNext->iActiveTokensNext ; ++k) {
				if (activeTokensNext[k].iLMState == iLMState) {
					Token *tokenRec = m_tokensNext+activeTokensNext[k].iToken;
					assert(tokenRec->state == arcNext->state);
						if (fScoreToken > tokenRec->fScore) {
							// recombine
							tokenRec->fScore = fScoreToken;
							tokenRec->iLexUnitPron = lexUnit->iLexUnitPron;
							// (a) not an end-of-word
							//if (m_iHistoryItemsAux == NULL) {
							if (bWordEnd == false) {
								tokenRec->iHistoryItem = token->iHistoryItem;
								 
							} 
							// (b) end-of-word
							else {
								assert(m_iHistoryItemsAux != NULL);
								if (m_iHistoryItemsAux[j] == -1) {
									m_iHistoryItemsAux[j] = newHistoryItem();
									HistoryItem &historyItem = m_historyItems[m_iHistoryItemsAux[j]];	
									historyItem.iLexUnitPron = lexUnit->iLexUnitPron;
									historyItem.iEndFrame = t-1;
									historyItem.fScore = token->fScore+fScoreLM;
									historyItem.iPrev = token->iHistoryItem;
									assert(m_historyItems[historyItem.iPrev].iEndFrame < historyItem.iEndFrame);
									historyItem.iActive = -1;
									historyItem.iWGToken = -1;
								}
								tokenRec->iHistoryItem = m_iHistoryItemsAux[j];

							}
						} 
					bFound = true;
					break;
				}
			}
			if (bFound) { continue; }
			// create expanded token (no recombination)
			int iToken = newToken();
			Token *tokenAux = m_tokensNext+iToken;
			tokenAux->fScore = fScoreToken;
			tokenAux->state = arcNext->state;
			tokenAux->iLMState = iLMState;
			tokenAux->iLexUnitPron = lexUnit->iLexUnitPron;
			tokenAux->iNode = (int)(nodeNext-m_nodes);
			 
			// (a) not an end-of-word
			if (bWordEnd == false) {
				tokenAux->iHistoryItem = token->iHistoryItem;
				// word-graph generation?
				 
			} 
			// (b) end-of-word
			else {
				assert(m_iHistoryItemsAux != NULL);
				if (m_iHistoryItemsAux[j] == -1) {
					m_iHistoryItemsAux[j] = newHistoryItem();
					HistoryItem &historyItem = m_historyItems[m_iHistoryItemsAux[j]];
					historyItem.iLexUnitPron = lexUnit->iLexUnitPron;
					historyItem.iEndFrame = t-1;
					historyItem.fScore = token->fScore+fScoreLM;
					historyItem.iPrev = token->iHistoryItem;
					assert(m_historyItems[historyItem.iPrev].iEndFrame < historyItem.iEndFrame);
					historyItem.iActive = -1;
					// word-graph generation?
					 historyItem.iWGToken = -1;
				}
				tokenAux->iHistoryItem = m_iHistoryItemsAux[j];
				// word-graph generation?
				 
				assert((tokenAux->iHistoryItem+m_historyItems)->iWGToken >= -1);
			}
			// activate the node
			if (nodeNext->iActiveTokensNextBase == -1) {
				assert(nodeNext->iActiveTokensNext == 0);
				nodeNext->iActiveTokensNextBase = newActiveTokenTable();
				nodeNext->iActiveTokensNext = 0;
				activeTokensNext = m_activeTokenNext+nodeNext->iActiveTokensNextBase;
				m_nodesActiveNext[m_iNodesActiveNext++] = nodeNext;
				assert(m_iNodesActiveNext < m_iNodesActiveNextMax);
			}
			activeTokensNext[nodeNext->iActiveTokensNext].iLMState = iLMState;
			activeTokensNext[nodeNext->iActiveTokensNext].iToken = iToken;
			nodeNext->iActiveTokensNext++;
			if (nodeNext->iActiveTokensNext >= m_iTokensNodeMax) {
				pruneExtraTokens(nodeNext);
			}	
			assert(nodeNext->iActiveTokensNext < m_iTokensNodeMax);
		}
	}
}


// pruning (token based)
void DynamicDecoderX::pruning() {
	assert(m_fScoreBestWE <= m_fScoreBest);
	m_fScoreBestWE = m_fScoreBest; //	HACK
	// compute threshold for beam based pruning
	int iNumberBins = NUMBER_BINS_HISTOGRAM;
	int iSurvivorsRegular = 0;
	float fThresholdRegular = m_fScoreBest-m_fBeamWidthNodes;	
	
	// (1) apply within-node pruning
	// apply pruning to each arc
	m_iNodesActiveCurrent = 0;
	int iPrunedTotal = 0;	
	int *iBins = new int[iNumberBins];
	int iBin;
	float fScoreBestNode = -FLT_MAX;
	float *fThresholdNode = new float[m_iNodesActiveNext];
	int iSurvivorsAll = 0;
	for(int i=0 ; i < m_iNodesActiveNext ; ++i) {
		DNode *node = m_nodesActiveNext[i];
		assert(node->iActiveTokensNext > 0);
		fThresholdNode[i] = fThresholdRegular;
		
		// (a) histogram pruning within the node
		if (node->iActiveTokensNext > m_iMaxActiveTokensNode) {
		
			// get the best score within the node
			fScoreBestNode = -FLT_MAX;
			for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
				Token *token = m_tokensNext+(m_activeTokenNext+node->iActiveTokensNextBase)[j].iToken;
				if (token->fScore > fScoreBestNode) {
					fScoreBestNode = token->fScore;
				}
			}
			// compute bin-size for histogram pruning
			float fLength = m_fBeamWidthTokensNode+1;
			float fBinSize = fLength/((float)iNumberBins);
			assert(fBinSize > 0);
					
			fThresholdNode[i] = max(fScoreBestNode-m_fBeamWidthTokensNode,fThresholdRegular);
		
			// (2.1) compute the size of each bin and initialize them
			for(int j = 0 ; j < iNumberBins ; ++j) {
				iBins[j] = 0;
			}
			// (2.2) fill the bins, the first bin keeps the best tokens
			ActiveToken *activeTokensNext = m_activeTokenNext+node->iActiveTokensNextBase;
			for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
				Token *token = m_tokensNext+activeTokensNext[j].iToken;
				if (token->fScore > fThresholdNode[i]) {
					iBin = (int)(fabs(token->fScore-fScoreBestNode)/fBinSize);
					assert((iBin >= 0) && (iBin < iNumberBins));
					iBins[iBin]++;
				}
			}
			// (2.3) get the threshold
			int iSurvivors = 0;
			float fThresholdHistogram = fThresholdRegular;
			for(int j = 0 ; j < iNumberBins-1 ; ++j) {
				iSurvivors += iBins[j];
				// this is the cut-off
				if (iSurvivors >= m_iMaxActiveTokensNode) {
					fThresholdHistogram = fScoreBestNode-((j+1)*fBinSize);
					iSurvivorsAll += iSurvivors;
					break;
				}
			}
			fThresholdNode[i] = max(fThresholdNode[i],fThresholdHistogram);
		} else {
			iSurvivorsAll += node->iActiveTokensNext;
		}
	}
	
	// (2) global-pruning
	// histogram pruning?
	if (iSurvivorsAll > m_iMaxActiveNodes) {
		// (2.1) compute the size of each bin and initialize them
		float fLengthRegular = m_fBeamWidthNodes+1;
		float fBinSizeRegular = ((float)fLengthRegular)/((float)iNumberBins);
		assert(fBinSizeRegular > 0);
		int *iBinsRegular = new int[iNumberBins];
		int iRegular = 0;
		for(int i = 0 ; i < iNumberBins ; ++i) {
			iBinsRegular[i] = 0;
		}
		// (2.2) fill the bins, the first bin keeps the best tokens
		int iBin;
		for(int i=0 ; i < m_iNodesActiveNext ; ++i) {
			ActiveToken *activeTokens = m_activeTokenNext+m_nodesActiveNext[i]->iActiveTokensNextBase;
			for(int j=0 ; j < m_nodesActiveNext[i]->iActiveTokensNext ; ++j) {
				Token *token = m_tokensNext+activeTokens[j].iToken;
				if (token->fScore <= fThresholdNode[i]) {
					continue;
				}	
				iBin = (int)(fabs(token->fScore-m_fScoreBest)/fBinSizeRegular);
				assert((iBin >= 0) && (iBin < iNumberBins));
				iBinsRegular[iBin]++;
				++iRegular;
			}
		}	
		// (2.3) get the threshold
		float fThresholdHistogramRegular = fThresholdRegular;
		for(int i = 0 ; i < iNumberBins-1 ; ++i) {
			iSurvivorsRegular += iBinsRegular[i];
			// this is the cut-off
			if (iSurvivorsRegular >= m_iMaxActiveNodes) {
				fThresholdHistogramRegular = m_fScoreBest-((i+1)*fBinSizeRegular);
				break;
			}
		}
		fThresholdRegular = max(fThresholdRegular,fThresholdHistogramRegular);
		delete [] iBinsRegular;
	}	
	// (3) actual pruning
	for(int i=0 ; i < m_iNodesActiveNext ; ++i) {
		DNode *node = m_nodesActiveNext[i];
		fThresholdNode[i] = max(fThresholdNode[i],fThresholdRegular);	
		
		int iPruned = 0;
		int iAvailable = -1;
		ActiveToken *activeTokensNext = m_activeTokenNext+node->iActiveTokensNextBase;
		for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
			Token *token = m_tokensNext+activeTokensNext[j].iToken;
			if (token->fScore < fThresholdNode[i]) {
				if (iAvailable == -1) {
					iAvailable = j;	
				}
				++iPruned;
			} else if (iAvailable != -1) {
				activeTokensNext[iAvailable] = activeTokensNext[j];
				++iAvailable;
			}
		}
		iPrunedTotal += iPruned;
		assert(node->iActiveTokensCurrentBase == -1);
		assert(node->iActiveTokensCurrent == 0);
		int iSurvivors = node->iActiveTokensNext-iPruned;
		//iSurvivorsH[iSurvivors]++;
		if (iSurvivors > 0) {
			assert(node->iActiveTokensNextBase != -1);
			node->iActiveTokensCurrentBase = node->iActiveTokensNextBase;
			node->iActiveTokensCurrent = iSurvivors;
			m_nodesActiveCurrent[m_iNodesActiveCurrent++] = node;
		}
		node->iActiveTokensNextBase = -1;
		node->iActiveTokensNext = 0;
	}
	
	delete [] iBins;
	delete [] fThresholdNode;
	//printf("# tokens active: (%d -> %d)\n",m_iTokensNext,m_iTokensNext-iPrunedTotal);
	// clean the table of next active states
	m_iNodesActiveNext = 0;	
	// swap the token tables
	swapTokenTables();
}
		
// return the BestPath
BestPath *DynamicDecoderX::getBestPath() {
	assert(m_bInitialized);
	// (1) get the best scoring token
	float fScoreBestWE = -FLT_MAX;
	Token *tokenBest = NULL;
	LexUnit *lexUnitLast = NULL;
	float fScoreToken;
	float fScoreLM1;
	int iLMState;
	
	// for each active token
	for(int i=0 ; i < m_iTokensNext ; ++i) {
		Token *token = &m_tokensNext[i];
		// check if pruned by "pruneExtraTokens"
		if (token->iNode == -1) {	
			continue;
		}
		DNode *node = m_nodes+token->iNode;
		// only tokens in word-end nodes
		if (node->bWordEnd == false) {
			continue;
		}
		// two possibilities: 
		// (a) monophones: the node goes to a word-arc before any hmm-node is seen 
		// (b) multi-phones: the node goes directly to an hmm-state
		VLexUnit vLexUnitDest;
		getDestinationMonophoneLexUnits(node,vLexUnitDest);
		// (a) monophones
		if (vLexUnitDest.empty() == false) {
			for(VLexUnit::iterator it = vLexUnitDest.begin() ; it != vLexUnitDest.end() ; ++it) {	
				fScoreLM1 = 0.0;
				iLMState = -1;
				if (m_lexiconManager->isStandard(*it)) {
					iLMState = m_lmFSM->updateLMState(token->iLMState,(*it)->iLexUnit,&fScoreLM1);
					fScoreLM1 *= m_fLMScalingFactor;
				} else {
					iLMState = token->iLMState;
				}
				float fScoreLM2 = m_lmFSM->toFinalState(iLMState)*m_fLMScalingFactor;
				fScoreToken = token->fScore + fScoreLM1 + fScoreLM2;
				if (fScoreToken > fScoreBestWE) {
					fScoreBestWE = fScoreToken;
					tokenBest = token;
					lexUnitLast = *it;
				}
			}
		}
		// (b) multi-phones 
		else {
			fScoreLM1 = m_lmFSM->toFinalState(token->iLMState)*m_fLMScalingFactor;
			fScoreToken = token->fScore + fScoreLM1;
			if (fScoreToken > fScoreBestWE) {
				fScoreBestWE = fScoreToken;
				tokenBest = token;
				lexUnitLast = m_lexiconManager->getLexUnitPron(token->iLexUnitPron);
			}
		}
	}
	
	// no active tokens at terminal nodes:
	if (tokenBest == NULL) {
		BVC_WARNING << "unable to retrieve the best decoding path, no active tokens found at terminal nodes";
		// TODO in this scenario, which is usually very rare, it would be possible to generate a best path by 
		// modifying the path in the best token by doing a) or b):
		// a) changing the alignment of the last word/silence so it ends a the last HMM-state (use forced alignment)
		// b) remove last word and run force alignment on the rest in order to compute best path score
		return NULL;
	}	
	
	// (2) get the best sequence of lexical units from the best scoring token
	BestPath *bestPath = new BestPath(m_lexiconManager,fScoreBestWE);
	int iHistoryItem = tokenBest->iHistoryItem;
	int iFrameStart;
	int iFrameEnd;
	float fScore;
	while (iHistoryItem != -1) {
		HistoryItem *historyItem = iHistoryItem+m_historyItems;
		if (historyItem->iPrev != -1) {
			iFrameStart = max(0,m_historyItems[historyItem->iPrev].iEndFrame+1);			// (INT_MIN is used for the initial item)
			iFrameEnd = historyItem->iEndFrame;
			assert(iFrameStart < iFrameEnd);
			fScore = historyItem->fScore;
		} else {
			iFrameStart = -1;
			iFrameEnd = -1;
			fScore = 0.0;
		}
		// get the observed lexical unit
		LexUnit *lexUnit = m_lexiconManager->getLexUnitPron(historyItem->iLexUnitPron);
		// add a new element
		bestPath->newElementFront(iFrameStart,iFrameEnd,fScore,0.0,0.0,0.0,lexUnit,0.0);
		iHistoryItem = m_historyItems[iHistoryItem].iPrev;
	}

	// add the final lexical unit if any
	assert(lexUnitLast != NULL);
	if (m_historyItems[tokenBest->iHistoryItem].iLexUnitPron != m_lexiconManager->m_lexUnitBegSentence->iLexUnitPron) {
		iFrameStart = m_historyItems[tokenBest->iHistoryItem].iEndFrame+1;
	} else {
		iFrameStart = 0;
	}
	bestPath->newElementBack(iFrameStart,m_iFeatureVectorsUtterance-1,tokenBest->fScore,0.0,0.0,0.0,lexUnitLast,0.0);
	// add the end of sentence
	bestPath->newElementBack(-1,-1,0.0,0.0,0.0,0.0,m_lexiconManager->m_lexUnitEndSentence,0.0);
	// TODO attach lm-scores and compute real am-scores by substracting lm-score and insertion penalty
	BVC_VERB	<< "best score:    " << FLT(12,4) << m_fScoreBest;
	BVC_VERB	<< "best WE score: " << FLT(12,4) << fScoreBestWE << " " << FLT(12,4) << m_fScoreBestWE;
	
	return bestPath;
}

// get monophone lexical units accessible right aftet the given hmm-node 
void DynamicDecoderX::getDestinationMonophoneLexUnits(DNode *node, VLexUnit &vLexUnitDest) {
	map<int,bool> mLexUnitSeen;

	DArc *arcEnd = m_arcs+(node+1)->iArcNext;
	for(DArc *arcNext = m_arcs+node->iArcNext ; arcNext != arcEnd ; ++arcNext) {
		
		// word-arc
		if (arcNext->iType == ARC_TYPE_WORD) {
			vLexUnitDest.push_back(arcNext->lexUnit);
		}
		// null-arc
		else if (arcNext->iType == ARC_TYPE_NULL) {
		
			DNode *node2 = m_nodes+arcNext->iNodeDest;	
			DArc *arcEnd2 = m_arcs+(node2+1)->iArcNext;
			for(DArc *arcNext2 = m_arcs+node2->iArcNext ; arcNext2 != arcEnd2 ; ++arcNext2) {	
				if (arcNext2->iType == ARC_TYPE_WORD) {
					if (arcNext2->lexUnit->vPhones.size() == 1) {
						if (mLexUnitSeen.find(arcNext2->lexUnit->iLexUnit) == mLexUnitSeen.end()) {
							//m_lexiconManager->print(arcNext2->lexUnit);	
							mLexUnitSeen.insert(map<int,bool>::value_type(arcNext2->lexUnit->iLexUnit,true));
							vLexUnitDest.push_back(arcNext2->lexUnit);
						}
					}
				}
			}	
		}
	}
}

// garbage collection of history items
// (1) it starts by marking the active items by traversing back items from the active states
// (2) it adds inactive items to the queue of available items
void DynamicDecoderX::historyItemGarbageCollection() {
	int iItemsActive = 0;
	
	// (1) check if garbage collection was already run within the current time frame
	// note: this is an undesirable situation because it requires an extra pass over the complete array of items
	// it should be avoided by allocating a larger number of entries from the beginning
	if (m_iTimeGarbageCollectionLast == m_iTimeCurrent) {
		// mark all the history items as inactive
		for(unsigned int i=0 ; i < m_iHistoryItems ; ++i) {
			m_historyItems[i].iActive = -1;
		}	
	}	
	
	// (2) mark items coming from active arcs as active
	// (2.1) active arcs for current time frame
	for(int i=0 ; i < m_iNodesActiveCurrent ; ++i) {
		ActiveToken *activeTokens = m_activeTokenCurrent+m_nodesActiveCurrent[i]->iActiveTokensCurrentBase;
		for(int j=0 ; j < m_nodesActiveCurrent[i]->iActiveTokensCurrent ; ++j) {
			Token *token = m_tokensCurrent+activeTokens[j].iToken;
			int iHistoryItem = token->iHistoryItem;	
			while((iHistoryItem != -1) && ((m_historyItems+iHistoryItem)->iActive != m_iTimeCurrent)) {
				(m_historyItems+iHistoryItem)->iActive = m_iTimeCurrent;	
				iHistoryItem = (m_historyItems+iHistoryItem)->iPrev;
				++iItemsActive;
			}
		}
	}	
	// (2.2) active arcs for next time frame
	// TODO: it would be faster to traverse the array of next tokens, but this should not be too bad
	for(int i=0 ; i < m_iNodesActiveNext ; ++i) {
		ActiveToken *activeTokens = m_activeTokenNext+m_nodesActiveNext[i]->iActiveTokensNextBase;
		for(int j=0 ; j < m_nodesActiveNext[i]->iActiveTokensNext ; ++j) {
			Token *token = m_tokensNext+activeTokens[j].iToken;
			int iHistoryItem = token->iHistoryItem;	
			while((iHistoryItem != -1) && ((m_historyItems+iHistoryItem)->iActive != m_iTimeCurrent)) {
				(m_historyItems+iHistoryItem)->iActive = m_iTimeCurrent;	
				iHistoryItem = (m_historyItems+iHistoryItem)->iPrev;
				++iItemsActive;
			}
		}
	}
	// check also auxiliar arrays for history items in use
	if (m_iHistoryItemsAux != NULL) {
		for(int i=0 ; i < m_iHistoryItemsAuxSize ; ++i) {
			if (m_iHistoryItemsAux[i] != -1) {
				if ((m_historyItems+m_iHistoryItemsAux[i])->iActive != m_iTimeCurrent) {	
					(m_historyItems+m_iHistoryItemsAux[i])->iActive = m_iTimeCurrent;
					++iItemsActive;
				}
			}
		}	
	}
	// (3) if a certain percentage* of the items are active then we need to allocate 
	// a bigger data structure to keep the new history items
	// (* if we wait until all the items are active there will be many calls to the garbage collector
	// when the array of reach a high occupation, that would introduce substantial overhead)
	assert(iItemsActive <= (int)m_iHistoryItems);
	if (iItemsActive >= (0.20*m_iHistoryItems)) {

		//printf("history item garbage collection...\n");	
		//printf("allocating space for new items (item sused: %d existing: %d)\n",iItemsActive,m_iHistoryItems);	
		
		// allocate a new data structure with double capacity
		HistoryItem *historyItems = NULL;
		try {
			historyItems = new HistoryItem[m_iHistoryItems*2];
		} 
		catch (const std::bad_alloc&) {
			int iBytes = m_iHistoryItems*2*sizeof(WGToken);
			BVC_ERROR << "unable to allocate memory for history items, " << iBytes << " Bytes needed";
		}
		
		// copy the active items from the old data structure
		for(unsigned int i=0 ; i < m_iHistoryItems ; ++i) {
			historyItems[i].iLexUnitPron = m_historyItems[i].iLexUnitPron;
			historyItems[i].iEndFrame = m_historyItems[i].iEndFrame;
			historyItems[i].fScore = m_historyItems[i].fScore;
			historyItems[i].iActive = m_iTimeCurrent;
			historyItems[i].iWGToken = m_historyItems[i].iWGToken;
			historyItems[i].iPrev = m_historyItems[i].iPrev;
		}
		
		// create the linked list of available items
		for(unsigned int i=m_iHistoryItems ; i < (2*m_iHistoryItems)-1 ; ++i) {
			historyItems[i].iPrev = i+1;
			historyItems[i].iActive = -1;	
		}
		historyItems[(2*m_iHistoryItems)-1].iPrev = -1;
		historyItems[(2*m_iHistoryItems)-1].iActive = -1;
		
		delete [] m_historyItems;
		m_historyItems = historyItems;
		m_iHistoryItemAvailable = m_iHistoryItems;
		m_iHistoryItems *= 2;
	}
	// (3') there are inactive items: create a linked list with them
	else {
		int *iHistoryItemAux = &m_iHistoryItemAvailable;
		for(unsigned int i = 0 ; i < m_iHistoryItems ; ++i) {
			if (m_historyItems[i].iActive != m_iTimeCurrent) {
				m_historyItems[i].iActive = -1;
				m_historyItems[i].iEndFrame = -1;
				*iHistoryItemAux = i;
				iHistoryItemAux = &m_historyItems[i].iPrev;	
			}
		}
		*iHistoryItemAux = -1;
	}
	
	m_iTimeGarbageCollectionLast = m_iTimeCurrent;
	
	//printf("%d used %d total\n",iItemsActive,m_iHistoryItems);	
}





// prune active tokens that are not in the top-N within a node 
void DynamicDecoderX::pruneExtraTokens(DNode *node) {
	assert(node->iActiveTokensNext >= m_iTokensNodeMax);

	// get the best score within the node
	float fScoreBestNode = -FLT_MAX;
	for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
		Token *token = m_tokensNext+(m_activeTokenNext+node->iActiveTokensNextBase)[j].iToken;
		if (token->fScore > fScoreBestNode) {
			fScoreBestNode = token->fScore;
		}
	}
		
	// compute bin-size for histogram pruning
	int iNumberBins = NUMBER_BINS_HISTOGRAM_WITHIN_NODE;
	int *iBins = new int[iNumberBins];
	int iBin;
	float fLength = m_fBeamWidthTokensNode+1;
	float fBinSize = fLength/((float)iNumberBins);
	assert(fBinSize > 0);
			
	float fThresholdLikelihood = fScoreBestNode-m_fBeamWidthTokensNode;

	// compute the size of each bin and initialize them
	for(int j = 0 ; j < iNumberBins ; ++j) {
		iBins[j] = 0;
	}
	// fill the bins, the first bin keeps the best tokens
	ActiveToken *activeTokensNext = m_activeTokenNext+node->iActiveTokensNextBase;
	for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
		Token *token = m_tokensNext+activeTokensNext[j].iToken;
		if (token->fScore >= fThresholdLikelihood) {
			iBin = (int)(fabs(token->fScore-fScoreBestNode)/fBinSize);
			assert((iBin >= 0) && (iBin < iNumberBins));
			iBins[iBin]++;
		}
	}
	// get the threshold
	int iSurvivors = 0;
	float fThresholdHistogram = -FLT_MAX;
	for(int j = 0 ; j < iNumberBins ; ++j) {
		int iSum = iSurvivors+iBins[j];
		// this is the cut-off
		if (iSum > m_iMaxActiveTokensNode) {	
			fThresholdHistogram = fScoreBestNode-(j*fBinSize);
			break;
		}
		iSurvivors = iSum;
	}
	float fThresholdNode = max(fThresholdLikelihood,fThresholdHistogram);
	
	// actual pruning
	int iPruned = 0;
	int iAvailable = -1;
	for(int j=0 ; j < node->iActiveTokensNext ; ++j) {
		Token *token = m_tokensNext+activeTokensNext[j].iToken;
		if (token->fScore < fThresholdNode) {
			if (iAvailable == -1) {
				iAvailable = j;	
			}
			token->iNode = -1;
			++iPruned;	
		} else if (iAvailable != -1) {
			activeTokensNext[iAvailable] = activeTokensNext[j];
			++iAvailable;
		}
	}
	node->iActiveTokensNext = node->iActiveTokensNext-iPruned;
	
	// this should never happen
	if ((node->iActiveTokensNext >= m_iTokensNodeMax) || (node->iActiveTokensNext == 0)) {
		for(int i=0 ; i < iNumberBins ; ++i) {
			BVC_VERB << setw(3) << i << " -> " << setw(4) << iBins[i];
		}
	}
	
	assert(node->iActiveTokensNext < m_iTokensNodeMax);

	delete [] iBins;
}

// uninitialize
void DynamicDecoderX::uninitialize() {
	assert(m_bInitialized);
	delete [] m_nodesActiveCurrent;
	delete [] m_nodesActiveNext;
	delete [] m_tokensCurrent;
	delete [] m_tokensNext;
	delete [] m_activeTokenCurrent;
	delete [] m_activeTokenNext;
	delete [] m_historyItems;
	delete [] m_iHistoryItemsAuxBuffer;
	//delete m_lmLookAhead;
	// word-graph generation?
	m_bInitialized = false;
}
// print the hash-contents (debugging)
void DynamicDecoderX::printHashContents() { }
// end utterance
void DynamicDecoderX::endUtterance() { }
// pruning (token based)
void DynamicDecoderX::pruningOriginal(){}
// destructor
DynamicDecoderX::~DynamicDecoderX(){}
// return the active lm-states at the current time (lm-state in active tokens)
void DynamicDecoderX::getActiveLMStates(map<int,bool> &mLMState) { }
// keeps the best history item for each unique word-sequence (auxiliar method)
void DynamicDecoderX::keepBestHistoryItem(int iHistoryItem) { }
 
// compute the load factor of the hash table containing unque word sequences
float DynamicDecoderX::computeLoadFactorHashWordSequences(int *iBucketsUsed, int *iCollisions) {return 0.0; }
// show hash-occupation information (debugging)
void DynamicDecoderX::printHashsStats() {}
// marks unused history items as available (lattice generation)
void DynamicDecoderX::historyItemGarbageCollectionLattice(bool bRecycleHistoryItems, bool bRecycleWGTokens) { }
// build a hypothesis lattice for the utterance


};	// end-of-namespace

