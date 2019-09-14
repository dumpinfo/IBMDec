/*---------------------------------------------------------------------------------------------*
 * Copyright (C) 2012 Daniel Bolaños - www.bltek.com - Boulder Language Technologies           *
 *                                                                                             *
 * www.bavieca.org is the website of the Bavieca Speech Recognition Toolkit                    *
 *                                                                                             *
 * Licensed under the Apache License, Version 2.0 (the "License");                             *
 * you may not use this file except in compliance with the License.                            *
 * You may obtain a copy of the License at                                                     *
 *                                                                                             *
 *         http://www.apache.org/licenses/LICENSE-2.0                                          *
 *                                                                                             *
 * Unless required by applicable law or agreed to in writing, software                         *
 * distributed under the License is distributed on an "AS IS" BASIS,                           *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                    *
 * See the License for the specific language governing permissions and                         *
 * limitations under the License.                                                              *
 *---------------------------------------------------------------------------------------------*/


#ifndef DYNAMICDECODERX_H
#define DYNAMICDECODERX_H

#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>
#include <vector>
using namespace std;

#include <list>

#include "Global.h"
#include "Vector.h"
#include "Matrix.h"


#include "DynamicNetworkX.h"
//----------------------------------- 
#include "Global.h"
#include "HMMManager.h"
#include "LexiconManager.h"
//-----------------------------------
//#include "HypothesisLattice.h"
#include "LexiconManager.h"

namespace Bavieca {

class BestPath;
class HMMManager;
class PhoneSet;
class LMLookAhead;
class LMFSM;
class LMManager;

#define NUMBER_BINS_HISTOGRAM							50
#define NUMBER_BINS_HISTOGRAM_WITHIN_NODE			100

// language model transition (auxiliar structure)
typedef struct {
	int iLMState;				// next lm-state
	float fScoreLM;			// associated score
} LMTransition;

// entry in the hash table that keeps unique word sequences (lattice generation)
typedef struct _WSHashEntry {
	int iTime;					// time frame of last insertion (-1 initially)
	int iLexUnits;				// number of lexical units in the word sequence
	int *iLexUnit;				// lexical units in the word-sequence
	int iNext;					// next table entry (to handle collisions)
} WSHashEntry;

// structure to keep the lexical unit history
typedef struct _HistoryItem {
   int iLexUnitPron;			// lexical unit (including alternative pronunciations)
   int iEndFrame;				// ending frame (the start frame can  be obtained from the previous lexical unit)
   float fScore;				// global score (accumulated across the whole utterance) includes lm and am scores
   int iPrev;					// previous history item
   int iActive;				// last time the item was active (part of an active token's history) (garbage collection)
   int iWGToken;				// best N paths that arrive at this word history item (each of them has a backpointer)
} HistoryItem;

typedef vector<HistoryItem*> VHistoryItem;
typedef list<HistoryItem*> LHistoryItem;
typedef map<HistoryItem*,bool> MHistoryItem;
//typedef map<HistoryItem*,LNode*> MHistoryItemLNode;

// word-graph token (word-graph generation)
typedef struct _WGToken {
	int iWordSequence;				// index in the hash table containing unique word sequences
	int iLexUnitPron;					// lexical-unit at the time this token was created (in case it was known)
	//add support for this field and remove added code used for debugging
	float fScore;						// path score
	int iHistoryItem;					// index of the previous item in the history
	int iActive;						// last time the token was active (part of an active token's history) (garbage collection)
	int iPrev;							// previous token in the list of available tokens (memory management and garbage collection)	
} WGToken;

typedef list<WGToken*> LWGToken;
typedef vector<WGToken*> VWGToken;
typedef map<WGToken*,bool> MWGToken;

// token
typedef struct {
	float fScore;						// path score
	HMMStateDecoding *state;		// hmm-state
	int iLMState;						// language model state
	int iLexUnitPron;					// lexical unit (including alternative pronunciations)
	int iNode;							// index of the node where the token is
	int iHistoryItem;					// index of the history item
} Token;

// active token
typedef struct {
	int iLMState;						// language model state
	int iToken;							// active token
} ActiveToken;

/**
	@author daniel <dani.bolanos@gmail.com>
*/
class DynamicDecoderX {

	private:
	
		PhoneSet *m_phoneSet;
		HMMManager *m_hmmManager;
		LexiconManager *m_lexiconManager;
		LMFSM *m_lmFSM;
		LMManager *m_lmManager;
		unsigned char m_iNGram;
		DynamicNetworkX *m_dynamicNetwork;
		
		// network properties
		int m_iArcs;
		DArc *m_arcs;
		int m_iNodes;
		DNode *m_nodes;
		
		// pruning parameters
		int m_iMaxActiveNodes;				// maximum number of active arcs
		int m_iMaxActiveNodesWE;			// maximum number of active arcs at word-ends
		int m_iMaxActiveTokensNode;		// maximum number of active tokens within an arc
		float m_fBeamWidthNodes;			// beam width for all arcs
		float m_fBeamWidthNodesWE;			// beam width for all arcs at word-ends
		float m_fBeamWidthTokensNode;		// beam width for all tokens within an arc
		
		// scaling factor
		float m_fLMScalingFactor;
		
		// current time frame
		int m_iTimeCurrent;
		
		bool m_bInitialized;
		
		// tokens
		int m_iTokensMax;						// number of tokens allocated
		Token *m_tokensCurrent;
		Token *m_tokensNext;
		int m_iTokensNext;
		
		// active nodes
		DNode **m_nodesActiveCurrent;
		DNode **m_nodesActiveNext;
		int m_iNodesActiveCurrent;
		int m_iNodesActiveNext;
		int m_iNodesActiveCurrentMax;
		int m_iNodesActiveNextMax;
		
		// active tokens
		int m_iTokensNodeMax;
		ActiveToken *m_activeTokenCurrent;
		ActiveToken *m_activeTokenNext;
		int m_iActiveTokenTables;
		int m_iActiveTokenMax;
		
		float m_fScoreBest;
		float m_fScoreBestWE;
		
		// history item management
		unsigned int m_iHistoryItems;					// number of history items allocated
		HistoryItem *m_historyItems;					// history items allocated
		int m_iHistoryItemBegSentence;				// initial history item
		int m_iHistoryItemAvailable;					//	next history item available to be used
		int m_iTimeGarbageCollectionLast;			// last time frame the garbage collection was run
		// auxiliar arrays (used for token expansion)
		int *m_iHistoryItemsAuxBuffer;
		int *m_iHistoryItemsAux;
		int m_iHistoryItemsAuxSize;
		
		// word-graph generation
		bool m_bLatticeGeneration;				// whether to generate a word-graph	
		int m_iMaxWordSequencesState;			// maximum number of word sequences arriving at any state	
		map<int,pair<float,int> > m_mWSHistoryItem;
		list<int> m_lHistoryItem;
				
		// hash table for hashing word sequences
		unsigned int m_iWSHashBuckets;		// # buckets in the hash table
		unsigned int m_iWSHashEntries;		//	# entries in the hash table (#entries = #buckets + #unique word sequences)
		WSHashEntry *m_wshashEntries;							// hash table
		int m_iWSHashEntryCollisionAvailable;		// next available entry in the hash table to store collisions
		
		// utterance information
		int m_iFeatureVectorsUtterance;	
		
		// unknown lexical unit
		int m_iLexUnitPronUnknown;
		
		// language model look-ahead
		LMLookAhead *m_lmLookAhead;
		
		// create a new token
		inline int newToken() {	
		
			assert(m_iTokensNext < m_iTokensMax);
			int iToken = m_iTokensNext;	
			++m_iTokensNext;
		
			return iToken;
		}
		
		inline int newActiveTokenTable() {
		
			assert(m_iActiveTokenTables+m_iTokensNodeMax < m_iActiveTokenMax);
			m_iActiveTokenTables += m_iTokensNodeMax; 
		
			return m_iActiveTokenTables-m_iTokensNodeMax;
		}
		
		// return an unused history item
		inline int newHistoryItem() {
		
			if (m_iHistoryItemAvailable == -1) {
				if (m_bLatticeGeneration == false) {
					historyItemGarbageCollection();
				} else {
					historyItemGarbageCollectionLattice(true,false);
				}
				assert(m_iHistoryItemAvailable != -1);
			}
			
			int iReturn = m_iHistoryItemAvailable;
			m_historyItems[m_iHistoryItemAvailable].iWGToken = -1;
			m_iHistoryItemAvailable = m_historyItems[m_iHistoryItemAvailable].iPrev; 
		
			return iReturn;
		}
		
		// return whether a history item is inactive (debugging)
		inline bool inactive(int iHistoryItem) {
		
			int iAux = m_iHistoryItemAvailable;
			while(iAux != -1) {
				if (iAux == iHistoryItem) {
					return true;
				}
				iAux = m_historyItems[iAux].iPrev;
			}
			
			return false;
		}
		
		// swap token tables (after processing a feature vector)
		inline void swapTokenTables() {
				
			// swap token tables
			Token *tokenAux = m_tokensCurrent;
			m_tokensCurrent = m_tokensNext;
			m_tokensNext = tokenAux;
			m_iTokensNext = 0;
			
			// swap active-token tables
			ActiveToken *activeTokenAux = m_activeTokenNext;
			m_activeTokenNext = m_activeTokenCurrent;
			m_activeTokenCurrent = activeTokenAux;
			m_iActiveTokenTables = 0;
		}		
		
		// root-node expansion
		void expandRoot(VectorBase<float> &vFeatureVector);
		
		// regular expansion
		void expand(VectorBase<float> &vFeatureVector, int t);
		
		// expand a series of tokens to a hmm-state
		void expandToHMM(DNode *node, DArc *arcNext, VectorBase<float> &vFeatureVector, int t);
		
		// expand a series of tokens to a hmm-state after obsering a new word
		void expandToHMMNewWord(DNode *node, DArc *arcNext, LexUnit *lexUnit, LMTransition *lmTransition, 
			VectorBase<float> &vFeatureVector, int t);	
		
		// pruning (token based)
		void pruningOriginal();
		
		// pruning (token based)
		void pruning();		
		
		// marks unused history items as available
		void historyItemGarbageCollection();
				
		// marks unused history items as available (lattice generation)
		void historyItemGarbageCollectionLattice(bool bRecycleHistoryItems, bool bRecycleWGTokens);
		
		// get the destination lexical units
		void getDestinationMonophoneLexUnits(DNode *node, VLexUnit &vLexUnitDest);
		
		// prune active tokens that are not in the top-N within a node 
		void pruneExtraTokens(DNode *node);
		
 
 
		// compute the load factor of the hash table containing unque word sequences (debugging)
		float computeLoadFactorHashWordSequences(int *iBucketsUsed, int *iCollisions);		
		
		// shows hash-occupation information (debugging)
		void printHashsStats();
		
		// print the hash-contents 
		void printHashContents();	
		
		// keeps the best history item for each unique word-sequence (auxiliar method)
		void keepBestHistoryItem(int iHistoryItem);	

	public:
		
		// constructor
		DynamicDecoderX(PhoneSet *phoneSet, HMMManager *hmmManager, 
			LexiconManager *lexiconManager, LMManager *lmManager, float fLMScalingFactor, 
			DynamicNetworkX *dynamicNetwork, int iMaxActiveNodes, 
			int iMaxActiveNodesWE, int iMaxActiveTokensNode, float fBeamWidthNodes, 
			float fBeamWidthNodesWE, float fBeamWidthTokensNode, bool bWordGraphGeneration, 
			int iMaxWordSequencesState);

		// destructor
		~DynamicDecoderX();
		
		// initialization
		void initialize();
		
		// uninitialize
		void uninitialize();
		
		// begin utterance
		void beginUtterance();
		
		// end utterance
		void endUtterance();
		
		// process input feature vectors
		void process(MatrixBase<float> &mFeatures);	
		
		// return the BestPath
		BestPath *getBestPath();
		
		// return a hypothesis lattice for the utterance
		//HypothesisLattice *getHypothesisLattice();
		
		// return the active lm-states at the current time (lm-state in active tokens)
		void getActiveLMStates(map<int,bool> &mLMState);	
		
};

};	// end-of-namespace

#endif
