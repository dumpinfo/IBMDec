#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <assert.h>

#include "LogMessage.h"
#include "VectorBase.h"

namespace Bavieca {

 
template<class Real>
class Vector : public VectorBase<Real> {
	public:
		// constructor
		Vector(int iDim) : VectorBase<Real>(iDim) {
			allocate();
			this->zero();
		}
		
		// destructor
		~Vector() {
			if (this->m_rData) {
				deallocate();	
			}
		}
			
		// allocate memory
		void allocate() {
			this->m_rData = new Real[this->m_iDim];
		 
		}
		
		// deallocate memory
		void deallocate() {
			delete [] this->m_rData;
			this->m_rData = NULL;
		}
		
 
};

};	// end-of-namespace

#endif
