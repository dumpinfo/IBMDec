#ifndef VECTORBASE_H
#define VECTORBASE_H

#include <assert.h>
#include <string.h>
#include <iostream>
using namespace std;

#undef min
#undef max
#undef abs

#include "Global.h"

namespace Bavieca {

template<typename Real>
class MatrixBase;

 
template<typename Real>
class VectorBase {

	protected: 
	
		unsigned int m_iDim;			// dimensionality
		Real *m_rData;					// data
		
		VectorBase() {
		}
		
		VectorBase(unsigned int iDim) {
			m_iDim = iDim;
			m_rData = NULL;
		}

	public:
	
		// return the dimensionality
		unsigned int getDim() const {
		
			return m_iDim;
		}
		
		// return the data
		Real *getData() {
		
			return m_rData;
		}
		
		// returns the given element as a r-value
		Real& operator() (unsigned int i) {
		
			assert((i>=0) && (i < m_iDim));
			return m_rData[i];
		}	
				
		// returns the given element as a r-value
		Real operator() (unsigned int i) const {
		
			assert((i>=0) && (i < m_iDim));
			return m_rData[i];
		}	
		
		// invert vector elements
		void invertElements() {
			
			for(unsigned int i=0 ; i<m_iDim ; ++i) {
				m_rData[i] = (Real)(1.0/m_rData[i]);
			}
		}
		
		// return the minimum value
		Real min() {
		
			Real rMin = m_rData[0];
			for(unsigned int i=1 ; i<m_iDim ; ++i) {
				if (m_rData[i] < rMin) {
					rMin = m_rData[i]; 
				}
			}
			
			return rMin;
		}	
				
		// return the maximum value
		Real max() {
		
			Real rMax = m_rData[0];
			for(unsigned int i=1 ; i<m_iDim ; ++i) {
				if (m_rData[i] > rMax) {
					rMax = m_rData[i]; 
				}
			}
			
			return rMax;
		}	
				
		// add another vector
		void add(VectorBase<Real> &v) {
			
			add(1.0,v);
		}
		
		// add another vector
		template<typename Real2>
		void add(VectorBase<Real2> &v) {
			
			add(1.0,v);
		}
		
		// add another vector multiplied by a constant
		void add(Real r, const VectorBase<Real> &v) {
			
			assert(m_iDim == v.getDim());
		 
			for(unsigned int i=0 ; i<m_iDim ; ++i) {
				m_rData[i] += r*v(i);
			}
		}
		
		// add another vector multiplied by a constant
		template<typename Real2>
		void add(Real r, const VectorBase<Real2> &v) {
			
			assert(m_iDim == v.getDim());
		 
			for(unsigned int i=0 ; i<m_iDim ; ++i) {
				m_rData[i] += (Real)(r*v(i));
			}
		}
		
		// add another vector (squaring elements) multiplied by a constant
		void addSquare(Real r, const VectorBase<Real> &v) {
			
			assert(m_iDim == v.getDim());
		 
			for(unsigned int i=0 ; i<m_iDim ; ++i) {
				m_rData[i] += r*v(i)*v(i);
			}
		}
		
		// add another vector (squaring elements) multiplied by a constant
		template<typename Real2>
		void addSquare(Real r, const VectorBase<Real2> &v) {
			
			assert(m_iDim == v.getDim());
		 
			for(unsigned int i=0 ; i<m_iDim ; ++i) {
				m_rData[i] += r*v(i)*v(i);
			}
		}
		
		// add the rows of the given matrix
		void addRows(MatrixBase<Real> &m) {
		
			assert(m_iDim == m.getCols());
			for(unsigned int i=0 ; i<m.getRows() ; ++i) {
				for(unsigned int j=0 ; j<m_iDim ; ++j) {
					m_rData[j] += m(i,j);
				}
			}	
		}
		
		// add the rows of the given matrix
		template<typename Real2>
		void addRows(MatrixBase<Real2> &m) {
		
			assert(m_iDim == m.getCols());
			for(unsigned int i=0 ; i<m.getRows() ; ++i) {
				for(unsigned int j=0 ; j<m_iDim ; ++j) {
					m_rData[j] += m(i,j);
				}
			}	
		}
		
		// add the square of the rows of the given matrix
		template<typename Real2>
		void addRowsSquare(MatrixBase<Real2> &m) {
		
			assert(m_iDim == m.getCols());
			for(unsigned int i=0 ; i<m.getRows() ; ++i) {
				for(unsigned int j=0 ; j<m_iDim ; ++j) {
					m_rData[j] += m(i,j)*m(i,j);
				}
			}	
		}
		
		// set elements to zero
		void zero() {
		
			for(unsigned int i=0 ; i < m_iDim ; ++i) {
				m_rData[i] = 0.0;
			}
		}
		
		// scale elements by the given factor
		void scale(Real rFactor) {
		
			for(unsigned int i=0 ; i < m_iDim ; ++i) {
				m_rData[i] *= rFactor;
			}
		}
		
		// copy from memory
		void copy(Real *rData, unsigned int iDim);
		
		// copy from memory
		//template<typename Real2>
		//void copy(Real2 *rData, unsigned int iDim);
		
		// copy from another vector (same length and type)
		void copy(const VectorBase<Real> &v) {
		
			assert(m_iDim == v.getDim());
			memcpy(m_rData,v.m_rData,m_iDim*sizeof(Real));	
		}
		
		// copy from another vector (same length and different type)
		template<typename Real2>
		void copy(const VectorBase<Real2> &v) {
		
			assert(m_iDim == v.getDim());
			for(unsigned int i=0 ; i < m_iDim ; ++i) {
				m_rData[i] = (Real)v(i);
			}		
		}
		
		// copy from another vector (diff length, same type)
		void copy(const VectorBase<Real> &v, unsigned int iOffset, unsigned int iLen) {
		
			assert((iOffset >= 0) && (iOffset < m_iDim));
			assert((iLen > 0) && (iOffset+iLen <= m_iDim));
			for(unsigned int i=0 ; i < iLen ; ++i) {
				m_rData[iOffset+i] = (Real)v(i);
			}		
		}
		
      void mul(Real r) {
      for(unsigned int i=0 ; i < m_iDim ; ++i) {
            m_rData[i] *= r;
            }
       }
     void sqrt();        
     // set to a vector multiplied by a constant
     void mul(Real r, const VectorBase<Real> &v) { 
       for(unsigned int i=0 ; i<m_iDim ; ++i) {
          m_rData[i] = r*v(i);
       }
     }
 
};

};	// end-of-namespace

#endif
