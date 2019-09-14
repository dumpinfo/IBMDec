#ifndef MATRIXBASE_H
#define MATRIXBASE_H

using namespace std;

#undef min
#undef max
#undef abs

#include "VectorBase.h"
#include "VectorStatic.h"
#include "Global.h"

#include <string.h>

namespace Bavieca {


typedef enum { yes ,no  } MatrixTransposed;

#define MATRIX_TRANSPOSED				'T'
#define MATRIX_NO_TRANSPOSED			'N'

template<typename Real>
class MatrixStatic;

 
template<typename Real>
class MatrixBase {

	protected:
	
		unsigned int m_iRows;		// number of rows in the matrix
		unsigned int m_iCols;		// number of columns in the matrix
		unsigned int m_iStride;		// number of elements per row (stride >= # columns)
		Real *m_rData;					// matrix data
	
		MatrixBase(unsigned int iRows, unsigned int iCols) {
		
			m_iRows = iRows;
			m_iCols = iCols;
			m_rData = NULL;
		}
		
	public:

		// return the number of rows
		unsigned int getRows() {
		
			return m_iRows;
		}
		
		// return the number of columns
		unsigned int getCols() {
		
			return m_iCols;
		}
		
		// return the stride
		unsigned int getStride() {
		
			return m_iStride;
		}
		
		// return the data
		Real *getData() {
		
			return m_rData;
		}
		
		// return a matrix element as a r-value
		Real& operator()(unsigned int iRow, unsigned int iCol) {
			
			return m_rData[iRow*m_iStride+iCol];
		}
		
		// return a matrix element as a l-value
		Real operator()(unsigned int iRow, unsigned int iCol) const {
			
			return m_rData[iRow*m_iStride+iCol];
		}
		
		// return the number of elements in the matrix
		unsigned int getElements() {
		
			return m_iCols*m_iRows;
		}
		
		// return the matrix size in terms of data allocated
		unsigned int getSize() {
		
			return m_iStride*m_iRows;
		}	
		
		// return a matrix row
		const VectorStatic<Real> getRow(unsigned int iRow) const {
		
			return VectorStatic<Real>(m_rData+iRow*m_iStride,m_iCols);
		}
		
		// return a matrix row
		VectorStatic<Real> getRow(unsigned int iRow) {
		
			return VectorStatic<Real>(m_rData+iRow*m_iStride,m_iCols);
		}
		
		// return a matrix row
		Real *getRowData(unsigned int iRow);
		
		// copy data from another matrix
		void copy(MatrixBase<Real> &m) {
		
			assert((m_iRows == m.getRows()) && (m_iCols == m.getCols()));
			memcpy(m_rData,m.getData(),getSize()*sizeof(Real));
		}
		
 
		
		// set to zero
		void zero();
		
 
		
		// return whether all the elements in the matrix are actual finite numbers  
		bool finite();		
		
 
		
		// compute derivative (iDelta = size of the regression window)
		void delta(MatrixBase<Real> &mDelta, unsigned int iDelta);
		
};

};	// end-of-namespace

#endif
