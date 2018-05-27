#include <igl/read_triangle_mesh.h>
#include <igl/viewer/Viewer.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>

//#define DEBUG

/*** insert any necessary libigl headers here ***/
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>

//#include <igl/png/render_to_png.h>

using namespace std;
using namespace Eigen;
using Viewer = igl::viewer::Viewer;

Viewer viewer;

// vertex array, #V x3
Eigen::MatrixXd V;

// face array, #F x3
Eigen::MatrixXi F;
Eigen::MatrixXd Colors; //For colors

// UV coordinates, #V x2
Eigen::MatrixXd UV;

bool showingUV = false;
bool freeBoundary = false;
bool showtextures = true;
int nbPointsBordure = 2;

double threshold = 0.1;
double Colormin = 0;
double Colormax = 150;
bool uniform = false;
bool cotan = false;
bool LSCM = false;
bool ARAP = false;
std::string meshName;
std::string NameFile = "";

int showAngleDistortion = 0;
double TextureResolution = 10;
igl::viewer::ViewerCore temp3D;
igl::viewer::ViewerCore temp2D;

//My functions
double calculateSquaredDistance(int i, int j);
void calculateColorsDistortion();

void Redraw()
{
	viewer.data.clear();

	if (!showingUV)
	{
		viewer.data.set_mesh(V, F);
		viewer.data.set_face_based(false);

		//There is a parametization
		if (UV.size() != 0)
		{
			if (showtextures)
			{
				viewer.data.set_uv(TextureResolution * UV);
				viewer.data.show_texture = true;
			}
			else
			{
#ifdef DEBUG
				std::cout << "Try to set new colors : " << endl;
				for (int i = 0; i < Colors.rows(); i++)
				{
					//std::cout << "Face N°" << i << " is " << Colors(i, 0) << " " << Colors(i, 1) << " " << Colors(i, 2) << endl;
				}
#endif
				viewer.data.set_uv(TextureResolution * UV);
				viewer.data.set_colors(Colors);
			}
		}
	}
	else
	{
		viewer.data.show_texture = false;
		viewer.data.set_mesh(UV, F);
		//There is a parametization
		if (UV.size() != 0)
		{
			if (showtextures)
			{
				viewer.data.set_uv(TextureResolution * UV);
				viewer.data.show_texture = true;
			}
			else
			{
#ifdef DEBUG
				std::cout << "Try to set new colors : " << endl;
				for (int i = 0; i < Colors.rows(); i++)
				{
					//std::cout << "Face N°" << i << " is " << Colors(i, 0) << " " << Colors(i, 1) << " " << Colors(i, 2) << endl;
				}
#endif
				viewer.data.set_uv(TextureResolution * UV);
				viewer.data.set_colors(Colors);
			}
		}
	}
}

bool callback_mouse_move(Viewer &viewer, int mouse_x, int mouse_y)
{
	if (showingUV)
		viewer.mouse_mode = igl::viewer::Viewer::MouseMode::Translation;
	return false;
}

static void computeSurfaceGradientMatrix(SparseMatrix<double> &D1, SparseMatrix<double> &D2)
{
	MatrixXd F1, F2, F3;
	SparseMatrix<double> DD, Dx, Dy, Dz;

	igl::local_basis(V, F, F1, F2, F3);
	igl::grad(V, F, DD);

	Dx = DD.topLeftCorner(F.rows(), V.rows());
	Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
	Dz = DD.bottomRightCorner(F.rows(), V.rows());

	D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
	D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;
}

static inline void SSVD2x2(const Eigen::Matrix2d &J, Eigen::Matrix2d &U, Eigen::Matrix2d &S, Eigen::Matrix2d &V)
{
	double e = (J(0) + J(3)) * 0.5;
	double f = (J(0) - J(3)) * 0.5;
	double g = (J(1) + J(2)) * 0.5;
	double h = (J(1) - J(2)) * 0.5;
	double q = sqrt((e * e) + (h * h));
	double r = sqrt((f * f) + (g * g));
	double a1 = atan2(g, f);
	double a2 = atan2(h, e);
	double rho = (a2 - a1) * 0.5;
	double phi = (a2 + a1) * 0.5;

	S(0) = q + r;
	S(1) = 0;
	S(2) = 0;
	S(3) = q - r;

	double c = cos(phi);
	double s = sin(phi);
	U(0) = c;
	U(1) = s;
	U(2) = -s;
	U(3) = c;

	c = cos(rho);
	s = sin(rho);
	V(0) = c;
	V(1) = -s;
	V(2) = s;
	V(3) = c;
}

void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system
	// Hint: The matrix C should contain only one non-zero element per row and d should contain the positions in the correct order.

	//Resize to the correct structure
	C.conservativeResize(2 * indices.rows(), 2 * V.rows());
	d.conservativeResize(2 * indices.rows(), 1);

	//Nettoyage
	for (int i = 0; i < C.rows(); i++)
	{
		for (int j = 0; j < 2 * V.rows(); j++)
		{
			//C.row(i)[j] = 0; //Not for sparse matrix
		}
	}
	for (int i = 0; i < d.rows(); i++)
	{
		d[i] = 0;
	}

#ifdef DEBUG
	std::cout << " Created matrix - C : " << endl;
	for (int i = 0; i < C.rows(); i++)
	{
		for (int j = 0; j < 2 * V.rows(); j++)
		{
			//std::cout << C.row(i)[j] << " "; //Not for sparse matrix
		}
		std::cout << endl;
	}
	std::cout << "> End of matrix " << endl;

	std::cout << " Created vector - d : " << endl;
	for (int i = 0; i < d.rows(); i++)
	{
		std::cout << d[i] << " ";
	}
	std::cout << "\n> End of vector " << endl;
#endif

	//C = Eigen::MatrixXd::Zero(indices.rows(), 2*V.rows());
	//C = Eigen::MatrixXd::Zero(indices.rows(), 2*V.rows());

	int tailleV = V.rows();
	int tailleB = positions.rows();

	//Construct of C
	for (int i = 0; i < indices.rows(); i++)
	{
		C.insert(i, indices[i]) = 1;
		C.insert(tailleB + i, tailleV + indices[i]) = 1;

		//C.row(i)[indices.row(i)] = 1;
		//C.row(tailleB+i)[tailleV+indices.row(i)] = 1;
	}

	//Construct d as all U constraints, and then all V constraints
	for (int i = 0; i < positions.rows(); i++)
	{
		d[i] = positions.row(i)[0];
		d[tailleB + i] = positions.row(i)[1];
	}

	//Sanity check
	assert(C.rows() == d.rows());

#ifdef DEBUG
	std::cout << " System Overview : " << endl;
	Eigen::MatrixXd printable4 = MatrixXd(C);
	for (int i = 0; i < C.rows(); i++)
	{

		for (int j = 0; j < C.row(0).size(); j++)
		{
			std::cout << printable4.row(i)[j] << " ";
		}
		std::cout << " * u? = ";
		std::cout << d[i] << endl;
	}
	std::cout << "> End of System Overview" << endl;

#endif
}

//compute the distance between two vertices of indices i and j in V
double calculateSquaredDistance(int i, int j)
{
	//Sanity check
	assert(i < V.rows());
	assert(j < V.rows());

	//Compute
	Eigen::RowVector3d curDistance = V.row(i) - V.row(j);
	double dist = curDistance[0] * curDistance[0] + curDistance[1] * curDistance[1] + curDistance[2] * curDistance[2];

	return dist;
}

void computeParameterization(int type)
{
	F = viewer.data.F;
	V = viewer.data.V;

	VectorXi fixed_UV_indices;
	MatrixXd fixed_UV_positions;

	Eigen::SparseMatrix<double> A;
	VectorXd b;
	Eigen::SparseMatrix<double> C;
	VectorXd d;
	// Find the indices of the boundary vertices of the mesh and put them in fixed_UV_indices
	igl::boundary_loop(F, fixed_UV_indices);

#ifdef DEBUG
	std::cout << " Boundaries indices : " << endl;
	for (int i = 0; i < fixed_UV_indices.rows(); i++)
	{
		std::cout << fixed_UV_indices.row(i) << " ";
	}
	std::cout << " \n end of list " << endl;
#endif

	igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

	if (!freeBoundary)
	{
	// The boundary vertices should be fixed to positions on the unit disc. Find these position and
	// save them in the #V x 2 matrix fixed_UV_position.
	//igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

#ifdef DEBUG
		std::cout << " Disque positions : " << endl;
		for (int i = 0; i < fixed_UV_positions.rows(); i++)
		{
			std::cout << "Point N°" << i << " : " << fixed_UV_positions.row(i)[0] << " " << fixed_UV_positions.row(i)[1] << endl;
		}
		std::cout << " \n end of list " << endl;
#endif
	}
	else
	{
		// Fix two UV vertices. This should be done in an intelligent way. Hint: The two fixed vertices should be the two most distant one on the mesh.

		/* OLD METHOD
		double resMax = 0;
		double resTmp = 0;
		int indicePointA = 0;
		int indicePointB = 0;

		//Get the two most extreme points of the mesh
		for (int i = 0; i < V.rows(); i++)
		{
			for (int j = 0; j < V.rows(); j++)
			{
				resTmp = calculateSquaredDistance(i, j);
				if (resMax < resTmp)
				{
					indicePointA = i;
					indicePointB = j;
					resMax = resTmp;
				}
			}
		}

		//We put it in the good place.
		fixed_UV_positions.conservativeResize(2, 3);
		fixed_UV_positions.row(0) = V.row(indicePointA);
		fixed_UV_positions.row(1) = V.row(indicePointB);
		// UNSURE ? fixed_UV_positions.conservativeResize(2, 2);
#ifdef DEBUG
		std::cout << "Max distance found : " << endl;
		std::cout << " Squared distance : " << resMax;
		std::cout << " between A(" << V.row(indicePointA)[0] << "," << V.row(indicePointA)[1] << "," << V.row(indicePointA)[2];
		std::cout << ") and B(" << V.row(indicePointB)[0] << "," << V.row(indicePointB)[1] << "," << V.row(indicePointB)[2] << ")" << endl;
#endif

		*/

		/* TO TEST, instead of distance calculus */
		//igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

		int nbOfNewPoints = 0;
		for (int i = 0; i < nbPointsBordure && i < fixed_UV_positions.rows(); i++)
		{
			int indicePointCurr = round(i * fixed_UV_positions.rows() / nbPointsBordure); //0, and 1/n mov each time

			// Fix two points on the boundary
			fixed_UV_positions.row(nbOfNewPoints) << fixed_UV_positions.row(indicePointCurr);
			fixed_UV_indices[nbOfNewPoints] = fixed_UV_indices[indicePointCurr];

			nbOfNewPoints++;
		}

		//We reformat originals for the positions and indices.
		fixed_UV_positions.conservativeResize(nbOfNewPoints, 3);
		fixed_UV_indices.conservativeResize(nbOfNewPoints, 1);

		/* OLD VERSION BUT WORKING 
		int indicePointA = 0; //fixed_UV_indices[0];
		int indicePointB = round(fixed_UV_positions.rows()/2); //fixed_UV_indices[round(fixed_UV_indices.rows()/2)];

		// Fix two points on the boundary
		fixed_UV_positions.row(0) << fixed_UV_positions.row(indicePointA);
		fixed_UV_positions.row(1) << fixed_UV_positions.row(indicePointB);
		fixed_UV_indices[0] = fixed_UV_indices[indicePointA];
		fixed_UV_indices[1] = fixed_UV_indices[indicePointB];

		//We reformat originals for the positions and indices.
		fixed_UV_positions.conservativeResize(2, 3);
		fixed_UV_indices.conservativeResize(2, 1);
*/

		/**/

#ifdef DEBUG
		std::cout << " Extreme positions : " << endl;
		for (int i = 0; i < fixed_UV_positions.rows(); i++)
		{
			std::cout << "Point N°" << i << " : " << fixed_UV_positions.row(i)[0] << " " << fixed_UV_positions.row(i)[1] << endl;
		}
		std::cout << " \n end of list " << endl;
#endif
	}

	ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, C, d);

	// Find the linear system for the parameterization (1- Tutte, 2- Harmonic, 3- LSCM, 4- ARAP)
	// and put it in the matrix A.

	// The dimensions of A should be 2#V x 2#V.
	A.conservativeResize(2 * V.rows(), 2 * V.rows());
	b.conservativeResize(2 * V.rows(), 1);

	//Cleaning
	for (int i = 0; i < b.rows(); i++)
	{
		b[i] = 0;
	}

	/* CREATION OF A and B depending on the type of parametization */

	if (type == '1')
	{
		uniform = true;
		cotan = false;
		LSCM = false;
		ARAP = false;

		Eigen::SparseMatrix<double> Adj;
		// Add your code for computing uniform Laplacian for Tutte parameterization
		// Hint: use the adjacency matrix of the mesh
		igl::adjacency_matrix(F, Adj);

		// sum each row
		Eigen::SparseVector<double> Asum;
		igl::sum(Adj, 1, Asum);

		// Convert row sums into diagonal of sparse matrix
		Eigen::SparseMatrix<double> Adiag;
		igl::diag(Asum, Adiag);

		// Build uniform laplacian
		Eigen::SparseMatrix<double> U;
		U = Adj - Adiag;

		int tailleV = V.rows();

//NEW VERSION
		//Construction of A  : See explanations on report
		Eigen::SparseMatrix<double> UpperPart;
		Eigen::SparseMatrix<double> LowerPart;

		Eigen::SparseMatrix<double> vide;
		vide.conservativeResize(U.rows(), U.rows());

		//Concat small parts
		igl::cat(2, U, vide, UpperPart);
		igl::cat(2, vide, U, LowerPart);

		//Construction of AT*A from Uniform Laplacian
		igl::cat(1, UpperPart, LowerPart, A);

/* OLD VERSION
		//Construction of AT*A from Uniform Laplacian
		for (int i = 0; i < U.rows(); i++)
		{
			for (int j = 0; j < U.row(0).size(); j++)
			{
				A.insert(i, j) = U.coeff(i, j);
				// For sparse matrix : double value = matrix.coeff(iRow, iCol);
				A.insert(tailleV + i, tailleV + j) = U.coeff(i, j);
			}
		}
*/
#ifdef DEBUG
		Eigen::MatrixXd printable1 = MatrixXd(A);
		std::cout << " A : adjacency matrix : has " << A.nonZeros() << " elements." << endl;
		for (int i = 0; i < printable1.rows(); i++)
		{
			for (int j = 0; j < printable1.row(0).size(); j++)
			{
				std::cout << printable1.row(i)[j] << " ";
			}
			std::cout << endl;
		}
#endif

		//Construction of b, all 0, size 2*#V
		//Nothing to do, already full of 0
	}

	if (type == '2')
	{
		uniform = false;
		cotan = true;
		LSCM = false;
		ARAP = false;

		Eigen::SparseMatrix<double> L;
		// Add your code for computing cotangent Laplacian for Harmonic parameterization
		// Use can use a function "cotmatrix" from libIGL, but ~~~~***READ THE DOCUMENTATION***~~~~
		igl::cotmatrix(V, F, L);

#ifdef DEBUG
		Eigen::MatrixXd printable2 = MatrixXd(L);
		std::cout << " L : cotan matrix : has " << L.nonZeros() << " elements." << endl;
		for (int i = 0; i < printable2.rows(); i++)
		{
			for (int j = 0; j < printable2.row(0).size(); j++)
			{
				std::cout << printable2.row(i)[j] << " ";
			}
			std::cout << endl;
		}
#endif

		int tailleV = V.rows();

//NEW VERSION
		//Construction of A  : See explanations on report
		Eigen::SparseMatrix<double> UpperPart;
		Eigen::SparseMatrix<double> LowerPart;

		Eigen::SparseMatrix<double> vide;
		vide.conservativeResize(L.rows(), L.rows());

		//Concat small parts
		igl::cat(2, L, vide, UpperPart);
		igl::cat(2, vide, L, LowerPart);

		//Construction of AT*A from Uniform Laplacian
		igl::cat(1, UpperPart, LowerPart, A);

/* OLD VERSION

		//Construction of AT*A, thanks to Dirichlet minimization of energy.
		for (int i = 0; i < L.rows(); i++)
		{
			for (int j = 0; j < L.row(0).size(); j++)
			{
				A.insert(i, j) = L.coeff(i, j);
				// For sparse matrix : double value = matrix.coeff(iRow, iCol);
				A.insert(tailleV + i, tailleV + j) = L.coeff(i, j);
			}
		}
*/
#ifdef DEBUG
		Eigen::MatrixXd printable3 = MatrixXd(A);
		std::cout << " At*A : superior left quart of the matrix : has " << A.nonZeros() << " elements." << endl;
		for (int i = 0; i < printable3.rows(); i++)
		{
			for (int j = 0; j < printable3.row(0).size(); j++)
			{
				std::cout << printable3.row(i)[j] << " ";
			}
			std::cout << endl;
		}
#endif

		//Construction of b, all 0, size 2*#V
		//Nothing to do, already full of 0
	}

	if (type == '3')
	{
		uniform = false;
		cotan = false;
		LSCM = true;
		ARAP = false;

		// Add your code for computing the system for LSCM parameterization
		// Note that the libIGL implementation is different than what taught in the tutorial! Do not rely on it!!
		Eigen::SparseMatrix<double> Du;
		Eigen::SparseMatrix<double> Dv;
		Eigen::SparseMatrix<double> DuTrans;
		Eigen::SparseMatrix<double> DvTrans;
		Eigen::VectorXd AreasTMP;
		Eigen::SparseMatrix<double> Areas;
		Eigen::SparseMatrix<double> aTMP;
		Eigen::SparseMatrix<double> bTMP;
		Eigen::SparseMatrix<double> cTMP;
		Eigen::SparseMatrix<double> dTMP;

		//We calculate the double area for each triangle.
		igl::doublearea(V, F, AreasTMP);

		//We transform it in a diagonal matrix
		Areas.conservativeResize(AreasTMP.rows(), AreasTMP.rows());
		// assert(Areas.rows()==Areas.cols()) //useful when it was done without a temp
		for (int i = 0; i < Areas.rows(); i++)
		{
			Areas.insert(i, i) = AreasTMP(i);
			//Areas.insert(i,0) = 0;
		}

		//We calcul the gradient for all triangles
		computeSurfaceGradientMatrix(Du, Dv);
		DuTrans = Du.transpose();
		DvTrans = Dv.transpose();

#ifdef DEBUG
		Eigen::MatrixXd printable11 = MatrixXd(Du);
		Eigen::MatrixXd printable12 = MatrixXd(Dv);

		std::cout << "Gradient Matrix " << endl;
		std::cout << "Du : " << Du.rows() << " lignes et " << Du.cols() << " colonnes." << endl;
		for (int i = 0; i < printable11.rows(); i++)
		{
			for (int j = 0; j < printable11.row(0).size(); j++)
			{
				std::cout << printable11.row(i)[j] << " ";
			}
			std::cout << endl;
		}
		std::cout << endl;

		std::cout << "Dv : " << Dv.rows() << " lignes et " << Dv.cols() << " colonnes." << endl;
		for (int i = 0; i < printable12.rows(); i++)
		{
			for (int j = 0; j < printable12.row(0).size(); j++)
			{
				std::cout << printable12.row(i)[j] << " ";
			}
			std::cout << endl;
		}
#endif

		//We calculate the parts of A
		aTMP = DuTrans * Areas * Du;
		bTMP = DvTrans * Areas * Dv;
		cTMP = DvTrans * Areas * Du;
		dTMP = DuTrans * Areas * Dv;

		//Construction of A  : See explanations on report
		Eigen::SparseMatrix<double> UpperPart;
		Eigen::SparseMatrix<double> LowerPart;
		Eigen::SparseMatrix<double> RightUp = cTMP - dTMP;
		Eigen::SparseMatrix<double> LeftUp = aTMP + bTMP;
		Eigen::SparseMatrix<double> RightDown = aTMP + bTMP;
		Eigen::SparseMatrix<double> LeftDown = dTMP - cTMP;

		//Concat small parts
		igl::cat(2, LeftUp, RightUp, UpperPart);
		igl::cat(2, LeftDown, RightDown, LowerPart);

		Eigen::SparseMatrix<double> result;
		igl::cat(1, UpperPart, LowerPart, result);

		A = result;
	}

	if (type == '4')
	{
		uniform = false;
		cotan = false;
		LSCM = false;
		ARAP = true;

		// Add your code for computing ARAP system and right-hand side
		// Implement a function that computes the local step first
		// Then construct the matrix with the given rotation matrices
	}

	// Construct the system as discussed in class and the assignment sheet
	// Use igl::cat to concatenate matrices
	// Use Eigen::SparseLU to solve the system. Refer to tutorial 3 for more detail

	// Data structure necessary
	int tailleB = fixed_UV_positions.rows();
	int tailleV = V.rows();

	Eigen::SparseMatrix<double> LeftSide;
	//Eigen::SparseMatrix<double> RightSide; // Error ! Don't want a sparse matrix as a right side !
	Eigen::VectorXd RightSide;
	Eigen::SparseMatrix<double> TmpUp;
	Eigen::SparseMatrix<double> TmpDown;
	Eigen::SparseMatrix<double> Ct = SparseMatrix<double>(C.transpose());

	Eigen::SparseMatrix<double> vide;

	Eigen::SparseMatrix<double> bSparse;
	bSparse = b.sparseView();

	Eigen::SparseMatrix<double> dSparse;
	dSparse = d.sparseView();

	vide.conservativeResize(2 * tailleB, 2 * tailleB);

	// Construction of the linear system
	igl::cat(2, A, Ct, TmpUp);
	igl::cat(2, C, vide, TmpDown);
	igl::cat(1, TmpUp, TmpDown, LeftSide);
	igl::cat(1, b, d, RightSide);

#ifdef DEBUG

	std::cout << " \n\n\n\n\n Linear System Overview " << endl;

	std::cout << "Matrice d : " << endl;
	Eigen::MatrixXd printable10 = MatrixXd(d);
	for (int i = 0; i < printable10.rows(); i++)
	{
		for (int j = 0; j < printable10.row(0).size(); j++)
		{
			std::cout << printable10.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice b : " << endl;
	Eigen::MatrixXd printable9 = MatrixXd(b);
	for (int i = 0; i < printable9.rows(); i++)
	{
		for (int j = 0; j < printable9.row(0).size(); j++)
		{
			std::cout << printable9.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice vide : " << endl;
	Eigen::MatrixXd printable8 = MatrixXd(vide);
	for (int i = 0; i < printable8.rows(); i++)
	{
		for (int j = 0; j < printable8.row(0).size(); j++)
		{
			std::cout << printable8.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice C : " << endl;
	Eigen::MatrixXd printable7 = MatrixXd(C);
	for (int i = 0; i < printable7.rows(); i++)
	{
		for (int j = 0; j < printable7.row(0).size(); j++)
		{
			std::cout << printable7.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice At.A : " << endl;
	Eigen::MatrixXd printable6 = MatrixXd(A);
	for (int i = 0; i < printable6.rows(); i++)
	{
		for (int j = 0; j < printable6.row(0).size(); j++)
		{
			std::cout << printable6.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "System : " << endl;
	Eigen::MatrixXd printable4 = MatrixXd(LeftSide);
	Eigen::MatrixXd printable5 = MatrixXd(RightSide);

	assert(LeftSide.rows() == RightSide.rows());

	for (int i = 0; i < printable4.rows(); i++)
	{
		for (int j = 0; j < printable4.row(0).size(); j++)
		{
			std::cout << printable4.row(i)[j] << " ";
		}
		std::cout << " = ";
		for (int j = 0; j < printable5.row(0).size(); j++)
		{
			std::cout << printable5.row(i)[j] << " ";
		}
		std::cout << endl;
	}

	std::cout << " LeftSide Non Zeros : " << LeftSide.nonZeros() << " elements." << endl;
	std::cout << " RightSide Non Zeros : " << RightSide.nonZeros() << " elements." << endl;
	std::cout << endl;
	std::cout << " Lignes à droites : " << RightSide.rows() << " lignes." << endl;
	std::cout << " Cols à droites : " << RightSide.cols() << " colonnes." << endl;
	std::cout << endl;
	std::cout << " Dimension de At.A : " << A.rows() << " lignes * " << A.cols() << " colonnes." << endl;
	std::cout << " Dimension de C : " << C.rows() << " lignes * " << C.cols() << " colonnes." << endl;
	std::cout << " Dimension de Ct : " << Ct.rows() << " lignes * " << Ct.cols() << " colonnes." << endl;
	std::cout << " Dimension de vide : " << vide.rows() << " lignes * " << vide.cols() << " colonnes." << endl;
	std::cout << endl;
	std::cout << " Lignes à gauche : " << LeftSide.rows() << " lignes." << endl;
	std::cout << " Cols à gauche : " << LeftSide.cols() << " colonnes." << endl;
	std::cout << endl;
	std::cout << " Dimension de b : " << b.rows() << " lignes * " << b.cols() << " colonnes." << endl;
	std::cout << " Dimension de d : " << d.rows() << " lignes * " << d.cols() << " colonnes." << endl;

#endif

	// Solve the linear system.
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

	// Compute the ordering permutation vector from the structural pattern of A
	solver.analyzePattern(LeftSide);
	// Compute the numerical factorization
	solver.factorize(LeftSide);

	//Use the factors to solve the linear system
	solver.compute(LeftSide);

	Eigen::VectorXd result;
	result = solver.solve(RightSide);

	// The solver will output a vector
	UV.resize(V.rows(), 2);

	//Block of size (p,q), starting at (i,j) <==> matrix.block(i,j,p,q);
	UV.col(0) = result.block(0, 0, tailleV, 1);
	UV.col(1) = result.block(tailleV, 0, tailleV, 1);

	//Create colors field
	Colors.conservativeResize(F.rows(), 3);

	calculateColorsDistortion();

	/*
	for (int i = 0; i < Colors.rows(); i++)
	{
		Colors(i, 0) =  1;
		double teinte = 1-(i / (double)(Colors.rows())); // 1 = blanc, 0 = rouge
		Colors(i, 1) = teinte;
		Colors(i, 2) = teinte;
	}
	*/
}

void calculateColorsDistortion()
{

#ifdef DEBUG
	std::cout << " Cacul of the distortion of each face for colorization" << endl;
#endif

	Eigen::SparseMatrix<double> Dx;
	Eigen::SparseMatrix<double> Dy;

	Eigen::SparseMatrix<double> bigJac;

	Eigen::SparseMatrix<double> UpLeft;
	Eigen::SparseMatrix<double> UpRight;
	Eigen::SparseMatrix<double> DownLeft;
	Eigen::SparseMatrix<double> DownRight;

	//Get Dx/Dy : ?
	computeSurfaceGradientMatrix(Dx, Dy);

	//Get U and V : ?
	Eigen::VectorXd uVector = UV.col(0);
	Eigen::VectorXd vVector = UV.col(1);
	Eigen::SparseMatrix<double> uVectorTMP;
	Eigen::SparseMatrix<double> vVectorTMP;
	uVectorTMP = uVector.sparseView();
	vVectorTMP = vVector.sparseView();

	std::cout << Dx.rows() << " " << Dx.cols() << " " << uVector.rows() << " " << uVector.cols() << endl;

	//We calcule a big "Jacobian" matrix, for all vertices
	UpLeft = (Dx * uVectorTMP);	// #Fx#V * #V*1 = #F * 1
	UpRight = (Dy * uVectorTMP);   // #Fx#V * #V*1 = #F * 1
	DownLeft = (Dx * vVectorTMP);  // #Fx#V * #V*1 = #F * 1
	DownRight = (Dy * vVectorTMP); //#Fx#V * #V*1 = #F * 1

	Eigen::SparseMatrix<double> tmpUp;
	Eigen::SparseMatrix<double> tmpDown;

	//Create the big Jack Matrix
	igl::cat(2, UpLeft, UpRight, tmpUp);
	igl::cat(2, DownLeft, DownRight, tmpDown);
	igl::cat(1, tmpUp, tmpDown, bigJac);

#ifdef DEBUG
	std::cout << " Jacobian of each points calculated " << endl;
#endif

	/*
#ifdef DEBUG

	std::cout << " \n\n\n\n\n Distortion calculus (before) " << endl;

	std::cout << "Matrice Dx : " << endl;
	Eigen::MatrixXd printable20 = MatrixXd(Dx);
	for (int i = 0; i < printable20.rows(); i++)
	{
		for (int j = 0; j < printable20.row(0).size(); j++)
		{
			std::cout << printable20.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice dy : " << endl;
	Eigen::MatrixXd printable21 = MatrixXd(Dy);
	for (int i = 0; i < printable21.rows(); i++)
	{
		for (int j = 0; j < printable21.row(0).size(); j++)
		{
			std::cout << printable21.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << " Dimension de dx : " << Dx.rows() << " lignes * " << Dx.cols() << " colonnes." << endl;
	std::cout << " Dimension de dy : " << Dy.rows() << " lignes * " << Dy.cols() << " colonnes." << endl;
	std::cout << endl;

#endif

	//Convertion to perform operations
	Eigen::MatrixXd DxTMP = Eigen::MatrixXd(Dx);
	Eigen::MatrixXd DyTMP = Eigen::MatrixXd(Dy);
	
	//Create a Gradient per face and not more per vertex
	DxTMP = DxTMP.rowwise().sum();//(DxTMP.rowwise() > 0).count();
	DyTMP = DyTMP.rowwise().sum();

	//Put it back in sparseMatrix for next computation
	Dx.resize(F.rows(), 1);
	Dy.resize(F.rows(), 1);
	Dx = DxTMP.sparseView();
	Dy = DyTMP.sparseView();

#ifdef DEBUG

	std::cout << " \n\n\n\n\n Distortion calculus (after) " << endl;

	std::cout << "Matrice Dx : " << endl;
	Eigen::MatrixXd printable16 = MatrixXd(Dx);
	for (int i = 0; i < printable16.rows(); i++)
	{
		for (int j = 0; j < printable16.row(0).size(); j++)
		{
			std::cout << printable16.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << "Matrice dy : " << endl;
	Eigen::MatrixXd printable17 = MatrixXd(Dy);
	for (int i = 0; i < printable17.rows(); i++)
	{
		for (int j = 0; j < printable17.row(0).size(); j++)
		{
			std::cout << printable17.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;

	std::cout << " Dimension de dx : " << Dx.rows() << " lignes * " << Dx.cols() << " colonnes." << endl;
	std::cout << " Dimension de dy : " << Dy.rows() << " lignes * " << Dy.cols() << " colonnes." << endl;
	std::cout << endl;

#endif
*/
	int tailleV = V.rows();

	Eigen::MatrixXd distorstionPerFace;
	distorstionPerFace.resize(F.rows(), 1);
	distorstionPerFace.setZero();

	//For each face, we have to calculate the distortion.
	for (int i = 0; i < F.rows(); i++)
	{

		//Represent the 4 values of the jacobian of the current face
		double DxU = 0;
		double DxV = 0;
		double DyU = 0;
		double DyV = 0;

		for (int j = 0; j < 3; j++)
		{

			int indicePtcur = F.row(i)[j];

			//We calculate the jacobian for each point of the face
			DxU += bigJac.coeff(indicePtcur, 0);
			DxV += bigJac.coeff(tailleV + indicePtcur, 0);
			DyU += bigJac.coeff(indicePtcur, 1);
			DyV += bigJac.coeff(tailleV + indicePtcur, 1);
		}

		//Take the mean
		DxU /= 3;
		DxV /= 3;
		DyU /= 3;
		DyV /= 3;

		//REMINDER :	static inline void SSVD2x2(const Eigen::Matrix2d &J, Eigen::Matrix2d &U, Eigen::Matrix2d &S, Eigen::Matrix2d &V)
		//Create the jacobian
		Eigen::Matrix2d J;
		J.setZero();

		//Set values of the jacobian
		J.row(0)[0] = DxU;
		J.row(0)[1] = DyU;
		J.row(1)[0] = DxV;
		J.row(1)[1] = DyV;

		//Necessary outputs for SVD
		Eigen::Matrix2d Utmp;
		Eigen::Matrix2d Stmp;
		Eigen::Matrix2d Vtmp;

		// Extract Singular Value via SSVD (Signed Singular Value Decomposition)
		SSVD2x2(J, Utmp, Stmp, Vtmp);

#ifdef DEBUG

		std::cout << " \n\n\n\n\n Signed Singular Values computation " << endl;

		std::cout << "Matrice U, S, V : " << endl;
		assert(Utmp.rows() == Stmp.rows());
		assert(Utmp.rows() == Vtmp.rows());

		for (int i = 0; i < Utmp.rows(); i++)
		{
			for (int j = 0; j < Utmp.row(0).size(); j++)
			{
				std::cout << Utmp.row(i)[j] << " ";
			}
			std::cout << "| ";
			for (int j = 0; j < Stmp.row(0).size(); j++)
			{
				std::cout << Stmp.row(i)[j] << " ";
			}
			std::cout << "| ";
			for (int j = 0; j < Vtmp.row(0).size(); j++)
			{
				std::cout << Vtmp.row(i)[j] << " ";
			}
			std::cout << endl;
		}
		std::cout << endl;

		std::cout << " Dimension de U : " << Utmp.rows() << " lignes * " << Utmp.cols() << " colonnes." << endl;
		std::cout << " Dimension de S : " << Stmp.rows() << " lignes * " << Stmp.cols() << " colonnes." << endl;
		std::cout << " Dimension de V : " << Vtmp.rows() << " lignes * " << Vtmp.cols() << " colonnes." << endl;

		std::cout << endl;

#endif

		double sigmaUn = Stmp.row(0)[0];
		double sigmaDeux = Stmp.row(1)[1];

		//Compute the Distorsion, depending on the kind of distorsion computed
		// Reminder : v.ngui->addVariable("0 = Angle Distortion / 1 = Length distortion / 2 = Area distortion", showAngleDistortion);
		if (showAngleDistortion == 0)
		{ // ANGLE
			distorstionPerFace.row(i)[0] = ((sigmaUn - sigmaDeux) * (sigmaUn - sigmaDeux));
		}
		else if (showAngleDistortion == 1)
		{ // LENGTH
			distorstionPerFace.row(i)[0] = ((sigmaUn - 1) * (sigmaUn - 1) + (sigmaDeux - 1) * (sigmaDeux - 1));
		}
		else if (showAngleDistortion == 2)
		{ // AREA
			distorstionPerFace.row(i)[0] = ((sigmaUn * sigmaDeux - 1) * (sigmaUn * sigmaDeux - 1));
		}
	}

	// Calculate color code
	double maxCoef = distorstionPerFace.maxCoeff();
	double minCoef = distorstionPerFace.minCoeff();

#ifdef DEBUG

	std::cout << " \n Distorsion calculated :  " << endl;

	for (int i = 0; i < distorstionPerFace.rows(); i++)
	{
		for (int j = 0; j < distorstionPerFace.row(0).size(); j++)
		{
			std::cout << distorstionPerFace.row(i)[j] << " ";
		}
		std::cout << endl;
	}
	std::cout << endl;
	std::cout << "Max : " << maxCoef << " Min : " << minCoef << endl;

#endif

	/* OLD VERSION
	if(threshold<minCoef){
		//Do what I do
		threshold = minCoef;
	}

	assert(Colors.rows() == distorstionPerFace.rows());
	for (int i = 0; i < Colors.rows(); i++)
	{
		Colors(i, 0) =  1;
		if(threshold > distorstionPerFace.row(i)[0]){
			//Under the threeshold = White
			Colors(i, 1) =  1;
			Colors(i, 2) =  1;
		} else {
			double teinte = (distorstionPerFace.row(i)[0]-threshold)/((double)(maxCoef));
			Colors(i, 1) = teinte;
			Colors(i, 2) = teinte;
		}
	}
*/

	assert(Colors.rows() == distorstionPerFace.rows());
	for (int i = 0; i < Colors.rows(); i++)
	{
		Colors(i, 0) = 1;
		if (distorstionPerFace.row(i)[0] < Colormin)
		{
			//Under the threeshold = White
			Colors(i, 1) = 1;
			Colors(i, 2) = 1;
#ifdef DEBUG
			std::cout << " \n Distorsion current :  " << distorstionPerFace.row(i)[0] << " Color current : "
					  << "1" << endl;
#endif
		}
		else if (distorstionPerFace.row(i)[0] > Colormax)
		{
			//Above the max = RED
			Colors(i, 1) = 0;
			Colors(i, 2) = 0;
#ifdef DEBUG
			std::cout << " \n Distorsion current :  " << distorstionPerFace.row(i)[0] << " Color current : "
					  << "0" << endl;
#endif
		}
		else
		{ // Between min and max values, we color "slowly"
			double teinte = 1 - ((distorstionPerFace.row(i)[0] - Colormin) / ((double)(Colormax)));
			Colors(i, 1) = teinte;
			Colors(i, 2) = teinte;
#ifdef DEBUG
			std::cout << " \n Distorsion current :  " << distorstionPerFace.row(i)[0] << " Color current : " << teinte << endl;
#endif
		}
	}
}

std::string stringConstructorPicture()
{
	std::string namePicture;

	namePicture += meshName + "_";
	if (uniform)
	{
		namePicture += "uni_";
	}
	else if (cotan)
	{
		namePicture += "cotan_";
	}
	else if (LSCM)
	{
		namePicture += "lscm_";
	}
	else if (ARAP)
	{
		namePicture += "arap_";
	}

	if(!freeBoundary){
		namePicture += "circle_";	
	} else {
		namePicture += "freeP" + std::to_string(nbPointsBordure) + "_";
	}

	if(showtextures){
		namePicture += "texture_";	
	} else {
		namePicture += "colorsMAX" + std::to_string((int)Colormax) + "MIN" + std::to_string((int)Colormin) + "_";
		namePicture += "Distortion_T" + std::to_string(showAngleDistortion/1) + "_";
	}

	if(!showingUV){
		namePicture += "mesh";	
	} else {
		namePicture += "plan";
	}
		namePicture += ".png";

	return namePicture;
}

bool callback_key_pressed(Viewer &viewer, unsigned char key, int modifiers)
{
	switch (key)
	{
	case '1':
		computeParameterization(key);
		break;
	case '2':
		computeParameterization(key);
		break;
	case '3':
		computeParameterization(key);
		break;
	case '4':
		computeParameterization(key);
		break;
	case '5':
		// Add your code for detecting and displaying flipped triangles in the
		// UV domain here
		break;
	case '6':
		showtextures = !showtextures;
		Redraw();
		break;
	case '9':
		//Construct the file name and save it.
		//igl::png::render_to_png(stringConstructorPicture(),800,600);
		//    igl::png::render_to_png(padnum.str(),width,height);
		NameFile = stringConstructorPicture();
		std::cout << stringConstructorPicture() << endl;
		break;
	case '+':
		TextureResolution /= 2;
		break;
	case '-':
		TextureResolution *= 2;
		break;
	case ' ': // space bar -  switches view between mesh and parameterization
		if (showingUV)
		{
			temp2D = viewer.core;
			viewer.core = temp3D;
			showingUV = false;
		}
		else
		{
			if (UV.rows() > 0)
			{
				temp3D = viewer.core;
				viewer.core = temp2D;
				showingUV = true;
			}
			else
			{
				std::cout << "ERROR ! No valid parameterization\n";
			}
		}
		break;
	}
	Redraw();
	NameFile = stringConstructorPicture();
	return true;
}

void init_core_states()
{
	// save initial viewer core state
	temp3D = viewer.core;
	temp2D = viewer.core;
	temp2D.orthographic = true;
}

bool load_mesh(string filename)
{
	igl::read_triangle_mesh(filename, V, F);
	
	std::size_t found = filename.find_last_of("/\\");
	meshName = filename.substr(found + 1);
	std::size_t found2 = meshName.find_last_of(".");
	meshName = meshName.substr(0,found);
	
	std::cout << "Mesh ouvert : " << meshName << endl;
	
	Redraw();
	viewer.core.align_camera_position(V);
	showingUV = false;

	return true;
}

bool callback_load_mesh(Viewer &viewer, string filename)
{
	load_mesh(filename);
	init_core_states();
	return true;
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "Usage ex4_bin <mesh.off/obj>" << endl;
		load_mesh("../data/cathead.obj");
	}
	else
	{
		// Read points and normals
		load_mesh(argv[1]);
	}

	viewer.callback_init = [&](Viewer &v) {
		// Add widgets to the sidebar.
		v.ngui->addGroup("Parmaterization");
		v.ngui->addVariable("Free boundary", freeBoundary);

		v.ngui->addVariable("Nb of points of the boundary (for non-free)", nbPointsBordure);

		v.ngui->addVariable("ON = Show texture /\n OFF = show distortion", showtextures);
		v.ngui->addVariable("0 = Angle Distortion /\n 1 = Length distortion /\n 2 = Area distortion", showAngleDistortion);

		v.ngui->addVariable("1 - Uniform Laplacian. Bound=OK", uniform);
		v.ngui->addVariable("2 - Cotan Laplacian. Bound=OK", cotan);
		v.ngui->addVariable("3 - LSCM. Bound=OK, NOBOUND=OK", LSCM);
		v.ngui->addVariable("4 - ARAP. Not implemented", ARAP);
		v.ngui->addVariable("Threshold color (outdated)", threshold);

		v.ngui->addVariable("Min scale for color", Colormin);
		v.ngui->addVariable("Max scale for color", Colormax);

		v.ngui->addVariable("Name", NameFile);
		
		// TODO: Add more parameters to tweak here...
		viewer.screen->performLayout();

		init_core_states();

		return false;
	};

	viewer.callback_key_pressed = callback_key_pressed;
	viewer.callback_mouse_move = callback_mouse_move;
	viewer.callback_load_mesh = callback_load_mesh;

	viewer.launch();
}
