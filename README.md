
##### TL;DR
This project was linked to a lecture about Shape Modeling. This project presents different parametization methods : we map 3D points in a 2D space, to apply modification (Especially when it's simpler in 2D, like texture mapping, etc.), to get back to 3D space.
<br/>
Mapping 3D meshs to 2D ("flat") space involved stretching, shrinking, ... and so globally, deformations. When we proceed to such process, we try to reduce this deformation. Assuming we try to reduce deformation, the parametization will depend on the way we compute deformation.
<br/>
We will see 4 methods : 
* Spring energy (uniform Laplacian)
* Dirichlet/harmonic energy (cotangent Laplacian)
* Least Squares Conformal Maps (LSCM)
* As-Rigid-As-Possible (ARAP)
<br/>

##### Table of Contents  
[1. Setting up the boundary conditions](#boundary)  <br/>
[1.1. Finding the fixed vertex indices and their positions](#fixedVertices)  <br/>
[1.2. Convert the boundary conditions to linear constraints](#linear)  <br/>
[2. Write the parameterization problem in matrix form and construct the matrix](#matrix)  <br/>
[2.1. Uniform and cotangent Laplacian](#Laplacian)  <br/>
[2.2. LSCM](#LSCM)  <br/>
[2.3. ARAP](#ARAP)  <br/>
[3. Construct the system and display the results](#results)  <br/>
[3.1. Solve and show the parameterization.](#parameterization)  <br/>
[3.2. Visualize the distortion](#distortion)  <br/>

<a name="boundary"/>

#### 1. Setting up the boundary conditions

<a name="fixedVertices"/>

#### 1.1. Creating the constraints

An easy solution to map points in a 2D space is to shrink the whole mesh in a single dot. It's equivalent to a null-function solution. To avoid this problem, we have to fix points of the 3D mesh in the 2D space, as anchors to map the rest of the mesh.
<br/>
A first solution is to find the vertices which constitue the boundary of the mesh (if the mesh has such a "border") and map them to a 2D circle.
<br/>
A second soltion is to find the boundary of the mesh, as previously, but only take a chosen "n" number of vertices, equally spaced, and map them to a 2D circle. In this configuration, we can for example take 2 points, and map them to a line. Idem with 3 points for a triangle, etc.
<br/>
Depending on the parametization, one or both of the solution will be usable.
<br/>

###### Documentation informations
We can use the following function : 
* __igl::boundary loop__
This function construct the list of indices of vertices of the external boundary loop
<br/>
[-> Documentation](https://github.com/libigl/libigl/blob/master/include/igl/boundary_loop.h)


###### Usage
The first idea, create a circle boundary, is straightforward, as the **IGL function** is doing everything.

```C++
  (...)
	if (!freeBoundary)
	{
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);
	}
  (...)
```

**The second idea is to have only 2 constrainted points**. 

* The first approach is to calculate all distances between all points of the mesh (per pair) and keep the two most distants points. But **distants points on the mesh may be not distants points on a 2D surface**. So this idea conduct to "bad" parametization, mainly because we don't consider the topological distance, but the euclidian distance only.

* The second approach, is a variant of the circle boundary : we only **keep some (two for example) extreme points of the circle boundary**. It conduct to much better results with the provided meshs, as the distance is "implicity" calculated as the path **on** the mesh, the topological distance.

Note that, we may want to choose the coordinates in the 2D space of the fixed points to try to **reduce the area distortion** of the triangles of the mesh.
But, in our case, as we can and will change the texture size as needed (+/- buttons) we don't need to scale the boundary to optimize the area distortion. It will only vary from a factor, homogenous for the whole mesh.
It won't change the distortion color either, for the same reason (homogenous for the whole mesh).

Therefore, we can distinguish a **"good" from "bad" parametization**, thanks to the **homogeneousity of the distortion values.**
So, for pictures showing the distortion, if there is areas in red and area in white, the parametization is not adapted.
So, if all areas are approximatly the same color (all light pink) we can consider the parametization as adapated.

```C++
  (...)
	if (freeBoundary)
	{
		igl::map_vertices_to_circle(V, fixed_UV_indices, fixed_UV_positions);

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

    (...)
  }
  (...)
```

###### Result
<div  style="text-align:center">
<p align="center">
<img style="height : 300px;" src="/media/compressed/bunny_CircleBoundary.gif">
<img style="height : 300px;" src="/media/compressed/bunny_CircleBoundary.png">
<br/>Mesh = bunny / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/catHead_CircleBoundary.gif">
<img style="height : 300px;" src="/media/compressed/catHead_CircleBoundary.png">
<br/>Mesh = cat / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/obj_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_texture_plan.png">
<br/>Left : Mesh = cow / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_texture_plan.png">
<br/>Left : Mesh = gargoyle / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/hemisphere_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemisphere_uni_circle_texture_plan.png">
<br/>Left : Mesh = hemisphere / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_texture_plan.png">
<br/>Left : Mesh = hemisphereNC / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/off_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_texture_plan.png">
<br/>Left : Mesh = max / Boundary = circle <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_texture_plan.png">
<br/>Left : Mesh = octo / Boundary = circle <br/>

</p>
</div>

__________________________________


<a name="linear"/>

#### 1.2. Convert the boundary conditions to linear constraints

We have to construct a constraint system, that will be added to the system we're going to solve, to get the new 2D positions of all the mesh's points.

###### Usage

The system will be of the form : 
* x_point_1 = x_forced_2DPosition_point1
* y_point_1 = y_forced_2DPosition_point1
* ...
* x_point_i = x_forced_2DPosition_pointi
* y_point_i = y_forced_2DPosition_pointi

Which can be displayed under a matrix formatting, as C * Positions = D 
<br/>
We create a C matrix, which is not a diagonal matrix, but a sparse matrix, with only 1 where this particular (u,v) has to match with this particular (x,y) values of d.
And we create a D matrix constitued by the 2D fixed coordinates of the boundary.
<br/>
The function works with all cases/types of boundaries conditions, circle or not.

<p align="center">
<img style="height : 500px;" src="/media/compressed/schema_4.jpg">
</p>

```C++
void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double> &C, VectorXd &d)
{
	// Convert the list of fixed indices and their fixed positions to a linear system

	//Resize to the correct structure
	C.conservativeResize(2 * indices.rows(), 2 * V.rows());
	d.conservativeResize(2 * indices.rows(), 1);
  (...)

	int tailleV = V.rows();
	int tailleB = positions.rows();

	//Construct of C
	for (int i = 0; i < indices.rows(); i++)
	{
		C.insert(i, indices[i]) = 1;
		C.insert(tailleB + i, tailleV + indices[i]) = 1;
	}

	//Construct d as all U constraints, and then all V constraints
	for (int i = 0; i < positions.rows(); i++)
	{
		d[i] = positions.row(i)[0];
		d[tailleB + i] = positions.row(i)[1];
	}

  (...)
}
```

###### Result
Exemple of system (cathead, Harmonic uniform)

```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.838304
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.48517
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.122623
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.401705
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.784592
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.988038
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.89542
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.278918
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.201572
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.637168
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.938674
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.545202
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0  * u? = 0.87442
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0  * u? = 0.992453
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  * u? = 0.915769
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  * u? = 0.620013
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  * u? = 0.154211
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.445223
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.960315
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.979474
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.770725
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  * u? = -0.344806
```

__________________________________

<a name="matrix"/>

#### 2. Write the parameterization problem in matrix form and construct the matrix

<a name="Laplacian"/>

#### 2.1. Uniform and cotangent Laplacian

We will solve a system for the parametization with all points.
The left part of system is composed of A - a matrix that will represent the kind of distortion we measure - and C - previously explained and computed, the constraints.
<br/>
First of the possible parametization, we compute the Laplacian (uniform and cotangent), which should be null (right side).
<br/>

###### Documentation informations
We can use the following function : 

* __igl::adjencency matrix()__
Construct the graph of adjacency of a mesh
<br/>
[-> Documentation](https://github.com/libigl/libigl/blob/master/include/igl/adjacency_matrix.h)
* __igl::cotmatrix__
Construct the cotangent matrix of a mesh, equivalent to the discrete Laplacian.
<br/>
[-> Documentation](https://github.com/libigl/libigl/blob/master/include/igl/cotmatrix.h)


For igl::adjencency matrix() : 


**Inputs** :
* F = #Fx3 list of mesh faces

**Outputs** :
* A  = max(F)x max(F) // = #Vx#V ? cotangent matrix, One row i represent the adjacents neighboors of the vertice i


For igl::cotmatrix :


**Inputs** :
* V = #Vx3 : list of mesh vertex positions
* F = #Fx3 (triangles) : list of mesh faces (must be triangles)

**Outputs** :
* L = #V x #V cotangent matrix, each row i corresponding to the vertix N° i


###### Usage
For the uniform Laplacian, we construct the A matrix by soustracting the adjacency matrix to the sum of each row of the same matrix, in diagonal.
As results, when uniform "springs" are selected, all edges will try to keep their lengths. It will try to create equilateral triangles.

```C++
if (type == '1')
	{
  		(...)
		igl::adjacency_matrix(F, Adj);

		// sum each row
		igl::sum(Adj, 1, Asum);

		// Convert row sums into diagonal of sparse matrix
		igl::diag(Asum, Adiag);

		// Build uniform laplacian
		U = Adj - Adiag;

		(...)

		//Construction of AT*A from Uniform Laplacian
		for (int i = 0; i < U.rows(); i++)
		{
			for (int j = 0; j < U.row(0).size(); j++)
			{
				A.insert(i, j) = U.coeff(i, j);
				A.insert(tailleV + i, tailleV + j) = U.coeff(i, j);
			}
		}

	(...)

	}
```

Idem for the cotangaent laplacian : We construct the A matrix by concatenating the cotmatrix twice, in diagonal, to create the A-matrix.
The cotangent Laplacian results will be similar to the uniform Laplacian, but a bit smoother.

```C++
if (type == '2')
	{
		(...)
		igl::cotmatrix(V, F, L);

		//Construction of AT*A, thanks to Dirichlet minimization of energy.
		for (int i = 0; i < L.rows(); i++)
		{
			for (int j = 0; j < L.row(0).size(); j++)
			{
				A.insert(i, j) = L.coeff(i, j);
				A.insert(tailleV + i, tailleV + j) = L.coeff(i, j);
			}
		}
		(...)

	}
```


###### Result
<div  style="text-align:center">
<p align="center">

<img style="height : 300px;" src="/media/compressed/bunny_uni_circle_all_all.gif">
<br/>Mesh = bunny / Boundary = circle / Type : Uniform<br/>

<img style="height : 300px;" src="/media/compressed/bunny_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/bunny_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : Uniform / Colored (Max = 25, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : Uniform / Colored (Max = 25, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX25MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : Uniform / Colored (Max = 25, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cathead_uni_circle_all_all.gif">
<br/>Mesh = cathead / Boundary = circle / Type : Uniform<br/>

<img style="height : 300px;" src="/media/compressed/cathead_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/cathead_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = cat / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = cat / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = cat / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_uni_circle_colorsMAX10MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = cat / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/hemisphere_uni_circle_all_all.gif">
  <br/>Mesh = hemisphere / Boundary = circle / Type : Uniform<br/>
<img style="height : 300px;" src="/media/compressed/hemisphere_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemisphere_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion angle)<br/>
Distortion (angle, circle) = Max : 2.33684 Min : 0.0212852<br/>
<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_uni_circle_colorsMAX5MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/hemisphereNC_uni_circle_all_all.gif">
  <br/>Mesh = hemisphereNC / Boundary = circle / Type : Uniform<br/>
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphereNC / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = hemisphereNCeNC / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = hemisphereNCeNC / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = hemisphereNCeNC / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/octo_uni_circle_all_all.gif">
  <br/>Mesh = octo / Boundary = circle / Type : Uniform<br/>
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = Octo / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = Octo_cut2 / Boundary = circle / Type : Uniform / Colored (Max = 150, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = Octo_cut2 / Boundary = circle / Type : Uniform / Colored (Max = 150, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = Octo_cut2 / Boundary = circle / Type : Uniform / Colored (Max = 150, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/cow_uni_circle_all_all.gif">
<br/>Mesh = cow / Boundary = circle / Type : Uniform<br/>
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/cowobj_uni_circle_colorsMAX01MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_colorsMAX0MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : Uniform / Colored (Max = 0.1, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_colorsMAX10MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : Uniform / Colored (Max = 10, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_uni_circle_colorsMAX5MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion area)<br/>


  <img style="height : 300px;" src="/media/compressed/gargo_uni_circle_all_all.gif">
  <br/>Mesh = gargo / Boundary = circle / Type : Uniform<br/>
  <img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_texture_mesh.png">
  <img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_texture_plan.png">
<br/>Left & right: Mesh = gargo / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX001MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = gargoyle / Boundary = circle / Type : Uniform / Colored (Max = 0.01, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX5MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = gargoyle / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX5MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = gargoyle / Boundary = circle / Type : Uniform / Colored (Max = 5, Min = 0, distortion area)<br/>


<img style="height : 300px;" src="/media/compressed/max_uni_circle_all_all.gif">
<br/>Mesh = max / Boundary = circle / Type : Uniform<br/>
<img style="height : 300px;" src="/media/compressed/off_uni_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_texture_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : Uniform / Textured <br/>

<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX0MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX0MIN0_Distortion_T0_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : Uniform / Colored (Max = 0.0001, Min = 0, distortion angle)<br/>
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX10MIN0_Distortion_T1_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : Uniform / Colored (Max = 10, Min = 0, distortion length)<br/>
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX10MIN0_Distortion_T2_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX10MIN0_Distortion_T2_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : Uniform / Colored (Max = 10, Min = 0, distortion area)<br/>

</p>
</div>


______________________________________________________

<div  style="text-align:center">
<p align="center">

<img style="height : 300px;" src="/media/compressed/bunny_cotan_circle_all_all.gif">
<br/>Mesh = bunny / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : Cotan / Textured <br/>
Distortion (angle, circle) = Max : 345.252 Min : 0.122168 <br/>

<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX150MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX150MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX150MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion angle)
<br/>Center: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion length)
<br/>Right: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cathead_cotan_circle_all_all.gif">
<br/>Mesh = cathead / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/cathead_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/cathead_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = cathead / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/o_cotan_circle_colorsMAX1MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_cotan_circle_colorsMAX1MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_cotan_circle_colorsMAX3MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cathead / Boundary = circle / Type : Cotan / Colored (Max = 1, Min = 0, distortion angle)
<br/>Center: Mesh = cathead / Boundary = circle / Type : Cotan / Colored (Max = 1, Min = 0, distortion length)
<br/>Right: Mesh = cathead / Boundary = circle / Type : Cotan / Colored (Max = 3, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphere_cotan_circle_all_all.gif">
<br/>Mesh = hemisphere / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : Cotan / Textured <br/>
Distortion (angle, circle) = Max : 2.33684 Min : 0.0212852<br/>

<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_colorsMAX1MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_colorsMAX10MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphere / Boundary = circle / Type : Cotan / Colored (Max = 1, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphere / Boundary = circle / Type : Cotan / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = hemisphere / Boundary = circle / Type : Cotan / Colored (Max = 10, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphereNC_cotan_circle_all_all.gif">
<br/>Mesh = hemisphereNC / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/hemispherNC_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphereNC / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_cotan_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_cotan_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_cotan_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphereNC / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphereNC / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = hemisphereNC / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/octo_cotan_circle_all_all.gif">
<br/>Mesh = octo / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/Octo_cut2_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = octo / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_cotan_circle_colorsMAX25MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_cotan_circle_colorsMAX25MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_cotan_circle_colorsMAX25MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = octo / Boundary = circle / Type : Cotan / Colored (Max = 25, Min = 0, distortion angle)
<br/>Center: Mesh = octo / Boundary = circle / Type : Cotan / Colored (Max = 25, Min = 0, distortion length)
<br/>Right: Mesh = octo / Boundary = circle / Type : Cotan / Colored (Max = 25, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/cow_cotan_circle_all_all.gif">
<br/>Mesh = cow / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/obj_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/cowobj_cotan_circle_colorsMAX001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_cotan_circle_colorsMAX0MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_cotan_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cow / Boundary = circle / Type : Cotan / Colored (Max = 0.01, Min = 0, distortion angle)
<br/>Center: Mesh = cow / Boundary = circle / Type : Cotan / Colored (Max = 0.01, Min = 0, distortion length)
<br/>Right: Mesh = cow / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/gargo_cotan_circle_all_all.gif">
<br/>Mesh = gargo / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/gargoyle__cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = gargo / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/gargoyle__cotan_circle_colorsMAX0001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__cotan_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/gargoyle__cotan_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = gargo / Boundary = circle / Type : Cotan / Colored (Max = 0.001, Min = 0, distortion angle)
<br/>Center: Mesh = gargo / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = gargo / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion area)<br/>

  <img style="height : 300px;" src="/media/compressed/max_cotan_circle_all_all.gif">
<br/>Mesh = max / Boundary = circle / Type : cotan<br/>
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : Cotan / Textured <br/>

<img style="height : 300px;" src="/media/compressed/maxoff_cotan_circle_colorsMAX00001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = max / Boundary = circle / Type : Cotan / Colored (Max = 0.001, Min = 0, distortion angle)
<br/>Center: Mesh = max / Boundary = circle / Type : Cotan / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = max / Boundary = circle / Type : Cotan / Colored (Max = 5, Min = 0, distortion area)<br/>
</p>
</div>
<br/><br/>

______________________________________________________

<br/>

<a name="LSCM"/>

#### 2.2. LSCM

<br/>
We can also use a Least Square Conformal Map to measure distortion. It is a conformal measure, and so it will try to preserve angles at the expense of scale and length conservation.
<br/>
We will use the surface gradient, and weight them with triangle areas.

###### Documentation informations

We can use the following function : 
* __igl::double area()__
Calculate the area of each triangle input * 2 
[-> Documentation](https://github.com/libigl/libigl/blob/master/include/igl/doublearea.h)
* __igl::cat()__
Already defined.

**Inputs** :
* V = #V x 3, list of mesh vertex positions
* F = #F x 3, list of mesh faces (because triangles)

**Outputs** :
* dblA = #F, list of triangle areas *2

###### Usage
We can decompose from the expression of the LSCM as a system to solve, with a (u v) vector on the right - of the left side - of the system.

Here are the calculus : 

<p align="center">
<img style="height : 500px;" src="/media/compressed/schema_3.jpg">
</p>

Then we just need to compute it from the double area and the gradient matrix.
```C++
	if (type == '3')
	{
		// Code for computing the system for LSCM parameterization
		(...)

		// We calculate the double area for each triangle.
		igl::doublearea(V, F, AreasTMP);

		// We transform it in a diagonal matrix
		(...)

		//We calcul the gradient for all triangles
		computeSurfaceGradientMatrix(Du, Dv);
		DuTrans = Du.transpose();
		DvTrans = Dv.transpose();

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

		igl::cat(1, UpperPart, LowerPart, result);

		A = result;
		(...)
	}
```

###### Result
<div  style="text-align:center">
<p align="center">

<img style="height : 300px;" src="/media/compressed/bunny_LSCM_circle_all_all.gif">
<br/> Mesh = bunny / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/bunny_LSCM_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/bunny_LSCM_circle_texture_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/off_lscm_circle_colorsMAX150MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_lscm_circle_colorsMAX150MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_lscm_circle_colorsMAX150MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion angle)
<br/>Center: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion length)
<br/>Right: Mesh = bunny / Boundary = circle / Type : Cotan / Colored (Max = 150, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/bunny_LSCM_2points_all_all.gif">
<br/>Right : Mesh = bunny / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/bunny_LSCM_2points_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/bunny_LSCM_2points_texture_plan.png">
<br/>Left & right: Mesh = bunny / Boundary = 2 Points / Type : LSCM / Textured <br/>
<br/>Distortion for 2 dots : Max : 246.76 Min : 0.0994094<br/>

<img style="height : 300px;" src="/media/compressed/off_lscm_freeP2_colorsMAX150MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_lscm_freeP2_colorsMAX150MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/off_lscm_freeP2_colorsMAX150MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = bunny / Boundary = 2 Points / Type : Cotan / Colored (Max = 150, Min = 0, distortion angle)
<br/>Center: Mesh = bunny / Boundary = 2 Points / Type : Cotan / Colored (Max = 150, Min = 0, distortion length)
<br/>Right: Mesh = bunny / Boundary = 2 Points / Type : Cotan / Colored (Max = 150, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cathead_LSCM_circle_all_all.gif">
<br/> Mesh = cathead / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/cathead_LSCM_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/cathead_LSCM_circle_texture_plan.png">
<br/>Left & right: Mesh = cathead / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/o_lscm_circle_colorsMAX3MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_circle_colorsMAX3MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_circle_colorsMAX3MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cathead / Boundary = circle / Type : LSCM / Colored (Max = 3, Min = 0, distortion angle)
<br/>Center: Mesh = cathead / Boundary = circle / Type : LSCM / Colored (Max = 3, Min = 0, distortion length)
<br/>Right: Mesh = cathead / Boundary = circle / Type : LSCM / Colored (Max = 3, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cathead_LSCM_2points_all_all.gif">
<br/>Right : Mesh = cathead / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/o_lscm_freeP2_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = cathead / Boundary = 2 Points / Type : LSCM / Textured <br/>
<br/>Distortion for 2 dots : Max : 0.895511 Min : 8.95423e-05<br/>

<img style="height : 300px;" src="/media/compressed/o_lscm_freeP2_colorsMAX3MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_freeP2_colorsMAX3MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_freeP2_colorsMAX3MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cathead / Boundary = 2 Points / Type : LSCM / Colored (Max = 3, Min = 0, distortion angle)
<br/>Center: Mesh = cathead / Boundary = 2 Points / Type : LSCM / Colored (Max = 3, Min = 0, distortion length)
<br/>Right: Mesh = cathead / Boundary = 2 Points / Type : LSCM / Colored (Max = 3, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphere_LSCM_circle_all_all.gif">
<br/> Mesh = hemisphere / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/hemispher_lscm_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispher_lscm_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphere / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphere / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = hemisphere / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphere_LSCM_2points_all_all.gif">
<br/>Right : Mesh = hemisphere / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/hemispher_lscm_freeP2_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = hemisphere / Boundary = 2 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispher_lscm_freeP2_colorsMAX1MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_freeP2_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_lscm_freeP2_colorsMAX10MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphere / Boundary = 2 Points / Type : LSCM / Colored (Max = 1, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphere / Boundary = 2 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = hemisphere / Boundary = 2 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphereNC_LSCM_circle_all_all.gif">
<br/> Mesh = hemisphereNC / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_circle_texture_plan.png">
<br/>Left & right: Mesh = hemisphereNC / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphereNC / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphereNC / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = hemisphereNC / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/hemisphereNC_LSCM_2points_all_all.gif">
<br/>Right : Mesh = hemisphereNC / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_freeP2_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = hemisphereNC / Boundary = 2 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_freeP2_colorsMAX5MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_freeP2_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_lscm_freeP2_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = hemisphereNC / Boundary = 2 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphereNC / Boundary = 2 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = hemisphereNC / Boundary = 2 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>


<img style="height : 300px;" src="/media/compressed/octo_LSCM_circle_all_all.gif">
<br/> Mesh = Octo / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_circle_texture_plan.png">
<br/>Left & right: Mesh = Octo / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_circle_colorsMAX40MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_circle_colorsMAX40MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_circle_colorsMAX40MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = Octo / Boundary = circle / Type : LSCM / Colored (Max = 40, Min = 0, distortion angle)
<br/>Center: Mesh = Octo / Boundary = circle / Type : LSCM / Colored (Max = 40, Min = 0, distortion length)
<br/>Right: Mesh = Octo / Boundary = circle / Type : LSCM / Colored (Max = 40, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/octo_LSCM_8Points_all_all.gif">
<br/>Right : Mesh = Octo / Boundary = 8 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP2_texture_mesh.png">
<img style="height : 150px;" src="/media/compressed/Octo_cut2_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = Octo / Boundary = 2 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP8_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP8_texture_plan.png">
<br/>Left & right: Mesh = Octo / Boundary = 8 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP8_colorsMAX25MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP8_colorsMAX25MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_lscm_freeP8_colorsMAX25MIN0_Distortion_T3_mesh.png">
<br/>Left: Mesh = Octo / Boundary = 8 Points / Type : LSCM / Colored (Max = 25, Min = 0, distortion angle)
<br/>Center: Mesh = Octo / Boundary = 8 Points / Type : LSCM / Colored (Max = 25, Min = 0, distortion length)
<br/>Right: Mesh = Octo / Boundary = 8 Points / Type : LSCM / Colored (Max = 25, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cow_LSCM_circle_all_all.gif">
<br/> Mesh = cow / Boundary = circle / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/obj_lscm_circle_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_circle_texture_plan.png">
<br/>Left & right: Mesh = cow / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/cowobj_lscm_circle_colorsMAX001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cow / Boundary = circle / Type : LSCM / Colored (Max = 0.01, Min = 0, distortion angle)
<br/>Center: Mesh = cow / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = cow / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 300px;" src="/media/compressed/cow_LSCM_8points_all_all.gif">
<br/>Right : Mesh = cow / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP2_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = cow / Boundary = 2 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP8_texture_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP8_texture_plan.png">
<br/>Left & right: Mesh = cow / Boundary = 8 Points / Type : LSCM / Textured <br/>

<img style="height : 300px;" src="/media/compressed/cowobj_lscm_freeP8_colorsMAX0001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP8_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/obj_lscm_freeP8_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = cow / Boundary = 8 Points / Type : LSCM / Colored (Max = 0.001, Min = 0, distortion angle)
<br/>Center: Mesh = cow / Boundary = 8 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = cow / Boundary = 8 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 400px;" src="/media/compressed/gargo_LSCM_circle_all_all.gif">
<br/> Mesh = gargo / Boundary = circle / Type : LSCM<br/>

<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_circle_texture_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__cotan_circle_texture_plan.png">
<br/>Left & right: Mesh = gargo / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_circle_colorsMAX0MIN0_Distortion_T0_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_circle_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = gargo / Boundary = circle / Type : LSCM / Colored (Max = 0.01, Min = 0, distortion angle)
<br/>Center: Mesh = gargo / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = gargo / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 400px;" src="/media/compressed/gargo_LSCM_2Points_all_all.gif">
<br/>Right : Mesh = gargo / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_freeP4_texture_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_freeP4_texture_plan.png">
<br/>Left & right: Mesh = gargo / Boundary = 4 Points / Type : LSCM / Textured <br/>

<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_freeP4_colorsMAX10MIN0_Distortion_T0_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_freeP4_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 400px;" src="/media/compressed/gargoyle__lscm_freeP4_colorsMAX10MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = gargo / Boundary = 4 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion angle)
<br/>Center: Mesh = gargo / Boundary = 4 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = gargo / Boundary = 4 Points / Type : LSCM / Colored (Max = 10, Min = 0, distortion area)<br/>

<img style="height : 400px;" src="/media/compressed/max_LSCM_circle_all_all.gif">
<br/> Mesh = max / Boundary = circle / Type : LSCM<br/>

<img style="height : 400px;" src="/media/compressed/off_lscm_circle_texture_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_circle_texture_plan.png">
<br/>Left & right: Mesh = max / Boundary = circle / Type : LSCM / Textured <br/>

<img style="height : 400px;" src="/media/compressed/maxoff_lscm_circle_colorsMAX00001MIN0_Distortion_T0_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_circle_colorsMAX10MIN0_Distortion_T1_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_circle_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = max / Boundary = circle / Type : LSCM / Colored (Max = 0.0001, Min = 0, distortion angle)
<br/>Center: Mesh = max / Boundary = circle / Type : LSCM / Colored (Max = 10, Min = 0, distortion length)
<br/>Right: Mesh = max / Boundary = circle / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>

<img style="height : 400px;" src="/media/compressed/max_LSCM_2Points_all_all.gif">
<br/>Right : Mesh = max / Boundary = 2 Points / Type : LSCM<br/>

<img style="height : 400px;" src="/media/compressed/off_lscm_freeP2_texture_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_freeP2_texture_plan.png">
<br/>Left & right: Mesh = max / Boundary = 2 Points / Type : LSCM / Textured <br/>

<img style="height : 400px;" src="/media/compressed/maxoff_lscm_freeP2_colorsMAX00001MIN0_Distortion_T0_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_freeP2_colorsMAX5MIN0_Distortion_T1_mesh.png">
<img style="height : 400px;" src="/media/compressed/off_lscm_freeP2_colorsMAX5MIN0_Distortion_T2_mesh.png">
<br/>Left: Mesh = max / Boundary = 2 Points / Type : LSCM / Colored (Max = 0.0001, Min = 0, distortion angle)
<br/>Center: Mesh = max / Boundary = 2 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion length)
<br/>Right: Mesh = max / Boundary = 2 Points / Type : LSCM / Colored (Max = 5, Min = 0, distortion area)<br/>
</p>
</div>

__________________________________

<a name="results"/>

#### 3. Construct the system and display the results

<a name="parameterization"/>

#### 3.1. Solve and show the parameterization.

<a name="distortion"/>

Once the system had been built, with the matrix C - the constraints - and the matrix A - the parametization mode - we can construct the final system to solve, and solve it.
As matrices are mainly sparse one, we can use the Eigen::SparseLU solver. 
<br/>
The parametization mode is chosen with keys '1','2','3'.
The computation and visualisation of the parameterization is chosen by pressing spacebar. 
You can use ’+’ and ’-’ to scale the texture to appropriate size.

###### Documentation informations

* __Eigen solver named "Eigen::SparseLU"__
* __igl::cat()__
Concatene sparse matrices.
[-> Documentation](https://github.com/libigl/libigl/blob/master/include/igl/cat.h)


**Inputs** :
* A = first input matrix
* B = second input matrix
* dim = the dimension on which we want to concatenate (1 = vertical , 2 = horizontal ) 

**Outputs** :
* C = the output concatenated matrices


###### Usage
In this part, it is just question of solving the system constructed before.
We concatenate A and C matrix in a good manner - with A in the upper left corner and C on the down left corner and its transposed matrix on the upper right corner - and we call SparseLU solver to get the solved coordinates vectors, that we store in global value, for the viewer.

```C++
  (...)
	// Data structure necessary
  (...)

	// Construction of the linear system
	igl::cat(2, A, Ct, TmpUp);
	igl::cat(2, C, vide, TmpDown);
	igl::cat(1, TmpUp, TmpDown, LeftSide);
	igl::cat(1, b, d, RightSide);

	// Solve the linear system.
	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;

	// Compute the ordering permutation vector from the structural pattern of A
	solver.analyzePattern(LeftSide);
  
	// Compute the numerical factorization
	solver.factorize(LeftSide);

	//Use the factors to solve the linear system
	solver.compute(LeftSide);

  (...)
	result = solver.solve(RightSide);

	// The solver will output a vector
	UV.resize(V.rows(), 2);

	//Block of size (p,q), starting at (i,j) <==> matrix.block(i,j,p,q);
	UV.col(0) = result.block(0, 0, tailleV, 1);
	UV.col(1) = result.block(tailleV, 0, tailleV, 1);

  (...)

	calculateColorsDistortion(); //Color the calculated UV.
  (...)
}
```

###### Result

No visual results. But a view of the solved system (cathead.obj, uniform) is displayed. (You can get this output in debug mode)


```
System : 
8.92974 -1.99509 -1.72084 0 0.844691 -2.83497 0 -1.78548 -1.43805 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.11022e-16 0 -2.22045e-16 0 -7.56158e-18 6.66134e-16 0 -2.45507e-16 -1.38778e-17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
-1.99509 8.3967 -1.89993 -2.12778 -2.3739 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1.11022e-16 -2.22045e-16 -1.52084e-16 -1.11022e-16 -5.73427e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
-1.72084 -1.89993 8.40898 0.306844 0 0 0 0 -0.258949 -1.20742 -0.139668 -1.3879 0 0 -2.10112 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.22045e-16 1.52084e-16 -1.11022e-16 0 0 0 0 0 -3.33067e-16 2.22045e-16 -2.23035e-16 4.25387e-16 0 0 2.22045e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
0 -2.12778 0.306844 9.42551 -0.794423 0 0 0 0 -2.59077 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.7224 0 0 0 0 0 -1.19023 -1.30674 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.11022e-16 1.11022e-16 -2.22045e-16 6.66134e-16 0 0 0 0 5.55112e-17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 6.48793e-17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
0.844691 -2.3739 0 -0.794423 7.70064 -1.73154 -1.2143 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.650007 -1.27378 -0.507379 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7.56158e-18 5.73427e-16 0 -6.66134e-16 0 5.56347e-17 -3.33067e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.11022e-16 -3.60822e-16 2.28194e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
-2.83497 0 0 0 -1.73154 8.69406 -2.029 -2.09855 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -7.77156e-16 0 0 0 -5.56347e-17 -4.44089e-16 3.33067e-16 -2.77575e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
0 0 0 0 -1.2143 -2.029 9.84172 0.451141 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -0.791492 -0.959929 -0.324802 -1.578 0 0 0 0 0 0 0 0 -3.39532 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.77556e-16 -5.55112e-16 -3.33067e-16 2.22045e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -3.29888e-16 4.996e-16 8.1425e-18 -5.55112e-16 0 0 0 0 0 0 0 0 3.45171e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
-1.78548 0 0 0 0 -2.09855 0.451141 8.76827 -0.767912 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.107561 -1.04801 -1.54338 -0.0698428 -2.0138 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4.67552e-16 0 0 0 0 2.77575e-16 -2.22045e-16 0 1.82776e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -3.80961e-16 3.03706e-17 5.55112e-17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
(...)
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.66715e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.63678e-16 0 0 0 0 0 0 0 0 0 0 0 0 -2.22045e-16 2.22045e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2.31975 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2.36604 0 0 0 0 0 0 0 0 0 0 0 9.5958 -0.7505 -4.15951 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2.1802e-16 0 0 0 0 0 0 0 -5.55112e-16 0 0 0 2.22045e-16 0 -6.66134e-16 2.22045e-16 -1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0443259 0 0 0 0 0 0 0 -0.0756202 0 0 0 -0.7505 4.93786 -0.55525 -2.75185 -0.644474 -0.204494 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0  = 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2.22045e-16 5.55112e-16 0 0 0 3.29199e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1.6309e-16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.255436 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -4.15951 -0.55525 12.0073 0 0 -5.39399 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2.15395 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4.93563e-16 0 -2.22045e-16 0 0 -2.22045e-16 0 -4.44089e-16 3.43892e-17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -2.06441 0 -2.95136 0 0 -2.75185 0 9.85926 -2.09164 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  = 0 
(...)
 
LeftSide Non Zeros : 3596 elements.
RightSide Non Zeros : 286 elements.

Lignes à droites : 286 lignes.
Cols à droites : 1 colonnes.

Dimension de At.A : 262 lignes * 262 colonnes.
Dimension de C : 24 lignes * 262 colonnes.
Dimension de Ct : 262 lignes * 24 colonnes.
Dimension de vide : 24 lignes * 24 colonnes.

Lignes à gauche : 286 lignes.
Cols à gauche : 286 colonnes.

Dimension de b : 262 lignes * 1 colonnes.
Dimension de d : 24 lignes * 1 colonnes.
```

__________________________________


#### 3.2. Visualize the distortion

Finally, we can use a color code to show the distortion of each face of the mesh after the parametization and so infer the quality of the results.
We consider 2 types of distortion : 
* angle distortion (Conformal, LSCM)
* edge length distortion (Isometric, ARAP)
* area distortion (Authalic)
<br/>
Highly distorted triangles appear in red and undistorted triangles in white.

###### Usage
So, we need to color face thanks to the distortion applied to them.
<br/>
The method used is to calculate the jacobian of the transformation applied to a face, for each face. Then we can extract the distortion, depending of the chosen mode of the user (angle distortion, length distortion ...)
<br/>
If we want the "length distortion" we can compute the distortion that isometric method try to minimize. Same for angle, or area .. 
So we have a solution, where we can compute the distortion (any kind, length, angle or area) whatever system we solved before (Harmonic, LSCM, ...)
<br/>
Then, for the coloration, two colorations are possibles : 
* First one, more automatic, not activated in the last version. From a threeshold defined by the user, we compute a gradient of color depending on the distorsion and the maximal distortion of the mesh. The problem is, even in case of "low distortion", we still have red triangles on the final picture.
* Second one, less automatic, activated in the last version. From a min/max value, we create a gradient of colors. We map the distortion values to this min/max color map. The tweaking is harder, but we can have more representative visualisation of the distortion.


```C++

void calculateColorsDistortion(){

  	(...)

	//Get Dx/Dy
	computeSurfaceGradientMatrix(Dx, Dy);

	//Get U and V
  	(...)
	uVectorTMP = uVector.sparseView();
	vVectorTMP = vVector.sparseView();
  	(...)

	//We calcule a big "Jacobian" matrix, for all vertices
	UpLeft = (Dx*uVectorTMP); // #Fx#V * #V*1 = #F * 1
	UpRight = (Dy*uVectorTMP); // #Fx#V * #V*1 = #F * 1
	DownLeft = (Dx*vVectorTMP); // #Fx#V * #V*1 = #F * 1
	DownRight = (Dy*vVectorTMP); //#Fx#V * #V*1 = #F * 1

  	(...)

	//Create the big Jack Matrix
	igl::cat(2, UpLeft, UpRight, tmpUp);
	igl::cat(2, DownLeft, DownRight, tmpDown);
	igl::cat(1, tmpUp, tmpDown, bigJac);

  	(...)
	
	//Create a Gradient per face and not more per vertex
	DxTMP = DxTMP.rowwise().sum();//(DxTMP.rowwise() > 0).count();
	DyTMP = DyTMP.rowwise().sum();

  	(...)

	//For each face, we have to calculate the distortion.
	for(int i =0 ; i <F.rows() ; i++){

		//Represent the 4 values of the jacobian of the current face
		double DxU =0;
		double DxV =0;
		double DyU =0;
		double DyV =0;

		for(int j = 0 ; j<3 ; j++){
			int indicePtcur = F.row(i)[j];

			//We calculate the jacobian for each point of the face
			DxU += bigJac.coeff(indicePtcur,0);
			DxV += bigJac.coeff(tailleV+indicePtcur,0);
			DyU += bigJac.coeff(indicePtcur,1);
			DyV += bigJac.coeff(tailleV+indicePtcur,1);
		}

		//Take the mean
		DxU /= 3;
		DxV /= 3;
		DyU /= 3;
		DyV /= 3;

		//Create the jacobian 
		Eigen::Matrix2d J;
		J.setZero();

		//Set values of the jacobian
		J.row(0)[0] = DxU;
		J.row(0)[1] = DyU;
		J.row(1)[0] = DxV;
		J.row(1)[1] = DyV;

  		(...)

		// Extract Singular Value via SSVD (Signed Singular Value Decomposition)
		SSVD2x2(J, Utmp, Stmp, Vtmp);

  		(...)

		//Compute the Distorsion, depending on the kind of distorsion computed
		if(showAngleDistortion == 0){ // ANGLE
			distorstionPerFace.row(i)[0] = ((sigmaUn-sigmaDeux)*(sigmaUn-sigmaDeux));
		} else if(showAngleDistortion == 1){ // LENGTH
			distorstionPerFace.row(i)[0]  = ((sigmaUn-1)*(sigmaUn-1) +(sigmaDeux-1)*(sigmaDeux-1));
		} else if(showAngleDistortion == 2){ // AREA
			distorstionPerFace.row(i)[0]  = ((sigmaUn*sigmaDeux-1)*(sigmaUn*sigmaDeux-1));
		}

	}

	// Calculate color code
	double maxCoef = distorstionPerFace.maxCoeff();
	double minCoef = distorstionPerFace.minCoeff();

	(...)
	distorstionPerFace.rows());
	for (int i = 0; i < Colors.rows(); i++)
	{
		Colors(i, 0) = 1;
		if (distorstionPerFace.row(i)[0] < Colormin)
		{
			//Under the threeshold = White
			Colors(i, 1) = 1;
			Colors(i, 2) = 1;
		}
		else if (distorstionPerFace.row(i)[0] > Colormax)
		{
			//Above the max = RED
			Colors(i, 1) = 0;
			Colors(i, 2) = 0;
		}
		else
		{ // Between min and max values, we color "slowly"
			double teinte = 1 - ((distorstionPerFace.row(i)[0] - Colormin) / ((double)(Colormax)));
			Colors(i, 1) = teinte;
			Colors(i, 2) = teinte;
		}
	}

  	(...)
}

```

###### Result
Many examples were dislayed before.
Some more examples are displayed here. (Min/max solution for presenting the distortion)

<div  style="text-align:center">
<p align="center">

<img style="height : 300px;" src="/media/compressed/bunny_uni_circle_all_all.gif">
<br/>Mesh = bunny / Boundary = circle / Type : Uniform<br/>

<img style="height : 300px;" src="/media/compressed/off_cotan_circle_colorsMAX150MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/o_lscm_circle_colorsMAX3MIN0_Distortion_T1_mesh.png">
<img style="height : 300px;" src="/media/compressed/cowobj_lscm_circle_colorsMAX001MIN0_Distortion_T0_mesh.png">
<br/>Left: Mesh = bunny / Boundary = circle / Type : cotan / Colored (Max = 15, Min = 0, distortion length)
<br/>Center: Mesh = cat / Boundary = circle / Type : LSCM / Colored (Max = 35, Min = 0, distortion length)
<br/>Right: Mesh = cow / Boundary = circle / Type : LSCM / Colored (Max = 0.01, Min = 0, distortion angle)<br/>

<img style="height : 300px;" src="/media/compressed/gargoyle__uni_circle_colorsMAX001MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispher_cotan_circle_colorsMAX1MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/hemispherNC_uni_circle_colorsMAX5MIN0_Distortion_T0_mesh.png">
<br/>Left: Mesh = gargoyle / Boundary = circle / Type : uni / Colored (Max = 0.01, Min = 0, distortion angle)
<br/>Center: Mesh = hemisphere / Boundary = circle / Type : cotan / Colored (Max = 1, Min = 0, distortion angle)
<br/>Right: Mesh = hemisphereNC / Boundary = circle / Type : uni / Colored (Max = 5, Min = 0, distortion angle)<br/>

<img style="height : 300px;" src="/media/compressed/off_uni_circle_colorsMAX0MIN0_Distortion_T0_mesh.png">
<img style="height : 300px;" src="/media/compressed/Octo_cut2_uni_circle_colorsMAX150MIN0_Distortion_T0_mesh.png">
<br/>Left: Mesh = max / Boundary = circle / Type : uni / Colored (Max = 0.0001, Min = 0, distortion angle)
<br/>Right: Mesh = octo / Boundary = circle / Type : uni / Colored (Max = 15, Min = 0, distortion angle)<br/>
</p>
</div>

Note that the code is not optimized by thoughtful laziness: this code is not made to be reused as is in other applications, and if this is the case, the code will be reviewed and optimized. Thanks for your understanding.
