// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/torch.h>

namespace fem
{
	class Mesh
	{
	public:
		at::Tensor points;
		at::Tensor vertex;
		at::Tensor line;
		at::Tensor triangle;
	};

	class PoissonSolution
	{
	public:
		at::Tensor u;
		at::Tensor A_position; // [numbers of nonzeros, 2]
		at::Tensor A_value; // [numbers of nonzeros, 1]
		at::Tensor b_value; // [numbers of dof, 1]
	};

	class PoissonSolver // f -> u
	{
	public:
		// computing in CPU
		PoissonSolution solve(const Mesh& mesh, const at::Tensor& f, c10::DeviceType return_device = at::kCPU);

	private:
		// compute (grad phi_i, grad phi_j) in the triangle tri_index (i, j in tri_index)
		at::Tensor A_e(const at::Tensor& points, const at::Tensor& tri_index, const at::Tensor& i, const at::Tensor& j);

		// compute (phi_i, f) in the triangle tri_index (i in tri_index)
		at::Tensor b_e(const at::Tensor& points, const at::Tensor& tri_index, const at::Tensor& i, const at::Tensor& f);
	};
	
}