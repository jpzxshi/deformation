// @author jpzxshi (jpz@pku.edu.cn)
#include "fem.hpp"
#include <Eigen/Sparse>

using namespace torch::indexing;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;

at::Tensor fem::PoissonSolver::A_e(const at::Tensor& points, const at::Tensor& tri_index, const at::Tensor& i, const at::Tensor& j)
{
	auto A = points.index({ tri_index });
	A.index_put_({ Slice(), 2 }, 1);
	//std::cout << A << A.options() << std::endl;

	auto b_i = torch::zeros({ 3 }, A.options());
	auto b_j = torch::zeros({ 3 }, A.options());
	auto index_i = torch::where(tri_index == i)[0];
	auto index_j = torch::where(tri_index == j)[0];
	b_i.index_put_({ index_i }, 1);
	b_j.index_put_({ index_j }, 1);
	//std::cout << b_i << b_i.options() << "\n" << b_j << b_j.options() << std::endl;

	auto a_i = torch::linalg::solve(A, b_i, true);
	auto a_j = torch::linalg::solve(A, b_j, true);
	//std::cout << a_i << a_i.options() << "\n" << a_j << a_j.options() << std::endl;
	//std::cout << torch::matmul(A, a_i) << torch::matmul(A, a_j) << std::endl;

	auto result = torch::dot(a_i.index({ Slice(None, 2) }), a_j.index({ Slice(None, 2) })) * torch::abs(torch::linalg::det(A)) / 2.0;
	//std::cout << result << result.options() << std::endl;
	return result;
}

at::Tensor fem::PoissonSolver::b_e(const at::Tensor& points, const at::Tensor& tri_index, const at::Tensor& i, const at::Tensor& f)
{
	auto A = points.index({ tri_index });
	A.index_put_({ Slice(), 2 }, 1);

	auto b_i = torch::zeros({ 3 }, A.options());
	auto index_i = torch::where(tri_index == i)[0];
	b_i.index_put_({ index_i }, 1);
	auto b = f.index({ tri_index }).ravel();

	auto s_i = torch::linalg::solve(A, b_i, true);
	auto s = torch::linalg::solve(A, b, true);

	auto a_i = s_i[0]; auto t_i = s_i[1]; auto c_i = s_i[2];
	auto a = s[0]; auto t = s[1]; auto c = s[2];
	auto x0 = A[0][0]; auto y0 = A[0][1]; auto x1 = A[1][0]; auto y1 = A[1][1]; auto x2 = A[2][0]; auto y2 = A[2][1];

	auto J = torch::tensor({
		{(x1 - x0).item<double>(), (x2 - x0).item<double>()},
		{(y1 - y0).item<double>(), (y2 - y0).item<double>()}
		}, A.options());
	auto det_J = torch::abs(torch::linalg::det(J));
	//std::cout << det_J << det_J.options() << std::endl;

	auto int_x2 = torch::pow(x0, 2) / 2.0 + torch::pow(x1 - x0, 2) / 12.0 + torch::pow(x2 - x0, 2) / 12.0 + (x1 - x0) * (x2 - x0) / 12.0 + x0 * (x1 - x0) / 3.0 + x0 * (x2 - x0) / 3.0;
	auto int_y2 = torch::pow(y0, 2) / 2.0 + torch::pow(y1 - y0, 2) / 12.0 + torch::pow(y2 - y0, 2) / 12.0 + (y1 - y0) * (y2 - y0) / 12.0 + y0 * (y1 - y0) / 3.0 + y0 * (y2 - y0) / 3.0;
	auto int_xy = (x1 - x0) * (y1 - y0) / 12.0 + (x1 - x0) * (y2 - y0) / 24.0 + (x1 - x0) * y0 / 6.0 + (x2 - x0) * (y1 - y0) / 24.0 + (x2 - x0) * (y2 - y0) / 12.0 + (x2 - x0) * y0 / 6.0
		+ (y1 - y0) * x0 / 6.0 + (y2 - y0) * x0 / 6.0 + x0 * y0 / 2.0;
	int_x2 = det_J * int_x2;
	int_y2 = det_J * int_y2;
	int_xy = det_J * int_xy;

	auto int_x = (x1 - x0) / 6.0 + (x2 - x0) / 6.0 + x0 / 2.0;
	auto int_y = (y1 - y0) / 6.0 + (y2 - y0) / 6.0 + y0 / 2.0;
	int_x = det_J * int_x;
	int_y = det_J * int_y;

	auto int_1 = det_J * 0.5;

	auto result = a_i * a * int_x2 + t_i * t * int_y2 + (a_i * t + t_i * a) * int_xy + (a * c_i + c * a_i) * int_x + (t * c_i + c * t_i) * int_y + c * c_i * int_1;
	//std::cout << result << result.options() << std::endl;
	return result;
}

fem::PoissonSolution fem::PoissonSolver::solve(const Mesh& mesh, const at::Tensor& f, c10::DeviceType return_device)
{
	auto points_cpu = mesh.points.to(at::kCPU);
	auto line_cpu = mesh.line.to(at::kCPU);
	auto triangle_cpu = mesh.triangle.to(at::kCPU);
	auto f_cpu = f.to(at::kCPU);

	auto N = points_cpu.size(0);
	SparseMatrix A(N, N);
	Vector b(N);

	std::cout << "Assembling A and b..." << std::endl;
	for (auto i = 0; i != triangle_cpu.size(0); ++i)
	{
		//std::cout << "triangle " << i << std::endl;
		auto tri_index = triangle_cpu[i];
		auto index_0 = tri_index[0].item<int>();
		auto index_1 = tri_index[1].item<int>();
		auto index_2 = tri_index[2].item<int>();
		// A
		A.coeffRef(index_0, index_0) += A_e(points_cpu, tri_index, tri_index[0], tri_index[0]).item<double>();
		A.coeffRef(index_0, index_1) += A_e(points_cpu, tri_index, tri_index[0], tri_index[1]).item<double>();
		A.coeffRef(index_0, index_2) += A_e(points_cpu, tri_index, tri_index[0], tri_index[2]).item<double>();
		A.coeffRef(index_1, index_1) += A_e(points_cpu, tri_index, tri_index[1], tri_index[1]).item<double>();
		A.coeffRef(index_1, index_2) += A_e(points_cpu, tri_index, tri_index[1], tri_index[2]).item<double>();
		A.coeffRef(index_2, index_2) += A_e(points_cpu, tri_index, tri_index[2], tri_index[2]).item<double>();
		// b
		b.coeffRef(index_0) += b_e(points_cpu, tri_index, tri_index[0], f_cpu).item<double>();
		b.coeffRef(index_1) += b_e(points_cpu, tri_index, tri_index[1], f_cpu).item<double>();
		b.coeffRef(index_2) += b_e(points_cpu, tri_index, tri_index[2], f_cpu).item<double>();
	}

	A += Eigen::SparseMatrix<double, Eigen::RowMajor>(A.triangularView<Eigen::StrictlyUpper>()).transpose() +
		Eigen::SparseMatrix<double, Eigen::RowMajor>(A.triangularView<Eigen::StrictlyLower>()).transpose();

	std::cout << "Deleting boundary rows and cols..." << std::endl;
	auto boundary = line_cpu.index({ Slice(), 0 });
	//std::cout << boundary.sizes() << std::endl;
	SparseMatrix A_delete(N - boundary.size(0), N - boundary.size(0));
	for (auto j = 0; j != A.outerSize(); ++j)
	{
		for (SparseMatrix::InnerIterator it(A, j); it; ++it)
		{
			//std::cout << "iter..." << std::endl;
			//std::cout << torch::isin(it.row(), boundary).item<bool>() << std::endl;
			if ((!torch::isin(it.row(), boundary).item<bool>()) && (!torch::isin(it.col(), boundary).item<bool>()))
			{
				//std::cout << "A insert..." << std::endl;
				//std::cout << torch::max(boundary) << " " << torch::min(boundary) << " " << it.row() << " " << it.col() << std::endl;
				auto row_bias = torch::where(boundary < it.row())[0].size(0);
				auto col_bias = torch::where(boundary < it.col())[0].size(0);
				//std::cout << "A insert (row, col)=" << it.row() << "-" << row_bias << "=" << it.row() - row_bias << ", " 
				//		  << it.col() << "-" << col_bias << "=" << it.col() - col_bias << "  max " << A_delete.rows() << " " << A_delete.cols() << std::endl;
				A_delete.insert(it.row() - row_bias, it.col() - col_bias) = it.value();
				//std::this_thread::sleep_for(std::chrono::milliseconds(500));
			}
		}
	}
	Vector b_delete(N - boundary.size(0));
	auto iter = 0;
	for (auto i = 0; i != b.size(); ++i)
	{
		if (!torch::isin(i, boundary).item<bool>())
		{
			//std::cout << "b insert " << iter << std::endl;
			b_delete[iter] = b[i];
			++iter;
		}
	}

	// for saving A_delete and b_delete, requre device = kCPU
	std::cout << "Saving A_delete and b_delete..." << std::endl;
	auto n = A_delete.nonZeros();
	auto A_position = torch::zeros({ n, 2 }, torch::TensorOptions().dtype(at::kInt).device(at::kCPU));
	auto A_value = torch::zeros({ n, 1 }, torch::TensorOptions().dtype(at::kDouble).device(at::kCPU));
	auto _A_position = A_position.accessor<int, 2>();
	auto _A_value = A_value.accessor<double, 2>();
	iter = 0;
	//std::cout << "init saving delete" << std::endl;
	for (auto j = 0; j != A_delete.outerSize(); ++j)
	{
		//std::cout << "saving A_delete col iter " << j << std::endl;
		for (SparseMatrix::InnerIterator it(A_delete, j); it; ++it)
		{
			_A_position[iter][0] = it.row();
			_A_position[iter][1] = it.col();
			_A_value[iter][0] = it.value();
			++iter;
		}
	}
	//std::cout << "saving b_delete" << std::endl;
	auto n_b = b_delete.size();
	auto b_value = torch::zeros({ n_b, 1 }, torch::TensorOptions().dtype(at::kDouble).device(at::kCPU));
	//std::cout << "init b_value" << std::endl;
	auto _b_value = b_value.accessor<double, 2>();
	//std::cout << "init _b_value" << std::endl;
	for (auto i = 0; i != b_delete.size(); ++i)
	{
		//std::cout << "saving b iter " << i << std::endl;
		_b_value[i][0] = b_delete(i);
	}
	//

	std::cout << "Solving..." << std::endl;
	Eigen::SimplicialCholesky<SparseMatrix> chol(A_delete);
	Vector u_delete = chol.solve(b_delete);

	std::cout << "Returning..." << std::endl;
	at::Tensor u = torch::zeros_like(f_cpu);
	auto _u = u.accessor<double, 2>();
	iter = 0;
	for (int i = 0; i != u.size(0); ++i)
	{
		if (!torch::isin(i, boundary).item<bool>())
		{
			_u[i][0] = u_delete(iter);
			++iter;
		}
	}
	//std::cout << iter << std::endl;

	PoissonSolution solution;
	solution.u = u.to(return_device);
	solution.A_position = A_position.to(return_device);
	solution.A_value = A_value.to(return_device);
	solution.b_value = b_value.to(return_device);
	return solution;
}