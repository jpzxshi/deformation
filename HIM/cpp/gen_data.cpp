// @author jpzxshi (jpz@pku.edu.cn)
#include "gen_data.hpp"
#include "cln.hpp"
#include "fem.hpp"
//#include <torch/script.h>
#include <Eigen/Sparse>

using namespace torch::indexing;
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

std::tuple<at::Tensor, at::Tensor> gen::load_kf(const std::string& filename)
{
	auto data = torch::jit::load(filename, at::kCPU);
	auto k = data.attr("k").toTensor().to(at::kCPU);
	auto f = data.attr("f").toTensor().to(at::kCPU);
	return std::make_tuple(k ,f);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gen::load_kfg(const std::string& filename)
{
	auto data = torch::jit::load(filename, at::kCPU);
	auto k = data.attr("k").toTensor().to(at::kCPU);
	auto f = data.attr("f").toTensor().to(at::kCPU);
	auto g = data.attr("g").toTensor().to(at::kCPU);
	return std::make_tuple(k, f, g);
}

at::Tensor gen::solve_sparse(const at::Tensor& k, const at::Tensor& f)
{
	c10::TensorOptions ops = at::device(k.device()).dtype(k.dtype());
	int n = k.size(0) - 2;
	int size = pow(n, 2);
	double h = 1.0 / (k.size(0) - 1.0);
	// D tensor
	at::Tensor D = (
		(4.0 / 3.0) * k.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 3.0) * (k.index({ Slice(None, -2), Slice(2, None) }) + k.index({ Slice(2, None), Slice(None, -2) })) +
		(1.0 / 2.0) * (k.index({ Slice(1, -1), Slice(2, None) }) + k.index({ Slice(1, -1), Slice(None, -2) }) +
			k.index({ Slice(2, None), Slice(1, -1) }) + k.index({ Slice(None, -2), Slice(1, -1) }))
		).ravel();
	// D sparse
	SparseMatrix D_s(size, size);
	auto _D = D.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		D_s.insert(i, i) = _D[i];
	}
	// U1 tensor
	at::Tensor U1 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -1), Slice(1, -2) }) + k.index({ Slice(1, -1), Slice(2, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(None, -2), Slice(2, -1) }) + k.index({ Slice(2, None), Slice(1, -2) }))
		);
	U1 = torch::cat({ U1, torch::zeros({n, 1}, ops) }, 1).ravel();
	// U1 sparse
	SparseMatrix U1_s(size, size);
	auto _U1 = U1.accessor<double, 1>();
	for (int i = 0; i != size - 1; ++i)
	{
		U1_s.insert(i, i + 1) = _U1[i];
	}
	// U2 tensor
	at::Tensor U2 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -2), Slice(1, -1) }) + k.index({ Slice(2, -1), Slice(1, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(1, -2), Slice(2, None) }) + k.index({ Slice(2, -1), Slice(None, -2) }))
		);
	U2 = torch::cat({ U2, torch::zeros({1, n}, ops) }, 0).ravel();
	// U2 sparse
	SparseMatrix U2_s(size, size);
	auto _U2 = U2.accessor<double, 1>();
	for (int i = 0; i != size - n; ++i)
	{
		U2_s.insert(i, i + n) = _U2[i];
	}

	SparseMatrix U_s = U1_s + U2_s;
	SparseMatrix L_s = SparseMatrix(U_s.transpose());
	SparseMatrix A_s = D_s + U_s + L_s;
	A_s.makeCompressed();

	// b tensor
	auto b = pow(h, 2) * (
		(1.0 / 2.0) * f.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 12.0) * (f.index({ Slice(2, None), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(2, None) }) + f.index({ Slice(None, -2), Slice(2, None) }) +
			f.index({ Slice(None, -2), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(None, -2) }) + f.index({ Slice(2, None), Slice(None, -2) }))
		).ravel();
	// b eigen vector
	Eigen::VectorXd b_s(size);
	auto _b = b.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		b_s[i] = _b[i];
	}
	// solve
	Eigen::SimplicialCholesky<SparseMatrix> chol(A_s);
	Eigen::VectorXd u_s = chol.solve(b_s);
	at::Tensor u = torch::zeros_like(b);
	auto _u = u.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		_u[i] = u_s[i];
	}
	u.resize_({ n, n });
	return torch::cat({ torch::zeros({n + 2, 1}, ops), torch::cat({torch::zeros({1, n}, ops), u, torch::zeros({1, n}, ops)}, 0), torch::zeros({n + 2, 1}, ops) }, 1);
}

at::Tensor gen::solve_sparse(const at::Tensor& k, const at::Tensor& f, const at::Tensor& g)
{
	c10::TensorOptions ops = at::device(k.device()).dtype(k.dtype());
	int n = k.size(0);
	int size = pow(n, 2);
	int n_in = k.size(0) - 2;
	int size_in = pow(n_in, 2);
	double h = 1.0 / (k.size(0) - 1.0);

	SparseMatrix A_s(size, size);
	at::Tensor index = torch::arange(size).reshape_symint({ n, n });

	// D tensor
	at::Tensor D = (
		(4.0 / 3.0) * k.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 3.0) * (k.index({ Slice(None, -2), Slice(2, None) }) + k.index({ Slice(2, None), Slice(None, -2) })) +
		(1.0 / 2.0) * (k.index({ Slice(1, -1), Slice(2, None) }) + k.index({ Slice(1, -1), Slice(None, -2) }) +
			k.index({ Slice(2, None), Slice(1, -1) }) + k.index({ Slice(None, -2), Slice(1, -1) }))
		).ravel();
	// D sparse
	SparseMatrix D_in(size_in, size_in);
	auto _D = D.accessor<double, 1>();
	for (int i = 0; i != size_in; ++i)
	{
		D_in.insert(i, i) = _D[i];
	}
	// U1 tensor
	at::Tensor U1 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -1), Slice(1, -2) }) + k.index({ Slice(1, -1), Slice(2, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(None, -2), Slice(2, -1) }) + k.index({ Slice(2, None), Slice(1, -2) }))
		);
	U1 = torch::cat({ U1, torch::zeros({n_in, 1}, ops) }, 1).ravel();
	// U1 sparse
	SparseMatrix U1_in(size_in, size_in);
	auto _U1 = U1.accessor<double, 1>();
	for (int i = 0; i != size_in - 1; ++i)
	{
		U1_in.insert(i, i + 1) = _U1[i];
	}
	// U2 tensor
	at::Tensor U2 = (
		(-1.0 / 3.0) * (k.index({ Slice(1, -2), Slice(1, -1) }) + k.index({ Slice(2, -1), Slice(1, -1) })) +
		(-1.0 / 6.0) * (k.index({ Slice(1, -2), Slice(2, None) }) + k.index({ Slice(2, -1), Slice(None, -2) }))
		);
	U2 = U2.ravel();
	// U2 sparse
	SparseMatrix U2_in(size_in, size_in);
	auto _U2 = U2.accessor<double, 1>();
	for (int i = 0; i != size_in - n_in; ++i)
	{
		U2_in.insert(i, i + n_in) = _U2[i];
	}

	SparseMatrix U_in = U1_in + U2_in;
	SparseMatrix L_in = SparseMatrix(U_in.transpose());
	SparseMatrix A_in = D_in + U_in + L_in;
	A_in.makeCompressed();
	// in
	at::Tensor index_in = index.index({ Slice(1, -1), Slice(1, -1) }).ravel();
	auto _index_in = index_in.accessor<int64_t, 1>();
	for (int i = 0; i != size_in; ++i)
	{
		for (SparseMatrix::InnerIterator it(A_in, i); it; ++it)
		{
			A_s.insert(_index_in[i], _index_in[it.col()]) = it.value();
		}
	}
	// up
	at::Tensor b_up = ((-1.0 / 3.0) * (k.index({ 1, Slice(1, -1) }) + k.index({ 0, Slice(1, -1) }))
		+ (-1.0 / 6.0) * (k.index({ 0, Slice(None, -2) }) + k.index({ 1, Slice(2, None) }))).ravel();
	at::Tensor index_up_r = index.index({ 1, Slice(1, -1) }).ravel();
	at::Tensor index_up_c = index.index({ 0, Slice(1, -1) }).ravel();
	auto _b_up = b_up.accessor<double, 1>();
	auto _index_up_r = index_up_r.accessor<int64_t, 1>();
	auto _index_up_c = index_up_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_up_r[i], _index_up_c[i]) = _b_up[i];
	}
	// left
	at::Tensor b_left = ((-1.0 / 3.0) * (k.index({ Slice(1, -1), 1 }) + k.index({ Slice(1, -1), 0 }))
		+ (-1.0 / 6.0) * (k.index({ Slice(None, -2), 0 }) + k.index({ Slice(2, None), 1 }))).ravel();
	at::Tensor index_left_r = index.index({ Slice(1, -1), 1 }).ravel();
	at::Tensor index_left_c = index.index({ Slice(1, -1), 0 }).ravel();
	auto _b_left = b_left.accessor<double, 1>();
	auto _index_left_r = index_left_r.accessor<int64_t, 1>();
	auto _index_left_c = index_left_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_left_r[i], _index_left_c[i]) = _b_left[i];
	}
	// bottom
	at::Tensor b_bottom = ((-1.0 / 3.0) * (k.index({ -2, Slice(1, -1) }) + k.index({ -1, Slice(1, -1) }))
		+ (-1.0 / 6.0) * (k.index({ -2, Slice(None, -2) }) + k.index({ -1, Slice(2, None) }))).ravel();
	at::Tensor index_bottom_r = index.index({ -2, Slice(1, -1) }).ravel();
	at::Tensor index_bottom_c = index.index({ -1, Slice(1, -1) }).ravel();
	auto _b_bottom = b_bottom.accessor<double, 1>();
	auto _index_bottom_r = index_bottom_r.accessor<int64_t, 1>();
	auto _index_bottom_c = index_bottom_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_bottom_r[i], _index_bottom_c[i]) = _b_bottom[i];
	}
	// right
	at::Tensor b_right = ((-1.0 / 3.0) * (k.index({ Slice(1, -1), -2 }) + k.index({ Slice(1, -1), -1 }))
		+ (-1.0 / 6.0) * (k.index({ Slice(None, -2), -2 }) + k.index({ Slice(2, None), -1 }))).ravel();
	at::Tensor index_right_r = index.index({ Slice(1, -1), -2 }).ravel();
	at::Tensor index_right_c = index.index({ Slice(1, -1), -1 }).ravel();
	auto _b_right = b_right.accessor<double, 1>();
	auto _index_right_r = index_right_r.accessor<int64_t, 1>();
	auto _index_right_c = index_right_c.accessor<int64_t, 1>();
	for (int i = 0; i != n_in; ++i)
	{
		A_s.insert(_index_right_r[i], _index_right_c[i]) = _b_right[i];
	}
	// boundary
	at::Tensor index_boundary = torch::cat({ index.index({0}), index.index({Slice(1, -1), Slice(0, None, n - 1)}).ravel(), index.index({-1}) }, 0);
	auto _index_boundary = index_boundary.accessor<int64_t, 1>();
	for (int i = 0; i != index_boundary.size(0); ++i)
	{
		A_s.insert(_index_boundary[i], _index_boundary[i]) = 1.0;
	}

	A_s.makeCompressed();

	// b tensor
	at::Tensor b = pow(h, 2) * (
		(1.0 / 2.0) * f.index({ Slice(1, -1), Slice(1, -1) }) +
		(1.0 / 12.0) * (f.index({ Slice(2, None), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(2, None) }) + f.index({ Slice(None, -2), Slice(2, None) }) +
			f.index({ Slice(None, -2), Slice(1, -1) }) + f.index({ Slice(1, -1), Slice(None, -2) }) + f.index({ Slice(2, None), Slice(None, -2) }))
		);
	b = torch::cat({ torch::flip(g.index({ Slice(3 * n - 2, None) }), {0}).reshape_symint({1, n_in}), b, g.index({Slice(n, 2 * n - 2)}).reshape_symint({1, n_in}) }, 0);
	b = torch::cat({ g.index({Slice(None, n)}).reshape_symint({n, 1}), b, torch::flip(g.index({Slice(2 * n - 2, 3 * n - 2)}), {0}).reshape_symint({n, 1})}, 1);
	b = b.ravel();
	// b eigen vector
	Eigen::VectorXd b_s(size);
	auto _b = b.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		b_s[i] = _b[i];
	}
	// solve (Eigen::SparseLU requires ColMajor!!!!!)
	Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> lu;
	lu.compute(Eigen::SparseMatrix<double, Eigen::ColMajor>(A_s));
	//std::cout << (lu.info() == Eigen::Success) << std::endl;
	Eigen::VectorXd u_s = lu.solve(b_s);
	at::Tensor u = torch::zeros_like(b);
	auto _u = u.accessor<double, 1>();
	for (int i = 0; i != size; ++i)
	{
		_u[i] = u_s[i];
	}
	return u.reshape_symint({ n, n });
}

void gen::generate_data_large(const std::string& loadname, const std::string& savename)
{
	std::tuple<at::Tensor, at::Tensor> kf = load_kf(loadname);
	auto k = std::get<0>(kf);
	auto f = std::get<1>(kf);
	auto u = torch::zeros_like(k);
	for (auto i = 0; i != k.size(0); ++i)
	{
		u.index_put_({i}, solve_sparse(k[i], f[i]));
		std::cout << "solved " << i << std::endl;
	}
	//std::cout << k.sizes() << f.sizes() << u.sizes() << std::endl;

	ln::pickle_save(savename, std::make_tuple(k, f, u));

	// test
	auto data = ln::pickle_load(savename).toTuple()->elements();
	auto data_k = data[0].toTensor();
	auto data_f = data[1].toTensor();
	auto data_u = data[2].toTensor();

	std::cout << data_k.sizes() << data_f.sizes() << data_u.sizes() << std::endl;
}

void gen::generate_data_boundary(const std::string& loadname, const std::string& savename)
{
	std::tuple<at::Tensor, at::Tensor, at::Tensor> kfg = load_kfg(loadname);
	auto k = std::get<0>(kfg);
	auto f = std::get<1>(kfg);
	auto g = std::get<2>(kfg);
	auto u = torch::zeros_like(k);
	for (auto i = 0; i != k.size(0); ++i)
	{
		u.index_put_({ i }, solve_sparse(k[i], f[i], g[i]));
		std::cout << "solved " << i << std::endl;
	}
	//std::cout << k.sizes() << f.sizes() << g.sizes() << u.sizes() << std::endl;

	ln::pickle_save(savename, std::make_tuple(k, f, g, u));

	// test
	auto data = ln::pickle_load(savename).toTuple()->elements();
	auto data_k = data[0].toTensor();
	auto data_f = data[1].toTensor();
	auto data_g = data[2].toTensor();
	auto data_u = data[3].toTensor();

	std::cout << data_k.sizes() << data_f.sizes() << data_g.sizes() << data_u.sizes() << std::endl;
}









void gen::generate_data_varying_domain(const std::string& loadname, const std::string& savename)
{
	auto mesh_f = torch::jit::load(loadname, at::kCUDA);

	fem::Mesh mesh; auto device = at::kCPU;
	mesh.points = mesh_f.attr("points").toTensor().to(device);
	mesh.vertex = mesh_f.attr("vertex").toTensor().to(device);
	mesh.line = mesh_f.attr("line").toTensor().to(device);
	mesh.triangle = mesh_f.attr("triangle").toTensor().to(device);
	auto f = mesh_f.attr("f").toTensor().to(device);
	// u_true
	//auto u_true = mesh_f.attr("u").toTensor().to(device);

	//std::cout << "Shapes of points, vertex, line, triangle, f are " << "\n";
	//std::cout << mesh.points.sizes() << mesh.vertex.sizes() << mesh.line.sizes() << mesh.triangle.sizes() << f.sizes() << std::endl;
	fem::PoissonSolver solver;
	auto solution = solver.solve(mesh, f);
	auto u = solution.u;
	auto A_position = solution.A_position;
	auto A_value = solution.A_value;
	auto b_value = solution.b_value;
	//std::cout << u_true.sizes() << std::endl;
	//std::cout << u.sizes() << std::endl;
	//std::cout << u_true.index({Slice(-5, None)}) << std::endl;
	//std::cout << u.index({ Slice(-5, None) }) << std::endl;
	//std::cout << "max error " << torch::max(torch::abs(u - u_true)) << std::endl;
	
	// save device
	device = at::kCPU;
	torch::Dict<std::string, at::Tensor> mesh_f_u;
	mesh_f_u.insert("points", mesh.points.to(device));
	mesh_f_u.insert("vertex", mesh.vertex.to(device));
	mesh_f_u.insert("line", mesh.line.to(device));
	mesh_f_u.insert("triangle", mesh.triangle.to(device));
	mesh_f_u.insert("f", f.to(device));
	mesh_f_u.insert("u", u.to(device));
	// A b
	mesh_f_u.insert("A_position", A_position.to(device));
	mesh_f_u.insert("A_value", A_value.to(device));
	mesh_f_u.insert("b_value", b_value.to(device));

	ln::pickle_save(savename, c10::IValue(mesh_f_u));
	std::cout << "data saved" << std::endl;

	// test
	auto data = ln::pickle_load(savename).toGenericDict();
	auto data_points = data.at("points").toTensor();
	auto data_vertex = data.at("vertex").toTensor();
	auto data_line = data.at("line").toTensor();
	auto data_triangle = data.at("triangle").toTensor();
	auto data_f = data.at("f").toTensor();
	auto data_u = data.at("u").toTensor();
	auto data_A_position = data.at("A_position").toTensor();
	auto data_A_value = data.at("A_value").toTensor();
	auto data_b_value = data.at("b_value").toTensor();


	std::cout << "Shapes of points, vertex, line, triangle, f, u, A_position, A_value, b_value are " << "\n";
	std::cout << data_points.sizes() << data_vertex.sizes() << data_line.sizes() << data_triangle.sizes() << data_f.sizes() << data_u.sizes() 
			  << data_A_position.sizes() << data_A_value.sizes() << data_b_value.sizes() << std::endl;
	std::cout << "Types of points, vertex, line, triangle, f, u, A_position, A_value, b_value are " << "\n";
	std::cout << data_points.dtype() << data_vertex.dtype() << data_line.dtype() << data_triangle.dtype() << data_f.dtype() << data_u.dtype()
			  << data_A_position.dtype() << data_A_value.dtype() << data_b_value.dtype() << std::endl;
	std::cout << "Devices of points, vertex, line, triangle, f, u, A_position, A_value, b_value are " << "\n";
	std::cout << data_points.device() << data_vertex.device() << data_line.device() << data_triangle.device() << data_f.device() << data_u.device()
			  << data_A_position.device() << data_A_value.device() << data_b_value.device() << std::endl;

	//// check A and b
	auto N = data_b_value.size(0);
	Eigen::SparseMatrix<double, Eigen::ColMajor> A(N, N);
	Eigen::VectorXd b(N);
	// init A
	std::vector<Eigen::Triplet<double>> triplets;
	auto data_A_position_cpu = data_A_position.to(at::kCPU);
	auto data_A_value_cpu = data_A_value.to(at::kCPU);
	auto _data_A_position = data_A_position_cpu.accessor<int, 2>();
	auto _data_A_value = data_A_value_cpu.accessor<double, 2>();
	for (auto i = 0; i != data_A_position_cpu.size(0); ++i)
	{
		triplets.emplace_back(_data_A_position[i][0], _data_A_position[i][1], _data_A_value[i][0]);
	}
	A.setFromTriplets(triplets.begin(), triplets.end());
	// init b
	auto data_b_value_cpu = data_b_value.to(at::kCPU);
	auto _data_b_value = data_b_value_cpu.accessor<double, 2>();
	for (auto i = 0; i != N; ++i)
	{
		b(i) = _data_b_value[i][0];
	}
	// solve
	Eigen::SimplicialCholesky<SparseMatrix> chol(A);
	Eigen::VectorXd u_solve = chol.solve(b);
	// init u_data
	Eigen::VectorXd u_data(N);
	auto data_u_delete = data_u.index({ Slice(-N, None), Slice() }).to(at::kCPU);
	auto _data_u_delete = data_u_delete.accessor<double, 2>();
	for (auto i = 0; i != data_u_delete.size(0); ++i)
	{
		u_data(i) = _data_u_delete[i][0];
	}
	// compute error
	std::cout << "u_data \n" << u_data.segment(3000, 5) << std::endl;
	std::cout << "u_solve \n" << u_solve.segment(3000, 5) << std::endl;
	std::cout << "max error " << (u_data - u_solve).lpNorm<Eigen::Infinity>() << std::endl;
}