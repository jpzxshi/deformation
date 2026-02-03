// @author jpzxshi (jpz@pku.edu.cn)
#include <filesystem>
#include "test.hpp"
#include <Eigen/Sparse>
#include "gen_data.hpp"
#include "fem.hpp"
#include "hybrid_solver.hpp"

void him_2d_varying_domain()
{
	// load net
	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/model_best_traced.pt");
	// load sampling points
	auto sample_points = torch::jit::load("./model/sample_points.pth", at::kCUDA);
	auto spts = sample_points.attr("disc_points_polar").toTensor();
	// load data
	auto data = ln::pickle_load("./data/data_482187.bin").toGenericDict();
	fem::Mesh mesh;
	mesh.points = data.at("points").toTensor();
	mesh.vertex = data.at("vertex").toTensor();
	mesh.line = data.at("line").toTensor();
	mesh.triangle = data.at("triangle").toTensor();
	auto f = data.at("f").toTensor();
	auto u = data.at("u").toTensor();
	auto A_position = data.at("A_position").toTensor();
	auto A_value = data.at("A_value").toTensor();
	auto b_value = data.at("b_value").toTensor();
	// test
	std::cout << "sampling points: " << spts.sizes() << " " << spts.dtype() << " " << spts.device() << std::endl;
	std::cout << "points: " << mesh.points.sizes() << " " << mesh.points.dtype() << " " << mesh.points.device() << std::endl;
	std::cout << "vertex: " << mesh.vertex.sizes() << " " << mesh.vertex.dtype() << " " << mesh.vertex.device() << std::endl;
	std::cout << "line: " << mesh.line.sizes() << " " << mesh.line.dtype() << " " << mesh.line.device() << std::endl;
	std::cout << "triangle: " << mesh.triangle.sizes() << " " << mesh.triangle.dtype() << " " << mesh.triangle.device() << std::endl;
	std::cout << "f: " << f.sizes() << " " << f.dtype() << " " << f.device() << std::endl;
	std::cout << "u: " << u.sizes() << " " << u.dtype() << " " << u.device() << std::endl;
	std::cout << "A_position: " << A_position.sizes() << " " << A_position.dtype() << " " << A_position.device() << std::endl;
	std::cout << "A_value: " << A_value.sizes() << " " << A_value.dtype() << " " << A_value.device() << std::endl;
	std::cout << "b_value: " << b_value.sizes() << " " << b_value.dtype() << " " << b_value.device() << std::endl;
	std::cout << std::endl;

	std::string mode = "GS";
	double error_threshold = 1e-13;
	int M = 13000;

	auto hybrid_solver = Hybrid_solver(mode, M, error_threshold, 1);
	std::tuple<double, int> time_iters = hybrid_solver.solve_poisson_2d_varying_domain(mesh, f, u, A_position, A_value, b_value, mionet, spts);
	std::cout << "size: 482187 (varying domain)" << std::endl;
	std::cout << "hybrid iterative method (" + mode + " + MIONet, M=" << M << "):\n";
	std::cout << std::setprecision(16);
	std::cout << "time: " << std::get<0>(time_iters) << " ms iterations: " << std::get<1>(time_iters) << std::endl;
}

int main()
{
	//device_test();

	//model_test();

	//data_test();

	if (!std::filesystem::exists("./data/data_482187.bin")) {
		gen::generate_data_varying_domain("./data/mesh_f_482187.pth", "./data/data_482187.bin");
	}
	him_2d_varying_domain();
}