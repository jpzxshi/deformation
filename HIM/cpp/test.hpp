// @author jpzxshi (jpz@pku.edu.cn)
#pragma once
#include <torch/torch.h>
#include "cln.hpp"

void device_test()
{
	torch::Tensor tensor = torch::rand({ 2, 3 });
	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU" << std::endl;
		auto tensor_cuda = tensor.cuda();
		std::cout << tensor_cuda << std::endl;
	}
	else
	{
		std::cout << "CUDA is not available! Training on CPU" << std::endl;
		std::cout << tensor << std::endl;
	}
}

void model_test()
{
	auto guard = ln::Inference_mode();
	
	//auto a = torch::tensor( 2.0 ).requires_grad_(true);
	//std::cout << a.requires_grad() << std::endl;
	//auto y = a * a;
	//auto grads = torch::autograd::grad({ y }, { a });

	ln::Loader loader(at::kCUDA, at::kDouble);
	auto mionet = loader.load_net("./model/model_best_traced.pt");

	auto r = torch::ones({ 2, 200 }, at::device(at::kCUDA).dtype(at::kDouble));
	auto f = torch::ones({ 2, 5000 }, at::device(at::kCUDA).dtype(at::kDouble));
	auto x = torch::tensor({ {1, 2}, {3, 4}, {5, 6}, {7, 8} }, at::device(at::kCUDA).dtype(at::kDouble));
	ln::Inputs inputs = { std::make_tuple(r, f, x) };
	
	at::Tensor u;
	// timing
	auto start = clock();
	auto iters = 1;
	for (auto i = 0; i != iters; ++i)
	{
		u = mionet.forward(inputs).toTensor();
		//std::cout << i;
	}
	auto end = clock();
	// once: debug 680 ms ; release 580 ms // 1000average: debug 7.8 ms ; release 7.2 ms
	std::cout << "prediction is \n";
	std::cout << u << "\n";
	std::cout << iters << " iters, time: " << end - start << " ms\n";
	//std::cout << torch::jit::getProfilingMode() << torch::jit::getExecutorMode();
	std::cout << std::endl;
}

void data_test()
{
	auto data = torch::jit::load("./data/mesh_f_482187.pth", at::kCUDA);
	auto sample_points = torch::jit::load("./model/sample_points.pth", at::kCUDA);

	auto points = data.attr("points").toTensor().to(at::kCUDA);
	auto vertex = data.attr("vertex").toTensor().to(at::kCUDA);
	auto line = data.attr("line").toTensor().to(at::kCUDA);
	auto triangle = data.attr("triangle").toTensor().to(at::kCUDA);
	auto f = data.attr("f").toTensor().to(at::kCUDA);
	auto disc_points_polar = sample_points.attr("disc_points_polar").toTensor().to(at::kCUDA);
	
	std::cout << "device: " << at::kCUDA << "\n" << "Shapes of points, vertex, line, triangle, f, disc_points_polar are " << "\n";
	std::cout << points.sizes() << vertex.sizes() << line.sizes() << triangle.sizes() << f.sizes() << disc_points_polar.sizes() << std::endl;
	std::cout << points.dtype() << vertex.dtype() << line.dtype() << triangle.dtype() << f.dtype() << disc_points_polar.dtype() << std::endl;
}