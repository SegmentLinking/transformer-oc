// Loads the exported TorchScript model with libtorch, runs it on one event and compares with Python.
// usage: run_torchscript <model.pt> <prefix> <cpu|cuda> [n_timing_runs]
//   reads <prefix>_x.bin and <prefix>_ref.bin written by export/make_cpp_testdata.py
// Build: see build.sh in this directory.

#include <torch/cuda.h>
#include <torch/script.h>

#include <algorithm>
#include <cstring>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  std::vector<char> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error("cannot open " + path);
    return std::vector<char>(std::istreambuf_iterator<char>(in), {});
  }

  int64_t readInt64(const std::vector<char>& buf, size_t offset) {
    int64_t v;
    std::memcpy(&v, buf.data() + offset, sizeof(v));
    return v;
  }

  float maxAbsDiff(const torch::Tensor& a, const torch::Tensor& b) {
    return (a.to(torch::kCPU).reshape(-1) - b.reshape(-1)).abs().max().item<float>();
  }
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <model.pt> <prefix> <cpu|cuda> [n_timing_runs]\n";
    return 2;
  }
  const std::string modelPath = argv[1], prefix = argv[2], devName = argv[3];
  const int nRuns = argc > 4 ? std::stoi(argv[4]) : 3;
  const torch::Device device(devName == "cuda" ? torch::kCUDA : torch::kCPU);

  // inputs and Python reference
  auto xBuf = readFile(prefix + "_x.bin");
  auto refBuf = readFile(prefix + "_ref.bin");
  const int64_t n = readInt64(xBuf, 0);
  const int64_t latent = readInt64(refBuf, 8);
  auto xRaw = torch::from_blob(xBuf.data() + 8, {n, 49}, torch::kFloat32).clone();
  auto betaRef = torch::from_blob(refBuf.data() + 16, {n}, torch::kFloat32).clone();
  auto hRef = torch::from_blob(refBuf.data() + 16 + 4 * n, {n, latent}, torch::kFloat32).clone();

  torch::NoGradGuard noGrad;
  torch::jit::Module module = torch::jit::load(modelPath, device);
  module.eval();
  auto x = xRaw.to(device);

  // 1. full model: forward(x_raw) -> (beta, H)
  auto runFull = [&] {
    auto out = module.forward({x}).toTuple();
    if (device.is_cuda())
      torch::cuda::synchronize();
    return out;
  };
  auto out = runFull();
  auto beta = out->elements()[0].toTensor(), h = out->elements()[1].toTensor();
  std::cout << "N=" << n << " device=" << devName << "\n";
  std::cout << "forward:     max|beta-ref|=" << maxAbsDiff(beta, betaRef) << " max|H-ref|=" << maxAbsDiff(h, hRef)
            << "\n";

  // 2. neighbours from the exported method, then forward_net (the interface LST will use with its own neighbours)
  auto nb = module.get_method("neighbors")({x}).toTuple();
  auto idx = nb->elements()[0].toTensor(), mask = nb->elements()[1].toTensor();
  auto outNet = module.get_method("forward_net")({x, idx, mask}).toTuple();
  std::cout << "forward_net: max|beta-ref|=" << maxAbsDiff(outNet->elements()[0].toTensor(), betaRef)
            << " max|H-ref|=" << maxAbsDiff(outNet->elements()[1].toTensor(), hRef)
            << "  (neighbours " << idx.size(0) << "x" << idx.size(1) << ")\n";

  // timing of the full model (first call above was warm-up)
  double best = 1e30;
  for (int i = 0; i < nRuns; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    runFull();
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::cout << "forward time (best of " << nRuns << "): " << best << " ms\n";
  return 0;
}
