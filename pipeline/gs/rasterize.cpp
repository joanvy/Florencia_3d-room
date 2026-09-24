// CPU tile rasterizer for 3D Gaussian Splatting (forward + backward).
//
// Everything before rasterization (projection, covariance, SH -> color) is
// done in PyTorch so autograd handles it; this file only implements the
// part that is too slow as tensor ops: binning 2D Gaussians into 16x16 tiles,
// depth sorting, front-to-back alpha compositing, and its analytic gradient.
//
// Conventions: pixel (x, y) has its centre at (x + 0.5, y + 0.5).
// conic = (a, b, c) is the inverse 2D covariance; the Gaussian falloff is
//   power = -0.5 * (a dx^2 + c dy^2) - b dx dy,   d = mean2d - pixel.

#include <torch/extension.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr int TILE = 16;
constexpr float MIN_ALPHA = 1.0f / 255.0f;
constexpr float MAX_ALPHA = 0.99f;
constexpr float T_EPS = 1e-4f;

struct Binning {
  int tiles_x, tiles_y;
  std::vector<int64_t> ranges;  // [num_tiles + 1]
  std::vector<int32_t> ids;     // gaussian ids, depth-sorted within each tile
};

Binning bin_gaussians(const float* means2d, const float* depths, const int32_t* radii, int64_t n, int width,
                      int height) {
  Binning b;
  b.tiles_x = (width + TILE - 1) / TILE;
  b.tiles_y = (height + TILE - 1) / TILE;
  const int num_tiles = b.tiles_x * b.tiles_y;

  // Tile rectangle per gaussian.
  std::vector<int32_t> rect(n * 4);
  std::vector<int64_t> counts(num_tiles, 0);
#pragma omp parallel
  {
    std::vector<int64_t> local(num_tiles, 0);
#pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
      int32_t* r = &rect[i * 4];
      const int rad = radii[i];
      if (rad <= 0) {
        r[0] = r[1] = r[2] = r[3] = 0;
        continue;
      }
      const float mx = means2d[i * 2], my = means2d[i * 2 + 1];
      r[0] = std::clamp((int)std::floor((mx - rad) / TILE), 0, b.tiles_x);
      r[1] = std::clamp((int)std::floor((my - rad) / TILE), 0, b.tiles_y);
      r[2] = std::clamp((int)std::ceil((mx + rad) / TILE), 0, b.tiles_x);
      r[3] = std::clamp((int)std::ceil((my + rad) / TILE), 0, b.tiles_y);
      for (int ty = r[1]; ty < r[3]; ++ty)
        for (int tx = r[0]; tx < r[2]; ++tx) local[ty * b.tiles_x + tx]++;
    }
#pragma omp critical
    for (int t = 0; t < num_tiles; ++t) counts[t] += local[t];
  }

  b.ranges.assign(num_tiles + 1, 0);
  for (int t = 0; t < num_tiles; ++t) b.ranges[t + 1] = b.ranges[t] + counts[t];
  b.ids.resize(b.ranges[num_tiles]);

  // Fill (sequential: preserves determinism and is cheap relative to rendering).
  std::vector<int64_t> cursor(b.ranges.begin(), b.ranges.end() - 1);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t* r = &rect[i * 4];
    for (int ty = r[1]; ty < r[3]; ++ty)
      for (int tx = r[0]; tx < r[2]; ++tx) b.ids[cursor[ty * b.tiles_x + tx]++] = (int32_t)i;
  }

#pragma omp parallel for schedule(dynamic, 4)
  for (int t = 0; t < num_tiles; ++t) {
    std::sort(b.ids.begin() + b.ranges[t], b.ids.begin() + b.ranges[t + 1],
              [depths](int32_t a, int32_t c) { return depths[a] < depths[c]; });
  }
  return b;
}

inline bool eval_alpha(const float* m, const float* con, float op, float px, float py, float& dx, float& dy,
                       float& g, float& alpha) {
  dx = m[0] - px;
  dy = m[1] - py;
  const float power = -0.5f * (con[0] * dx * dx + con[2] * dy * dy) - con[1] * dx * dy;
  if (power > 0.f) return false;
  g = std::exp(power);
  alpha = op * g;
  if (alpha < MIN_ALPHA) return false;
  return true;
}

}  // namespace

// Returns: image [H,W,3], final_T [H,W], n_contrib [H,W] (int32), tile_ranges, tile_ids
std::vector<torch::Tensor> rasterize_forward(torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
                                             torch::Tensor opacities, torch::Tensor depths, torch::Tensor radii,
                                             int64_t width, int64_t height, torch::Tensor background) {
  TORCH_CHECK(means2d.is_contiguous() && conics.is_contiguous() && colors.is_contiguous() &&
              opacities.is_contiguous() && depths.is_contiguous() && radii.is_contiguous());
  const int64_t n = means2d.size(0);
  const float* M = means2d.data_ptr<float>();
  const float* CON = conics.data_ptr<float>();
  const float* COL = colors.data_ptr<float>();
  const float* OP = opacities.data_ptr<float>();
  const float* bg = background.data_ptr<float>();

  Binning b = bin_gaussians(M, depths.data_ptr<float>(), radii.data_ptr<int32_t>(), n, width, height);

  auto image = torch::empty({height, width, 3}, torch::kFloat32);
  auto final_T = torch::empty({height, width}, torch::kFloat32);
  auto n_contrib = torch::empty({height, width}, torch::kInt32);
  float* IMG = image.data_ptr<float>();
  float* FT = final_T.data_ptr<float>();
  int32_t* NC = n_contrib.data_ptr<int32_t>();
  const int num_tiles = b.tiles_x * b.tiles_y;

#pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < num_tiles; ++t) {
    const int tx = t % b.tiles_x, ty = t / b.tiles_x;
    const int64_t start = b.ranges[t], end = b.ranges[t + 1];
    for (int y = ty * TILE; y < std::min<int>((ty + 1) * TILE, height); ++y) {
      for (int x = tx * TILE; x < std::min<int>((tx + 1) * TILE, width); ++x) {
        const float px = x + 0.5f, py = y + 0.5f;
        float T = 1.f, c0 = 0.f, c1 = 0.f, c2 = 0.f;
        int32_t last = 0;
        for (int64_t k = start; k < end; ++k) {
          const int32_t i = b.ids[k];
          float dx, dy, g, alpha;
          if (!eval_alpha(M + i * 2, CON + i * 3, OP[i], px, py, dx, dy, g, alpha)) continue;
          alpha = std::min(alpha, MAX_ALPHA);
          const float nextT = T * (1.f - alpha);
          if (nextT < T_EPS) break;
          const float w = alpha * T;
          c0 += COL[i * 3] * w;
          c1 += COL[i * 3 + 1] * w;
          c2 += COL[i * 3 + 2] * w;
          T = nextT;
          last = (int32_t)(k - start + 1);
        }
        const int64_t p = (int64_t)y * width + x;
        IMG[p * 3] = c0 + T * bg[0];
        IMG[p * 3 + 1] = c1 + T * bg[1];
        IMG[p * 3 + 2] = c2 + T * bg[2];
        FT[p] = T;
        NC[p] = last;
      }
    }
  }

  auto ranges_t = torch::from_blob(b.ranges.data(), {(int64_t)b.ranges.size()}, torch::kInt64).clone();
  auto ids_t = torch::from_blob(b.ids.data(), {(int64_t)b.ids.size()}, torch::kInt32).clone();
  return {image, final_T, n_contrib, ranges_t, ids_t};
}

// Returns: d_means2d [N,2], d_conics [N,3], d_colors [N,3], d_opacities [N], absgrad_means2d [N,2]
std::vector<torch::Tensor> rasterize_backward(torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
                                              torch::Tensor opacities, torch::Tensor background,
                                              torch::Tensor final_T, torch::Tensor n_contrib,
                                              torch::Tensor tile_ranges, torch::Tensor tile_ids,
                                              torch::Tensor grad_image, int64_t width, int64_t height) {
  const int64_t n = means2d.size(0);
  const float* M = means2d.data_ptr<float>();
  const float* CON = conics.data_ptr<float>();
  const float* COL = colors.data_ptr<float>();
  const float* OP = opacities.data_ptr<float>();
  const float* bg = background.data_ptr<float>();
  const float* FT = final_T.data_ptr<float>();
  const int32_t* NC = n_contrib.data_ptr<int32_t>();
  const int64_t* R = tile_ranges.data_ptr<int64_t>();
  const int32_t* IDS = tile_ids.data_ptr<int32_t>();
  grad_image = grad_image.contiguous();
  const float* GI = grad_image.data_ptr<float>();

  const int tiles_x = (width + TILE - 1) / TILE;
  const int tiles_y = (height + TILE - 1) / TILE;
  const int num_tiles = tiles_x * tiles_y;

  // Per-thread accumulation buffers, reduced at the end (no atomics).
  constexpr int G = 11;  // dmx dmy | da db dc | dr dg db | dop | |dmx| |dmy|
  const int nthreads = omp_get_max_threads();
  std::vector<float> buf((size_t)nthreads * n * G, 0.f);

#pragma omp parallel
  {
    float* acc = &buf[(size_t)omp_get_thread_num() * n * G];
#pragma omp for schedule(dynamic, 1)
    for (int t = 0; t < num_tiles; ++t) {
      const int tx = t % tiles_x, ty = t / tiles_x;
      const int64_t start = R[t];
      for (int y = ty * TILE; y < std::min<int>((ty + 1) * TILE, height); ++y) {
        for (int x = tx * TILE; x < std::min<int>((tx + 1) * TILE, width); ++x) {
          const int64_t p = (int64_t)y * width + x;
          const float px = x + 0.5f, py = y + 0.5f;
          const float T_final = FT[p];
          float T = T_final;
          const float dLdp[3] = {GI[p * 3], GI[p * 3 + 1], GI[p * 3 + 2]};
          const float bg_dot = bg[0] * dLdp[0] + bg[1] * dLdp[1] + bg[2] * dLdp[2];
          float accum[3] = {0.f, 0.f, 0.f};
          float last_alpha = 0.f;
          float last_color[3] = {0.f, 0.f, 0.f};

          for (int64_t k = start + NC[p] - 1; k >= start; --k) {
            const int32_t i = IDS[k];
            float dx, dy, g, alpha;
            if (!eval_alpha(M + i * 2, CON + i * 3, OP[i], px, py, dx, dy, g, alpha)) continue;
            const bool clamped = alpha > MAX_ALPHA;
            alpha = std::min(alpha, MAX_ALPHA);
            T = T / (1.f - alpha);
            const float w = alpha * T;
            float* a = acc + (size_t)i * G;

            float dL_dalpha = 0.f;
            for (int ch = 0; ch < 3; ++ch) {
              const float c = COL[i * 3 + ch];
              accum[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum[ch];
              last_color[ch] = c;
              dL_dalpha += (c - accum[ch]) * dLdp[ch];
              a[5 + ch] += w * dLdp[ch];
            }
            dL_dalpha *= T;
            last_alpha = alpha;
            dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot;
            if (clamped) continue;  // alpha = min(op*g, 0.99) has zero gradient when clamped

            a[8] += g * dL_dalpha;  // d opacity
            const float dL_dpower = OP[i] * g * dL_dalpha;
            const float con_a = CON[i * 3], con_b = CON[i * 3 + 1], con_c = CON[i * 3 + 2];
            // d power / d dx = -(a dx + b dy), and d(dx)/d(mean.x) = 1
            const float gmx = dL_dpower * (-(con_a * dx + con_b * dy));
            const float gmy = dL_dpower * (-(con_c * dy + con_b * dx));
            a[0] += gmx;
            a[1] += gmy;
            a[9] += std::fabs(gmx);
            a[10] += std::fabs(gmy);
            a[2] += dL_dpower * (-0.5f * dx * dx);
            a[3] += dL_dpower * (-dx * dy);
            a[4] += dL_dpower * (-0.5f * dy * dy);
          }
        }
      }
    }
  }

  auto grads = torch::zeros({n, G}, torch::kFloat32);
  float* out = grads.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < n * G; ++j) {
    float s = 0.f;
    for (int th = 0; th < nthreads; ++th) s += buf[(size_t)th * n * G + j];
    out[j] = s;
  }
  return {grads.slice(1, 0, 2).contiguous(), grads.slice(1, 2, 5).contiguous(), grads.slice(1, 5, 8).contiguous(),
          grads.select(1, 8).contiguous(), grads.slice(1, 9, 11).contiguous()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rasterize_forward, "3DGS CPU rasterize forward");
  m.def("backward", &rasterize_backward, "3DGS CPU rasterize backward");
}
