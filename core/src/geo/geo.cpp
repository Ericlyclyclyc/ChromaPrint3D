#include "chromaprint3d/voxel.h"
#include "chromaprint3d/mesh.h"
#include "chromaprint3d/error.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ChromaPrint3D {
namespace {
inline size_t GridIndex(int w, int h, int l, int width, int height, int layers) {
    return (static_cast<size_t>(h) * static_cast<size_t>(width) + static_cast<size_t>(w)) *
               static_cast<size_t>(layers) +
           static_cast<size_t>(l);
}

uint32_t CheckedU32Index(std::size_t value) {
    if (value > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw InputError("Mesh vertex index exceeds uint32_t range");
    }
    return static_cast<uint32_t>(value);
}

uint32_t CheckedU32Coord(int value) {
    if (value < 0) { throw InputError("Mesh coordinate is negative"); }
    return static_cast<uint32_t>(value);
}

inline uint64_t UndirectedEdgeKey(uint32_t a, uint32_t b) {
    if (b < a) { std::swap(a, b); }
    return (static_cast<uint64_t>(a) << 32) | b;
}

// Grid cell of the solid voxel bounded by triangle `tri` on the side adjacent
// to the unit edge (va, vb). Faces meeting at a diagonal-contact edge are
// paired by this cell so each solid keeps its own topological shell.
std::array<int, 3> OwnerCellAtEdge(const Vec3u& tri, const std::vector<Vec3u>& grid_pts,
                                   uint32_t va, uint32_t vb) {
    const Vec3u& g0 = grid_pts[tri.x];
    const Vec3u& g1 = grid_pts[tri.y];
    const Vec3u& g2 = grid_pts[tri.z];
    const int P[3][3] = {
        {static_cast<int>(g0.x), static_cast<int>(g0.y), static_cast<int>(g0.z)},
        {static_cast<int>(g1.x), static_cast<int>(g1.y), static_cast<int>(g1.z)},
        {static_cast<int>(g2.x), static_cast<int>(g2.y), static_cast<int>(g2.z)},
    };

    int d = 0;
    for (int k = 0; k < 3; ++k) {
        if (P[0][k] == P[1][k] && P[1][k] == P[2][k]) {
            d = k;
            break;
        }
    }
    const int u = (d + 1) % 3;
    const int v = (d + 2) % 3;

    // CCW in the (u, v) plane means the face normal points along +d, so the
    // solid voxel sits on the (slice - 1) side.
    const int64_t signed2 =
        static_cast<int64_t>(P[1][u] - P[0][u]) * static_cast<int64_t>(P[2][v] - P[0][v]) -
        static_cast<int64_t>(P[1][v] - P[0][v]) * static_cast<int64_t>(P[2][u] - P[0][u]);

    const int A[3] = {static_cast<int>(grid_pts[va].x), static_cast<int>(grid_pts[va].y),
                      static_cast<int>(grid_pts[va].z)};
    const int B[3] = {static_cast<int>(grid_pts[vb].x), static_cast<int>(grid_pts[vb].y),
                      static_cast<int>(grid_pts[vb].z)};

    std::array<int, 3> cell{};
    cell[d] = (signed2 > 0) ? P[0][d] - 1 : P[0][d];
    if (A[u] != B[u]) {
        cell[u]        = std::min(A[u], B[u]);
        const int m    = A[v];
        const int gsum = P[0][v] + P[1][v] + P[2][v];
        cell[v]        = (gsum > 3 * m) ? m : m - 1;
    } else {
        cell[v]        = std::min(A[v], B[v]);
        const int m    = A[u];
        const int gsum = P[0][u] + P[1][u] + P[2][u];
        cell[u]        = (gsum > 3 * m) ? m : m - 1;
    }
    return cell;
}

// Voxel solids that touch only along a grid edge share that edge between four
// faces after positional vertex welding, which slicers report as non-manifold
// edges. For each such edge one solid keeps the straight edge while the other
// re-routes its two faces through a midpoint nudged slightly into its own
// voxel, so every edge ends up used by exactly two faces again. Unlike
// splitting the vertex fans this stays correct when the two solids are also
// connected elsewhere (pinched shells sandwiched between common layers).
void SplitNonManifoldContacts(Mesh& mesh, const std::vector<Vec3u>& grid_pts, float offset_mm) {
    if (mesh.indices.empty()) { return; }

    std::vector<uint64_t> edge_keys;
    edge_keys.reserve(mesh.indices.size() * 3);
    for (const Vec3u& t : mesh.indices) {
        edge_keys.push_back(UndirectedEdgeKey(t.x, t.y));
        edge_keys.push_back(UndirectedEdgeKey(t.y, t.z));
        edge_keys.push_back(UndirectedEdgeKey(t.z, t.x));
    }
    std::sort(edge_keys.begin(), edge_keys.end());

    std::unordered_set<uint64_t> nm_edges;
    for (size_t i = 0; i < edge_keys.size();) {
        size_t j = i + 1;
        while (j < edge_keys.size() && edge_keys[j] == edge_keys[i]) { ++j; }
        if (j - i > 2) { nm_edges.insert(edge_keys[i]); }
        i = j;
    }
    edge_keys.clear();
    edge_keys.shrink_to_fit();
    if (nm_edges.empty()) { return; }

    struct NmFace {
        uint32_t tri;
        std::array<int, 3> owner;
    };
    std::unordered_map<uint64_t, std::vector<NmFace>> nm_faces;
    nm_faces.reserve(nm_edges.size());
    for (uint32_t ti = 0; ti < mesh.indices.size(); ++ti) {
        const Vec3u& t       = mesh.indices[ti];
        const uint32_t vs[3] = {t.x, t.y, t.z};
        for (int e = 0; e < 3; ++e) {
            const uint32_t a   = vs[e];
            const uint32_t b   = vs[(e + 1) % 3];
            const uint64_t key = UndirectedEdgeKey(a, b);
            if (nm_edges.count(key) != 0) {
                nm_faces[key].push_back({ti, OwnerCellAtEdge(t, grid_pts, a, b)});
            }
        }
    }

    // A subdivision may move an edge of another tracked non-manifold edge onto
    // the appended triangle; its face record has to follow.
    auto retarget = [&](uint32_t a, uint32_t b, uint32_t from, uint32_t to) {
        auto it = nm_faces.find(UndirectedEdgeKey(a, b));
        if (it == nm_faces.end()) { return; }
        for (NmFace& f : it->second) {
            if (f.tri == from) {
                f.tri = to;
                return;
            }
        }
    };

    size_t rerouted = 0;
    for (auto& [key, faces] : nm_faces) {
        const uint32_t va = static_cast<uint32_t>(key >> 32);
        const uint32_t vb = static_cast<uint32_t>(key & 0xffffffffu);

        // Group the faces by the solid voxel they bound at this edge.
        std::vector<std::pair<std::array<int, 3>, std::vector<size_t>>> groups;
        for (size_t fi = 0; fi < faces.size(); ++fi) {
            auto it = std::find_if(groups.begin(), groups.end(),
                                   [&](const auto& g) { return g.first == faces[fi].owner; });
            if (it == groups.end()) {
                groups.push_back({faces[fi].owner, {fi}});
            } else {
                it->second.push_back(fi);
            }
        }

        const Vec3u& A       = grid_pts[va];
        const Vec3u& B       = grid_pts[vb];
        const int a_coord[3] = {static_cast<int>(A.x), static_cast<int>(A.y),
                                static_cast<int>(A.z)};
        const int b_coord[3] = {static_cast<int>(B.x), static_cast<int>(B.y),
                                static_cast<int>(B.z)};

        // The first solid keeps the straight edge; every other one detours its
        // face pair through a midpoint nudged into its own voxel interior.
        for (size_t gi = 1; gi < groups.size(); ++gi) {
            if (groups[gi].second.size() != 2) {
                spdlog::warn("SplitNonManifoldContacts: edge {}-{} bounds {} faces of one voxel, "
                             "skipped",
                             va, vb, groups[gi].second.size());
                continue;
            }

            const auto& pa = mesh.vertices[va];
            const auto& pb = mesh.vertices[vb];
            float m[3]     = {(pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f, (pa.z + pb.z) * 0.5f};
            for (int k = 0; k < 3; ++k) {
                if (a_coord[k] != b_coord[k]) { continue; } // edge axis stays at the midpoint
                m[k] += (groups[gi].first[k] == a_coord[k]) ? offset_mm : -offset_mm;
            }
            const uint32_t mid = CheckedU32Index(mesh.vertices.size());
            mesh.vertices.emplace_back(m[0], m[1], m[2]);

            for (const size_t fi : groups[gi].second) {
                const uint32_t ti = faces[fi].tri;
                Vec3u t           = mesh.indices[ti];
                int rot           = 0;
                while (rot < 3 &&
                       !((t.x == va && t.y == vb) || (t.x == vb && t.y == va))) {
                    t = Vec3u{t.y, t.z, t.x};
                    ++rot;
                }
                if (rot == 3) { continue; } // stale record, should not happen

                const uint32_t nt = CheckedU32Index(mesh.indices.size());
                mesh.indices[ti]  = Vec3u{t.x, mid, t.z};
                mesh.indices.emplace_back(mid, t.y, t.z);
                retarget(t.y, t.z, ti, nt);
            }
            ++rerouted;
        }
    }

    spdlog::debug("SplitNonManifoldContacts: {} non-manifold edges, {} face pairs rerouted",
                  nm_edges.size(), rerouted);
}
} // namespace

bool VoxelGrid::Get(int w, int h, int l) const {
    if (w < 0 || h < 0 || l < 0 || w >= width || h >= height || l >= num_layers) { return false; }
    const size_t idx = GridIndex(w, h, l, width, height, num_layers);
    if (idx >= ooc.size()) { return false; }
    return ooc[idx] != 0;
}

bool VoxelGrid::Set(int w, int h, int l, bool v) {
    if (w < 0 || h < 0 || l < 0 || w >= width || h >= height || l >= num_layers) { return false; }
    const size_t idx = GridIndex(w, h, l, width, height, num_layers);
    if (idx >= ooc.size()) { return false; }
    ooc[idx] = v ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    return true;
}

ModelIR ModelIR::Build(const RecipeMap& recipe_map, const ColorDB& db,
                       const BuildModelIRConfig& cfg) {
    if (recipe_map.width <= 0 || recipe_map.height <= 0) {
        throw InputError("RecipeMap size is invalid");
    }
    if (recipe_map.color_layers < 0 || recipe_map.num_channels < 0) {
        throw InputError("RecipeMap layers or channels are invalid");
    }

    const int width        = recipe_map.width;
    const int height       = recipe_map.height;
    const int color_layers = recipe_map.color_layers;

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (color_layers > 0) {
        const size_t expected = pixel_count * static_cast<size_t>(color_layers);
        if (recipe_map.recipes.size() < expected) {
            throw InputError("RecipeMap recipes size mismatch");
        }
    }
    if (!recipe_map.mask.empty() && recipe_map.mask.size() < pixel_count) {
        throw InputError("RecipeMap mask size mismatch");
    }
    if (!cfg.base_only_mask.empty() && cfg.base_only_mask.size() != pixel_count) {
        throw InputError("base_only_mask size mismatch");
    }

    int num_channels = recipe_map.num_channels;
    if (num_channels <= 0) { num_channels = static_cast<int>(db.NumChannels()); }
    if (num_channels <= 0) { throw InputError("num_channels is invalid"); }
    if (db.NumChannels() > 0 && static_cast<int>(db.NumChannels()) != num_channels) {
        throw ConfigError("RecipeMap num_channels does not match ColorDB");
    }

    const int base_layers = (cfg.base_layers >= 0) ? cfg.base_layers : db.base_layers;
    if (base_layers < 0) { throw InputError("base_layers is invalid"); }

    const int base_channel_idx = db.base_channel_idx;
    if (base_layers > 0 && (base_channel_idx < 0 || base_channel_idx >= num_channels)) {
        throw InputError("base_channel_idx is out of range");
    }

    const bool double_sided = cfg.double_sided;
    const int base_start    = double_sided ? color_layers : 0;
    const int total_layers  = base_start + base_layers + color_layers;
    if (total_layers < 0) { throw InputError("total_layers is invalid"); }

    ModelIR result;
    result.name             = recipe_map.name;
    result.width            = width;
    result.height           = height;
    result.color_layers     = color_layers;
    result.base_layers      = base_layers;
    result.base_channel_idx = base_channel_idx;
    result.palette          = db.palette;
    if (result.palette.empty()) {
        result.palette.resize(static_cast<size_t>(num_channels));
    } else if (static_cast<int>(result.palette.size()) != num_channels) {
        throw ConfigError("palette size does not match num_channels");
    }
    const bool has_base_grid = base_layers > 0;
    const int base_grid_idx  = has_base_grid ? num_channels : -1;
    result.voxel_grids.resize(static_cast<size_t>(num_channels + (has_base_grid ? 1 : 0)));

    for (int ch = 0; ch < num_channels; ++ch) {
        VoxelGrid& grid  = result.voxel_grids[static_cast<size_t>(ch)];
        grid.width       = width;
        grid.height      = height;
        grid.num_layers  = total_layers;
        grid.channel_idx = ch;
        grid.ooc.assign(pixel_count * static_cast<size_t>(total_layers), 0);
    }
    if (has_base_grid) {
        VoxelGrid& grid  = result.voxel_grids[static_cast<size_t>(base_grid_idx)];
        grid.width       = width;
        grid.height      = height;
        grid.num_layers  = total_layers;
        grid.channel_idx = base_grid_idx;
        grid.ooc.assign(pixel_count * static_cast<size_t>(total_layers), 0);
    }

    const bool has_mask           = !recipe_map.mask.empty();
    const bool has_base_only_mask = !cfg.base_only_mask.empty();
    const uint8_t* base_only_mask = has_base_only_mask ? cfg.base_only_mask.data() : nullptr;
    VoxelGrid* base_grid =
        has_base_grid ? &result.voxel_grids[static_cast<size_t>(base_grid_idx)] : nullptr;

#pragma omp parallel for schedule(dynamic, 64)
    for (int r = 0; r < height; ++r) {
        const int vh = cfg.flip_y ? (height - 1 - r) : r;
        for (int c = 0; c < width; ++c) {
            const size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
            if (has_mask && recipe_map.mask[idx] == 0) { continue; }

            if (has_base_grid && base_grid) {
                for (int l = 0; l < base_layers; ++l) {
                    const int base_layer = base_start + l;
                    const size_t offset = GridIndex(c, vh, base_layer, width, height, total_layers);
                    if (offset < base_grid->ooc.size()) { base_grid->ooc[offset] = 1; }
                }
            }

            if (color_layers == 0) { continue; }
            const uint8_t* recipe = recipe_map.RecipeAt(r, c);
            if (!recipe) { continue; }

            const bool route_base_channel =
                has_base_grid && has_base_only_mask && base_only_mask && base_only_mask[idx] != 0;

            for (int layer = 0; layer < color_layers; ++layer) {
                const int mapped_layer = (recipe_map.layer_order == LayerOrder::Top2Bottom)
                                             ? (color_layers - 1 - layer)
                                             : layer;
                const int stored_layer = base_start + base_layers + mapped_layer;
                const int channel_idx  = static_cast<int>(recipe[layer]);
                if (channel_idx < 0 || channel_idx >= num_channels) { continue; }

                VoxelGrid& grid =
                    (route_base_channel && channel_idx == base_channel_idx && base_grid)
                        ? *base_grid
                        : result.voxel_grids[static_cast<size_t>(channel_idx)];
                const size_t offset = GridIndex(c, vh, stored_layer, width, height, total_layers);
                if (offset < grid.ooc.size()) { grid.ooc[offset] = 1; }
            }

            if (!double_sided) { continue; }
            for (int layer = 0; layer < color_layers; ++layer) {
                const int mapped_layer = (recipe_map.layer_order == LayerOrder::Top2Bottom)
                                             ? (color_layers - 1 - layer)
                                             : layer;
                const int stored_layer = (base_start - 1) - mapped_layer;
                const int channel_idx  = static_cast<int>(recipe[layer]);
                if (channel_idx < 0 || channel_idx >= num_channels) { continue; }

                VoxelGrid& grid =
                    (route_base_channel && channel_idx == base_channel_idx && base_grid)
                        ? *base_grid
                        : result.voxel_grids[static_cast<size_t>(channel_idx)];
                const size_t offset = GridIndex(c, vh, stored_layer, width, height, total_layers);
                if (offset < grid.ooc.size()) { grid.ooc[offset] = 1; }
            }
        }
    }

    spdlog::info("ModelIR::Build: {}x{}, {} grids, total_layers={}", result.width, result.height,
                 result.voxel_grids.size(), total_layers);
    return result;
}

Mesh Mesh::Build(const VoxelGrid& voxel_grid, const BuildMeshConfig& cfg) {
    Mesh mesh;
    const int width  = voxel_grid.width;
    const int height = voxel_grid.height;
    const int layers = voxel_grid.num_layers;
    if (width <= 0 || height <= 0 || layers <= 0) {
        spdlog::debug("Mesh::Build: returning empty mesh (dimensions {}x{}x{})", width, height,
                      layers);
        return mesh;
    }
    if (cfg.pixel_mm <= 0.0f || cfg.layer_height_mm <= 0.0f) {
        throw InputError("BuildMeshConfig values must be positive");
    }

    const size_t expected =
        static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(layers);
    if (voxel_grid.ooc.size() < expected) { throw InputError("VoxelGrid ooc size mismatch"); }

    // Vertex welding packs grid coordinates into 21 bits per axis.
    constexpr int kMaxGridDim = (1 << 21) - 1;
    if (width > kMaxGridDim || height > kMaxGridDim || layers > kMaxGridDim) {
        throw InputError("VoxelGrid dimension exceeds vertex welding limit");
    }

    const float px    = cfg.pixel_mm;
    const float pz    = cfg.layer_height_mm;
    const int dims[3] = {width, height, layers};
    const uint8_t* ooc = voxel_grid.ooc.data();

    // ── Pass 1: greedy merge per (axis, slice), run in parallel ─────────────
    // Each task owns one slice of one axis: it builds its own face mask, runs
    // the greedy rectangle merge and emits triangles as integer grid-coordinate
    // triples into a private buffer. The shared vertex table is only built in
    // pass 2, so tasks are fully independent. Tasks are flattened over
    // (d, slice) so the loop has width+height+layers+3 items — enough
    // granularity to keep all threads busy even when a single channel (e.g.
    // the base grid) dominates the volume.
    const int task_counts[3] = {dims[0] + 1, dims[1] + 1, dims[2] + 1};
    const int total_tasks    = task_counts[0] + task_counts[1] + task_counts[2];
    std::vector<std::vector<Vec3u>> task_tris(static_cast<size_t>(total_tasks));

    std::atomic<bool> has_error{false};
    std::string error_message;

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic)
#endif
    for (int t = 0; t < total_tasks; ++t) {
        if (has_error.load(std::memory_order_relaxed)) { continue; }
        try {
            int d     = 0;
            int slice = t;
            while (slice >= task_counts[d]) {
                slice -= task_counts[d];
                ++d;
            }
            const int u = (d + 1) % 3;
            const int v = (d + 2) % 3;

            // All calls below stay in-bounds by construction (see mask fill),
            // so the per-voxel bounds check is omitted on this hot path.
            auto is_filled = [&](int x, int y, int z) -> bool {
                const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                    static_cast<size_t>(x)) *
                                       static_cast<size_t>(layers) +
                                   static_cast<size_t>(z);
                return ooc[idx] != 0;
            };

            std::vector<Vec3u>& tris = task_tris[static_cast<size_t>(t)];
            std::vector<int8_t> mask(static_cast<size_t>(dims[u]) * static_cast<size_t>(dims[v]),
                                     0);

            int x[3] = {0, 0, 0};
            int n    = 0;
            for (x[v] = 0; x[v] < dims[v]; ++x[v]) {
                for (x[u] = 0; x[u] < dims[u]; ++x[u]) {
                    bool a = false;
                    bool b = false;
                    if (slice > 0) {
                        x[d] = slice - 1;
                        a    = is_filled(x[0], x[1], x[2]);
                    }
                    if (slice < dims[d]) {
                        x[d] = slice;
                        b    = is_filled(x[0], x[1], x[2]);
                    }
                    mask[static_cast<size_t>(n++)] =
                        (a != b) ? (a ? int8_t{1} : int8_t{-1}) : int8_t{0};
                }
            }

            // Emits one merged rectangle [i, i+w) x [j, j+h) in plane axis d at
            // `slice`, triangulated so that every edge on the rectangle outline
            // is a unit grid segment. Adjacent faces (coplanar or
            // perpendicular) therefore always share whole edges at identical
            // vertex indices, which is what keeps the exported mesh free of
            // T-junction open edges while preserving the greedy merge (triangle
            // count stays proportional to the perimeter).
            //
            // The interior is fanned from per-block centers: the rectangle is
            // cut along its longer side (in mm) into the fewest blocks that
            // keep every fan triangle's altitude above ~30 um, each block
            // fanned from an interior grid point. Block-to-block cuts are
            // single full-length segments shared by the two adjacent fans, so
            // the triangle count stays ~2(w + h) + blocks. Fanning the whole
            // outline from one corner instead would create near-degenerate
            // slivers (altitude ~ unit/perimeter) on large rectangles, which
            // collapse at slicer precision and stall slicing.
            auto emit_rect = [&](int i, int j, int w, int h, bool positive) {
                auto P = [&](int a, int b) {
                    int p[3] = {0, 0, 0};
                    p[d]     = slice;
                    p[u]     = a;
                    p[v]     = b;
                    return Vec3u{CheckedU32Coord(p[0]), CheckedU32Coord(p[1]),
                                 CheckedU32Coord(p[2])};
                };
                auto tri = [&](const Vec3u& a, const Vec3u& b, const Vec3u& c) {
                    tris.push_back(a);
                    if (positive) {
                        tris.push_back(b);
                        tris.push_back(c);
                    } else {
                        tris.push_back(c);
                        tris.push_back(b);
                    }
                };

                // Single-row/column strips: the plain unit-step ladder is
                // already minimal and its right triangles have bounded aspect.
                if (w == 1 || h == 1) {
                    for (int r = 0; r < h; ++r) {
                        for (int s = 0; s < w; ++s) {
                            tri(P(i + s, j + r), P(i + s + 1, j + r), P(i + s, j + r + 1));
                            tri(P(i + s + 1, j + r), P(i + s + 1, j + r + 1),
                                P(i + s, j + r + 1));
                        }
                    }
                    return;
                }

                // Cut along the longer side (in mm). The worst fan triangle
                // sits on a long-side unit segment near a block corner:
                // altitude ~= step_along * across_mm / (block_len_mm *
                // sqrt(2)). Choose the fewest blocks that keep that above
                // ~30 um.
                const float step_u = (u == 2) ? pz : px;
                const float step_v = (v == 2) ? pz : px;
                const bool along_u =
                    static_cast<float>(w) * step_u >= static_cast<float>(h) * step_v;
                const int len          = along_u ? w : h;
                const int across       = along_u ? h : w;
                const float step_along = along_u ? step_u : step_v;
                const float len_mm     = static_cast<float>(len) * step_along;
                const float across_mm  = static_cast<float>(across) * (along_u ? step_v : step_u);
                constexpr float kMinAltitudeMm = 0.03f * 1.4142135f;
                const int blocks               = std::clamp(
                    static_cast<int>(std::ceil(len_mm * kMinAltitudeMm / (step_along * across_mm))),
                    1, std::max(1, len / 2));

                auto Q = [&](int al, int ac) {
                    return along_u ? P(i + al, j + ac) : P(i + ac, j + al);
                };

                std::vector<std::pair<int, int>> loop;
                for (int k = 0; k < blocks; ++k) {
                    const int al0 = static_cast<int>(static_cast<int64_t>(len) * k / blocks);
                    const int al1 = static_cast<int>(static_cast<int64_t>(len) * (k + 1) / blocks);

                    // Block boundary, counter-clockwise in (along, across).
                    // Outline sides use unit steps; internal block cuts are
                    // single segments shared verbatim by the neighbouring
                    // block's fan.
                    loop.clear();
                    for (int a = al0; a < al1; ++a) { loop.emplace_back(a, 0); }
                    if (al1 == len) {
                        for (int c = 0; c < across; ++c) { loop.emplace_back(al1, c); }
                    } else {
                        loop.emplace_back(al1, 0);
                    }
                    for (int a = al1; a > al0; --a) { loop.emplace_back(a, across); }
                    if (al0 == 0) {
                        for (int c = across; c > 0; --c) { loop.emplace_back(al0, c); }
                    } else {
                        loop.emplace_back(al0, across);
                    }

                    const Vec3u apex = Q((al0 + al1) / 2, across / 2);
                    for (size_t s = 0; s < loop.size(); ++s) {
                        const auto& p0 = loop[s];
                        const auto& p1 = loop[(s + 1) % loop.size()];
                        // Swapping the axes mirrors the plane, so restore the
                        // winding.
                        if (along_u) {
                            tri(apex, Q(p0.first, p0.second), Q(p1.first, p1.second));
                        } else {
                            tri(apex, Q(p1.first, p1.second), Q(p0.first, p0.second));
                        }
                    }
                }
            };

            n = 0;
            for (int j = 0; j < dims[v]; ++j) {
                for (int i = 0; i < dims[u];) {
                    const int c = mask[static_cast<size_t>(n)];
                    if (c == 0) {
                        ++i;
                        ++n;
                        continue;
                    }

                    int w = 1;
                    while (i + w < dims[u] && mask[static_cast<size_t>(n + w)] == c) { ++w; }

                    int h = 1;
                    for (; j + h < dims[v]; ++h) {
                        bool ok = true;
                        for (int k = 0; k < w; ++k) {
                            if (mask[static_cast<size_t>(n + k + h * dims[u])] != c) {
                                ok = false;
                                break;
                            }
                        }
                        if (!ok) { break; }
                    }

                    emit_rect(i, j, w, h, c > 0);

                    for (int dy = 0; dy < h; ++dy) {
                        for (int dx = 0; dx < w; ++dx) {
                            mask[static_cast<size_t>(n + dx + dy * dims[u])] = 0;
                        }
                    }

                    i += w;
                    n += w;
                }
            }
        } catch (const std::exception& e) {
#ifdef _OPENMP
#    pragma omp critical(mesh_build_slice_error)
#endif
            {
                if (!has_error.load(std::memory_order_relaxed)) {
                    error_message = e.what();
                    has_error.store(true, std::memory_order_relaxed);
                }
            }
        } catch (...) {
#ifdef _OPENMP
#    pragma omp critical(mesh_build_slice_error)
#endif
            {
                if (!has_error.load(std::memory_order_relaxed)) {
                    error_message = "Unknown error in greedy mesh slice extraction";
                    has_error.store(true, std::memory_order_relaxed);
                }
            }
        }
    }

    if (has_error.load(std::memory_order_relaxed)) {
        throw InputError(error_message.empty() ? "Greedy mesh slice extraction failed"
                                               : error_message);
    }

    // ── Pass 2: serial vertex welding in deterministic (axis, slice) order ──
    // Iterating the task buffers in flattened order reproduces the exact
    // triangle and first-use vertex order of the previous single-threaded
    // implementation, so the output mesh is bit-identical.
    const std::size_t estimated_surface =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 2 +
        static_cast<std::size_t>(width) * static_cast<std::size_t>(layers) * 2 +
        static_cast<std::size_t>(height) * static_cast<std::size_t>(layers) * 2;
    std::size_t total_tri_count = 0;
    for (const auto& tris : task_tris) { total_tri_count += tris.size() / 3; }

    mesh.vertices.reserve(estimated_surface);
    mesh.indices.reserve(total_tri_count);
    // Integer grid coordinates per vertex, kept for topology post-processing.
    std::vector<Vec3u> grid_pts;
    grid_pts.reserve(estimated_surface);

    // splitmix64 finalizer over the packed 21-bit-per-axis key: much stronger
    // mixing than the previous per-component xor hash, which clustered badly
    // on regular grid coordinates.
    struct PackedVertexHash {
        size_t operator()(uint64_t key) const {
            key += 0x9e3779b97f4a7c15ull;
            key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
            key = (key ^ (key >> 27)) * 0x94d049bb133111ebull;
            return static_cast<size_t>(key ^ (key >> 31));
        }
    };
    std::unordered_map<uint64_t, uint32_t, PackedVertexHash> vertex_map;
    vertex_map.reserve(estimated_surface);

    auto add_vertex = [&](const Vec3u& v) {
        const uint64_t key = (static_cast<uint64_t>(v.x) << 42) |
                             (static_cast<uint64_t>(v.y) << 21) | static_cast<uint64_t>(v.z);
        auto it = vertex_map.find(key);
        if (it != vertex_map.end()) { return it->second; }
        float z = static_cast<float>(v.z) * pz;
        for (const auto& gap : cfg.interface_offsets) {
            if (gap.z_index >= 0 && v.z == static_cast<uint32_t>(gap.z_index)) {
                z += gap.offset_mm;
                break;
            }
        }
        const uint32_t idx = CheckedU32Index(mesh.vertices.size());
        mesh.vertices.emplace_back(static_cast<float>(v.x) * px, static_cast<float>(v.y) * px, z);
        grid_pts.push_back(v);
        vertex_map.emplace(key, idx);
        return idx;
    };

    for (auto& tris : task_tris) {
        for (std::size_t k = 0; k + 2 < tris.size(); k += 3) {
            const uint32_t ia = add_vertex(tris[k]);
            const uint32_t ib = add_vertex(tris[k + 1]);
            const uint32_t ic = add_vertex(tris[k + 2]);
            mesh.indices.emplace_back(ia, ib, ic);
        }
        // Free each buffer as soon as it is welded to cap peak memory.
        std::vector<Vec3u>{}.swap(tris);
    }

    SplitNonManifoldContacts(mesh, grid_pts, 0.125f * std::min(px, pz));

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        spdlog::warn("Mesh::Build: ch={} produced empty mesh (vertices={}, triangles={})",
                     voxel_grid.channel_idx, mesh.vertices.size(), mesh.indices.size());
    } else {
        spdlog::debug("Mesh::Build: ch={}, vertices={}, triangles={}", voxel_grid.channel_idx,
                      mesh.vertices.size(), mesh.indices.size());
    }
    return mesh;
}

} // namespace ChromaPrint3D
