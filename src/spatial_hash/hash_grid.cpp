#include <ipc/spatial_hash/hash_grid.hpp>

#include <iostream>

#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/concurrent_unordered_set.h>
#endif
#include <tbb/parallel_sort.h> // Still use this even if TBB is disabled

#ifdef IPC_TOOLKIT_WITH_LOGGER
#include <ipc/utils/logger.hpp>
#endif
#include <igl/Timer.h>
namespace ipc {

bool AABB::are_overlaping(const AABB& a, const AABB& b)
{
    // https://bit.ly/2ZP3tW4
    assert(a.dim == b.dim);
    return (abs(a.center.x() - b.center.x())
            <= (a.half_extent.x() + b.half_extent.x()))
        && (abs(a.center.y() - b.center.y())
            <= (a.half_extent.y() + b.half_extent.y()))
        && (a.dim == 2
            || abs(a.center.z() - b.center.z())
                <= (a.half_extent.z() + b.half_extent.z()));
};

void HashGrid::resize(
    Eigen::VectorX3d min, Eigen::VectorX3d max, double cellSize)
{
    clear();
    assert(cellSize != 0.0);
    m_cellSize = cellSize;
    m_domainMin = min;
    m_domainMax = max;
    m_gridSize = ((max - min) / m_cellSize).array().ceil().cast<int>().max(1);
#ifdef IPC_TOOLKIT_WITH_LOGGER
    logger().debug(
        "hash-grid resized with a size of {:d}x{:d}x{:d}", m_gridSize[0],
        m_gridSize[1], m_gridSize.size() == 3 ? m_gridSize[2] : 1);
#endif
}

/// @brief Compute an AABB around a given 2D mesh.
void calculate_mesh_extents(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    Eigen::VectorX3d& lower_bound,
    Eigen::VectorX3d& upper_bound)
{
    Eigen::VectorX3d lower_bound_t0 = vertices_t0.colwise().minCoeff();
    Eigen::VectorX3d upper_bound_t0 = vertices_t0.colwise().maxCoeff();

    Eigen::VectorX3d lower_bound_t1 = vertices_t1.colwise().minCoeff();
    Eigen::VectorX3d upper_bound_t1 = vertices_t1.colwise().maxCoeff();

    lower_bound = lower_bound_t0.cwiseMin(lower_bound_t1);
    upper_bound = upper_bound_t0.cwiseMax(upper_bound_t1);
}

/// @brief Compute the average edge length of a mesh.
double average_edge_length(
    const Eigen::MatrixXd& V_t0,
    const Eigen::MatrixXd& V_t1,
    const Eigen::MatrixXi& E)
{
    double avg = 0;
    for (unsigned i = 0; i < E.rows(); ++i) {
        avg += (V_t0.row(E(i, 0)) - V_t0.row(E(i, 1))).norm();
        avg += (V_t1.row(E(i, 0)) - V_t1.row(E(i, 1))).norm();
    }
    return avg / (2 * E.rows());
}

/// @brief Compute the average displacement length.
double average_displacement_length(const Eigen::MatrixXd& displacements)
{
    return displacements.rowwise().norm().sum() / displacements.rows();
}

void HashGrid::resize(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const double inflation_radius)
{
    Eigen::VectorX3d mesh_min, mesh_max;
    calculate_mesh_extents(vertices_t0, vertices_t1, mesh_min, mesh_max);
    double edge_len = average_edge_length(vertices_t0, vertices_t1, edges);
    double disp_len = average_displacement_length(vertices_t1 - vertices_t0);
    // double cell_size = 2 * std::max(edge_len, disp_len) + inflation_radius;
    double cell_size = 2 * edge_len + inflation_radius;
    this->resize(
        mesh_min.array() - inflation_radius,
        mesh_max.array() + inflation_radius, cell_size);
}

/// @brief Compute a AABB for a vertex moving through time (i.e. temporal edge).
void calculate_vertex_extents(
    const Eigen::VectorX3d& vertex_t0,
    const Eigen::VectorX3d& vertex_t1,
    Eigen::VectorX3d& lower_bound,
    Eigen::VectorX3d& upper_bound)
{
    lower_bound = vertex_t0.cwiseMin(vertex_t1);
    upper_bound = vertex_t0.cwiseMax(vertex_t1);
}

void HashGrid::addVertex(
    const Eigen::VectorX3d& vertex_t0,
    const Eigen::VectorX3d& vertex_t1,
    const long index,
    const double inflation_radius)
{
    Eigen::VectorX3d lower_bound, upper_bound;
    calculate_vertex_extents(vertex_t0, vertex_t1, lower_bound, upper_bound);
    this->addElement(
        AABB(
            lower_bound.array() - inflation_radius,
            upper_bound.array() + inflation_radius),
        index, m_vertexItems);
}

void HashGrid::addVertices(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const double inflation_radius)
{
    igl::Timer timer;
    timer.start();
    assert(vertices_t0.rows() == vertices_t1.rows());

#ifndef IPC_TOOLKIT_SPATIAL_HASH_USE_TBB
    for (long i = 0; i < vertices_t0.rows(); i++) {
        addVertex(vertices_t0.row(i), vertices_t1.row(i), i, inflation_radius);
    }
#else
    // A vector of AABBs and where they start in the vertex items
    std::vector<std::pair<AABB, int>> aabbs;
    aabbs.reserve(vertices_t0.rows());

    // TODO
    // blocked range
    // hash grid per thread
    // add each vertex to
    // combine hashgrids

    size_t num_items = m_vertexItems.size();
    for (long i = 0; i < vertices_t0.rows(); i++) {
        Eigen::VectorX3d lower_bound, upper_bound;
        calculate_vertex_extents(
            vertices_t0.row(i), vertices_t1.row(i), lower_bound, upper_bound);
        aabbs.emplace_back(
            AABB(
                lower_bound.array() - inflation_radius,
                upper_bound.array() + inflation_radius),
            num_items);
        num_items += countNewItems(aabbs[i].first);
    }
    m_vertexItems.resize(num_items);

    tbb::parallel_for(0l, long(vertices_t0.rows()), [&](long i) {
        insertElement(aabbs[i].first, i, m_vertexItems, aabbs[i].second);
    });
    assert(m_vertexItems.size() <= num_items);
#endif

    timer.stop();
    std::cout << "m_vertexItems.size()=" << m_vertexItems.size() << " "
              << timer.getElapsedTime() << " s" << std::endl;
}

/// @brief Compute a AABB for an edge moving through time (i.e. temporal quad).
void calculate_edge_extents(
    const Eigen::VectorX3d& edge_vertex0_t0,
    const Eigen::VectorX3d& edge_vertex1_t0,
    const Eigen::VectorX3d& edge_vertex0_t1,
    const Eigen::VectorX3d& edge_vertex1_t1,
    Eigen::VectorX3d& lower_bound,
    Eigen::VectorX3d& upper_bound)
{
    lower_bound = edge_vertex0_t0.cwiseMin(edge_vertex1_t0)
                      .cwiseMin(edge_vertex0_t1)
                      .cwiseMin(edge_vertex1_t1);
    upper_bound = edge_vertex0_t0.cwiseMax(edge_vertex1_t0)
                      .cwiseMax(edge_vertex0_t1)
                      .cwiseMax(edge_vertex1_t1);
}

void HashGrid::addEdge(
    const Eigen::VectorX3d& edge_vertex0_t0,
    const Eigen::VectorX3d& edge_vertex1_t0,
    const Eigen::VectorX3d& edge_vertex0_t1,
    const Eigen::VectorX3d& edge_vertex1_t1,
    const long index,
    const double inflation_radius)
{
    Eigen::VectorX3d lower_bound, upper_bound;
    calculate_edge_extents(
        edge_vertex0_t0, edge_vertex1_t0, edge_vertex0_t1, edge_vertex1_t1,
        lower_bound, upper_bound);
    this->addElement(
        AABB(
            lower_bound.array() - inflation_radius,
            upper_bound.array() + inflation_radius),
        index, m_edgeItems); // Edges have a positive id
}

void HashGrid::addEdges(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& edges,
    const double inflation_radius)
{
    igl::Timer timer;
    timer.start();
    assert(vertices_t0.rows() == vertices_t1.rows());

#ifndef IPC_TOOLKIT_SPATIAL_HASH_USE_TBB
    for (long i = 0; i < edges.rows(); i++) {
        addEdge(
            vertices_t0.row(edges(i, 0)), vertices_t0.row(edges(i, 1)),
            vertices_t1.row(edges(i, 0)), vertices_t1.row(edges(i, 1)), i,
            inflation_radius);
    }
#else
    // A vector of AABBs and where they start in the edge items
    std::vector<std::pair<AABB, int>> aabbs;
    aabbs.reserve(edges.rows());

    size_t num_items = m_edgeItems.size();
    for (long i = 0; i < edges.rows(); i++) {
        Eigen::VectorX3d lower_bound, upper_bound;
        calculate_edge_extents(
            vertices_t0.row(edges(i, 0)), vertices_t0.row(edges(i, 1)),
            vertices_t1.row(edges(i, 0)), vertices_t1.row(edges(i, 1)),
            lower_bound, upper_bound);
        aabbs.emplace_back(
            AABB(
                lower_bound.array() - inflation_radius,
                upper_bound.array() + inflation_radius),
            num_items);
        num_items += countNewItems(aabbs.back().first);
    }
    m_edgeItems.resize(num_items);

    tbb::parallel_for(0l, long(edges.rows()), [&](long i) {
        insertElement(aabbs[i].first, i, m_edgeItems, aabbs[i].second);
    });
    assert(m_edgeItems.size() <= num_items);
#endif

    timer.stop();
    std::cout << "m_edgeItems.size()=" << m_edgeItems.size() << " "
              << timer.getElapsedTime() << " s" << std::endl;
}

/// @brief Compute a AABB for an edge moving through time (i.e. temporal quad).
void calculate_face_extents(
    const Eigen::VectorX3d& face_vertex0_t0,
    const Eigen::VectorX3d& face_vertex1_t0,
    const Eigen::VectorX3d& face_vertex2_t0,
    const Eigen::VectorX3d& face_vertex0_t1,
    const Eigen::VectorX3d& face_vertex1_t1,
    const Eigen::VectorX3d& face_vertex2_t1,
    Eigen::VectorX3d& lower_bound,
    Eigen::VectorX3d& upper_bound)
{
    lower_bound = face_vertex0_t0.cwiseMin(face_vertex1_t0)
                      .cwiseMin(face_vertex2_t0)
                      .cwiseMin(face_vertex0_t1)
                      .cwiseMin(face_vertex1_t1)
                      .cwiseMin(face_vertex2_t1);
    upper_bound = face_vertex0_t0.cwiseMax(face_vertex1_t0)
                      .cwiseMax(face_vertex2_t0)
                      .cwiseMax(face_vertex0_t1)
                      .cwiseMax(face_vertex1_t1)
                      .cwiseMax(face_vertex2_t1);
}

void HashGrid::addFace(
    const Eigen::VectorX3d& face_vertex0_t0,
    const Eigen::VectorX3d& face_vertex1_t0,
    const Eigen::VectorX3d& face_vertex2_t0,
    const Eigen::VectorX3d& face_vertex0_t1,
    const Eigen::VectorX3d& face_vertex1_t1,
    const Eigen::VectorX3d& face_vertex2_t1,
    const long index,
    const double inflation_radius)
{
    Eigen::VectorX3d lower_bound, upper_bound;
    calculate_face_extents(
        face_vertex0_t0, face_vertex1_t0, face_vertex2_t0, //
        face_vertex0_t1, face_vertex1_t1, face_vertex2_t1, //
        lower_bound, upper_bound);
    this->addElement(
        AABB(
            lower_bound.array() - inflation_radius,
            upper_bound.array() + inflation_radius),
        index, m_faceItems);
}

void HashGrid::addFaces(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const Eigen::MatrixXi& faces,
    const double inflation_radius)
{
    igl::Timer timer;
    timer.start();
    assert(vertices_t0.rows() == vertices_t1.rows());

#ifndef IPC_TOOLKIT_SPATIAL_HASH_USE_TBB
    for (long i = 0; i < faces.rows(); i++) {
        addFace(
            vertices_t0.row(faces(i, 0)), vertices_t0.row(faces(i, 1)),
            vertices_t0.row(faces(i, 2)), vertices_t1.row(faces(i, 0)),
            vertices_t1.row(faces(i, 1)), vertices_t1.row(faces(i, 2)), i,
            inflation_radius);
    }
#else
    // A vector of AABBs and where they start in the face items
    std::vector<std::pair<AABB, int>> aabbs;
    aabbs.reserve(faces.rows());

    size_t num_items = m_faceItems.size();
    for (long i = 0; i < faces.rows(); i++) {
        Eigen::VectorX3d lower_bound, upper_bound;
        calculate_face_extents(
            vertices_t0.row(faces(i, 0)), vertices_t0.row(faces(i, 1)),
            vertices_t0.row(faces(i, 2)), vertices_t1.row(faces(i, 0)),
            vertices_t1.row(faces(i, 1)), vertices_t1.row(faces(i, 2)),
            lower_bound, upper_bound);
        aabbs.emplace_back(
            AABB(
                lower_bound.array() - inflation_radius,
                upper_bound.array() + inflation_radius),
            num_items);
        num_items += countNewItems(aabbs.back().first);
    }
    m_faceItems.resize(num_items);

    tbb::parallel_for(0l, long(faces.rows()), [&](long i) {
        insertElement(aabbs[i].first, i, m_faceItems, aabbs[i].second);
    });
    assert(m_faceItems.size() <= num_items);
#endif

    timer.stop();
    std::cout << "m_faceItems.size()=" << m_faceItems.size() << " "
              << timer.getElapsedTime() << " s" << std::endl;
}

void HashGrid::AABB_to_cells(
    const AABB& aabb, Eigen::VectorX3i& cell_min, Eigen::VectorX3i& cell_max)
{
    cell_min =
        ((aabb.getMin() - m_domainMin) / m_cellSize).template cast<int>();
    // We can round down to -1, but not less
    assert((cell_min.array() >= -1).all());
    assert((cell_min.array() <= m_gridSize.array()).all());
    cell_min = cell_min.array().max(0).min(m_gridSize.array() - 1);

    cell_max =
        ((aabb.getMax() - m_domainMin) / m_cellSize).template cast<int>();
    assert((cell_max.array() >= -1).all());
    assert((cell_max.array() <= m_gridSize.array()).all());
    cell_max = cell_max.array().max(0).min(m_gridSize.array() - 1);
    assert((cell_min.array() <= cell_max.array()).all());
}

int HashGrid::countNewItems(const AABB& aabb)
{
    Eigen::VectorX3<int> cell_min, cell_max;
    AABB_to_cells(aabb, cell_min, cell_max);
    return (cell_max.array() - cell_min.array() + 1).prod();
}

void HashGrid::addElement(const AABB& aabb, const int id, HashItems& items)
{
    insertElement(aabb, id, items, items.size()); // Insert at end
}

void HashGrid::insertElement(
    const AABB& aabb, const int id, HashItems& items, size_t start_i)
{
    Eigen::VectorX3<int> cell_min, cell_max;
    AABB_to_cells(aabb, cell_min, cell_max);

    int min_z = cell_min.size() == 3 ? cell_min.z() : 0;
    int max_z = cell_max.size() == 3 ? cell_max.z() : 0;
    size_t item_idx = start_i;
    for (int x = cell_min.x(); x <= cell_max.x(); ++x) {
        for (int y = cell_min.y(); y <= cell_max.y(); ++y) {
            for (int z = min_z; z <= max_z; ++z) {
                if (item_idx >= items.size()) {
                    items.emplace_back(hash(x, y, z), id, aabb);
                } else {
                    items[item_idx].key = hash(x, y, z);
                    items[item_idx].id = id;
                    items[item_idx].aabb = aabb;
                }
                item_idx++;
            }
        }
    }
}

template <typename Candidates>
void getPairs(
    const std::function<bool(int, int)>& is_endpoint,
    const std::function<bool(int, int)>& is_same_group,
    HashItems& items0,
    HashItems& items1,
    Candidates& candidates)
{
    // Sort all they (key,value) pairs, where key is the hash key, and value
    tbb::parallel_sort(items0.begin(), items0.end());
    tbb::parallel_sort(items1.begin(), items1.end());

    // Entries with the same key means they share a cell (that cell index
    // hashes to the same key) and should be flagged for low-level intersection
    // testing. So we loop over the entire sorted set of (key,value) pairs
    // creating Candidate entries for vertex-edge pairs with the same key
    int i = 0, j_start = 0;
    while (i < items0.size() && j_start < items1.size()) {
        const HashItem& item0 = items0[i];

        int j = j_start;
        while (j < items1.size()) {
            const HashItem& item1 = items1[j];

            if (item0.key == item1.key) {
                if (!is_endpoint(item0.id, item1.id)
                    && !is_same_group(item0.id, item1.id)
                    && AABB::are_overlaping(item0.aabb, item1.aabb)) {
                    candidates.emplace_back(item0.id, item1.id);
                }
            } else {
                break;
            }
            j++;
        }

        if (i == items0.size() - 1 || item0.key != items0[i + 1].key) {
            j_start = j;
        }
        i++;
    }

    // Remove the duplicate candidates
    tbb::parallel_sort(candidates.begin(), candidates.end());
    auto new_end = std::unique(candidates.begin(), candidates.end());
    candidates.erase(new_end, candidates.end());
}

template <typename Candidates>
void getPairs(
    const std::function<bool(int, int)>& is_endpoint,
    const std::function<bool(int, int)>& is_same_group,
#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
    const std::function<long(const typename Candidates::value_type&)>& hash,
#endif
    HashItems& items,
    Candidates& candidates)
{
    // Sort all they (key,value) pairs, where key is the hash key, and value
    tbb::parallel_sort(items.begin(), items.end());

    igl::Timer timer;
    timer.start();

#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_TBB
    std::vector<int> cell_starts;
    int prev_key = -1;
    for (int i = 0; i < items.size(); i++) {
        if (items[i].key != prev_key) {
            cell_starts.push_back(i);
            prev_key = items[i].key;
        }
    }
    cell_starts.push_back(items.size()); // extra cell start for the end

#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
    tbb::concurrent_unordered_set<
        typename Candidates::value_type, decltype(hash)>
    candidate_set(long(0.1 * items.size()), hash);
#else
    std::vector<Candidates> cell_candidates(cell_starts.size() - 1);
#endif
    tbb::parallel_for(size_t(0), cell_starts.size() - 1, [&](size_t ci) {
        size_t current_cell = cell_starts[ci], next_cell = cell_starts[ci + 1];
        for (int i = current_cell; i < next_cell; i++) {
            const HashItem& item0 = items[i];
            for (int j = current_cell + 1; j < next_cell; j++) {
                const HashItem& item1 = items[j];
                assert(item0.key == item1.key);
                if (!is_endpoint(item0.id, item1.id)
                    && !is_same_group(item0.id, item1.id)
                    && AABB::are_overlaping(item0.aabb, item1.aabb)) {
#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
                    candidate_set.emplace(item0.id, item1.id);
#else
                    cell_candidates[ci].emplace_back(item0.id, item1.id);
#endif
                }
            }
        }
    });

#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
    candidates.reserve(candidate_set.size());
    candidates.insert(
        candidates.end(), candidate_set.begin(), candidate_set.end());
#else
    size_t total_candidates = 0;
    for (const auto& cell_candidate : cell_candidates) {
        total_candidates += cell_candidate.size();
    }
    candidates.reserve(total_candidates);
    for (const auto& cell_candidate : cell_candidates) {
        candidates.insert(
            candidates.end(), cell_candidate.begin(), cell_candidate.end());
    }
#endif
#else
    // Entries with the same key means they share a cell (that cell index
    // hashes to the same key) and should be flagged for low-level intersection
    // testing. So we loop over the entire sorted set of (key,value) pairs
    // creating Candidate entries for pairs with the same key.
    for (size_t i = 0; i < items.size(); i++) {
        const HashItem& item0 = items[i];
        for (int j = i + 1; j < items.size(); j++) {
            const HashItem& item1 = items[j];
            if (item0.key == item1.key) {
                if (!is_endpoint(item0.id, item1.id)
                    && !is_same_group(item0.id, item1.id)
                    && AABB::are_overlaping(item0.aabb, item1.aabb)) {
                    candidates.emplace_back(item0.id, item1.id);
                }
            } else {
                break; // This avoids a brute force comparison
            }
        }
    }
#endif

    timer.stop();
    std::cout << "find_intersections " << timer.getElapsedTime() << " s"
              << "\n";

#ifndef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
    timer.start();
    // Remove the duplicate candidates
    tbb::parallel_sort(candidates.begin(), candidates.end());
    auto new_end = std::unique(candidates.begin(), candidates.end());
    candidates.erase(new_end, candidates.end());
    timer.stop();
    std::cout << "remove_duplicates " << timer.getElapsedTime() << " s"
              << std::endl;
#endif
}

void HashGrid::getVertexEdgePairs(
    const Eigen::MatrixXi& edges,
    const Eigen::VectorXi& group_ids,
    std::vector<EdgeVertexCandidate>& ev_candidates)
{
    igl::Timer timer;
    timer.start();

    auto is_endpoint = [&](int ei, int vi) {
        return edges(ei, 0) == vi || edges(ei, 1) == vi;
    };

    bool check_groups = group_ids.size() > 0;
    auto is_same_group = [&](int ei, int vi) {
        return check_groups
            && (group_ids(vi) == group_ids(edges(ei, 0))
                || group_ids(vi) == group_ids(edges(ei, 1)));
    };

    getPairs(
        is_endpoint, is_same_group, m_edgeItems, m_vertexItems, ev_candidates);

    timer.stop();
    std::cout << "HashGrid::getVertexEdgePairs " << timer.getElapsedTime()
              << " s" << std::endl;
}

void HashGrid::getEdgeEdgePairs(
    const Eigen::MatrixXi& edges,
    const Eigen::VectorXi& group_ids,
    std::vector<EdgeEdgeCandidate>& ee_candidates)
{
    igl::Timer timer;

    auto is_endpoint = [&](int ei, int ej) {
        return edges(ei, 0) == edges(ej, 0) || edges(ei, 0) == edges(ej, 1)
            || edges(ei, 1) == edges(ej, 0) || edges(ei, 1) == edges(ej, 1);
    };

    bool check_groups = group_ids.size() > 0;
    auto is_same_group = [&](int ei, int ej) {
        return check_groups
            && (group_ids(edges(ei, 0)) == group_ids(edges(ej, 0))
                || group_ids(edges(ei, 0)) == group_ids(edges(ej, 1))
                || group_ids(edges(ei, 1)) == group_ids(edges(ej, 0))
                || group_ids(edges(ei, 1)) == group_ids(edges(ej, 1)));
    };

    timer.start();

#ifdef IPC_TOOLKIT_SPATIAL_HASH_USE_UNORDERED_SET
    auto hash = [&](const EdgeEdgeCandidate& candidate) {
        return std::min(candidate.edge0_index, candidate.edge1_index)
            * edges.rows()
            + std::max(candidate.edge0_index, candidate.edge1_index);
    };

    getPairs(is_endpoint, is_same_group, hash, m_edgeItems, ee_candidates);
#else
    getPairs(is_endpoint, is_same_group, m_edgeItems, ee_candidates);
#endif

    timer.stop();
    std::cout << "HashGrid::getEdgeEdgePairs " << timer.getElapsedTime() << " s"
              << "\n";
}

void HashGrid::getEdgeFacePairs(
    const Eigen::MatrixXi& edges,
    const Eigen::MatrixXi& faces,
    const Eigen::VectorXi& group_ids,
    std::vector<EdgeFaceCandidate>& ef_candidates)
{
    auto is_endpoint = [&](int ei, int fi) {
        // Check if the edge and face have a common end-point
        return edges(ei, 0) == faces(fi, 0) || edges(ei, 0) == faces(fi, 1)
            || edges(ei, 0) == faces(fi, 2) || edges(ei, 1) == faces(fi, 0)
            || edges(ei, 1) == faces(fi, 1) || edges(ei, 1) == faces(fi, 2);
    };

    bool check_groups = group_ids.size() > 0;
    auto is_same_group = [&](int ei, int fi) {
        return check_groups
            && (group_ids(edges(ei, 0)) == group_ids(faces(fi, 0))
                || group_ids(edges(ei, 0)) == group_ids(faces(fi, 1))
                || group_ids(edges(ei, 0)) == group_ids(faces(fi, 2))
                || group_ids(edges(ei, 1)) == group_ids(faces(fi, 0))
                || group_ids(edges(ei, 1)) == group_ids(faces(fi, 1))
                || group_ids(edges(ei, 1)) == group_ids(faces(fi, 2)));
    };

    getPairs(
        is_endpoint, is_same_group, m_edgeItems, m_faceItems, ef_candidates);
}

void HashGrid::getFaceVertexPairs(
    const Eigen::MatrixXi& faces,
    const Eigen::VectorXi& group_ids,
    std::vector<FaceVertexCandidate>& fv_candidates)
{
    igl::Timer timer;
    timer.start();

    auto is_endpoint = [&](int fi, int vi) {
        return vi == faces(fi, 0) || vi == faces(fi, 1) || vi == faces(fi, 2);
    };

    bool check_groups = group_ids.size() > 0;
    auto is_same_group = [&](int fi, int vi) {
        return check_groups
            && (group_ids(vi) == group_ids(faces(fi, 0))
                || group_ids(vi) == group_ids(faces(fi, 1))
                || group_ids(vi) == group_ids(faces(fi, 2)));
    };

    getPairs(
        is_endpoint, is_same_group, m_faceItems, m_vertexItems, fv_candidates);

    timer.stop();
    std::cout << "HashGrid::getFaceVertexPairs " << timer.getElapsedTime()
              << " s" << std::endl;
}

} // namespace ipc
