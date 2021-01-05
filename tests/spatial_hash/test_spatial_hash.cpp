#include <catch2/catch.hpp>

#include <Eigen/Core>

#include <igl/IO>
#include <igl/edges.h>
#include <igl/Timer.h>

#include <ipc/ipc.hpp>
#include <ipc/spatial_hash/hash_grid.hpp>

using namespace ipc;

TEST_CASE("AABB initilization", "[spatial_hash][AABB]")
{
    int dim = GENERATE(2, 3);
    CAPTURE(dim);
    AABB aabb;
    Eigen::VectorXd actual_center(dim);
    SECTION("Empty AABB")
    {
        aabb = AABB(Eigen::VectorXd::Zero(dim), Eigen::VectorXd::Zero(dim));
        actual_center = Eigen::VectorXd::Zero(dim);
    }
    SECTION("Box centered at zero")
    {
        Eigen::VectorXd min =
            Eigen::VectorXd::Random(dim).array() - 1; // in range [-2, 0]
        Eigen::VectorXd max = -min;
        aabb = AABB(min, max);
        actual_center = Eigen::VectorXd::Zero(dim);
    }
    SECTION("Box not centered at zero")
    {
        Eigen::VectorXd min(dim), max(dim);
        if (dim == 2) {
            min << 5.1, 3.14;
            max << 10.4, 7.89;
            actual_center << 7.75, 5.515;
        } else {
            min << 5.1, 3.14, 7.94;
            max << 10.4, 7.89, 10.89;
            actual_center << 7.75, 5.515, 9.415;
        }
        aabb = AABB(min, max);
    }
    Eigen::VectorXd center_diff = aabb.getCenter() - actual_center;
    CHECK(center_diff.norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("AABB overlapping", "[spatial_hash][AABB]")
{
    AABB a, b;
    bool are_overlaping = false;
    SECTION("a to the right of b")
    {
        a = AABB(Eigen::Vector2d(-1, 0), Eigen::Vector2d(0, 1));
        SECTION("overlapping")
        {
            b = AABB(Eigen::Vector2d(-0.5, 0), Eigen::Vector2d(0.5, 1));
            are_overlaping = true;
        }
        SECTION("not overlapping")
        {
            b = AABB(Eigen::Vector2d(0.5, 0), Eigen::Vector2d(1.5, 1));
            are_overlaping = false;
        }
    }
    SECTION("b to the right of a")
    {
        b = AABB(Eigen::Vector2d(-1, 0), Eigen::Vector2d(0, 1));
        SECTION("overlapping")
        {
            a = AABB(Eigen::Vector2d(-0.5, 0), Eigen::Vector2d(0.5, 1));
            are_overlaping = true;
        }
        SECTION("not overlapping")
        {
            a = AABB(Eigen::Vector2d(0.5, 0), Eigen::Vector2d(1.5, 1));
            are_overlaping = false;
        }
    }
    SECTION("a above b")
    {
        a = AABB(Eigen::Vector2d(0, -1), Eigen::Vector2d(1, 0));
        SECTION("overlapping")
        {
            b = AABB(Eigen::Vector2d(0, -0.5), Eigen::Vector2d(1, 0.5));
            are_overlaping = true;
        }
        SECTION("not overlapping")
        {
            b = AABB(Eigen::Vector2d(0, 0.5), Eigen::Vector2d(1, 1.5));
            are_overlaping = false;
        }
    }
    SECTION("a above b")
    {
        b = AABB(Eigen::Vector2d(0, -1), Eigen::Vector2d(1, 0));
        SECTION("overlapping")
        {
            a = AABB(Eigen::Vector2d(0, -0.5), Eigen::Vector2d(1, 0.5));
            are_overlaping = true;
        }
        SECTION("not overlapping")
        {
            a = AABB(Eigen::Vector2d(0, 0.5), Eigen::Vector2d(1, 1.5));
            are_overlaping = false;
        }
    }
    CHECK(AABB::are_overlaping(a, b) == are_overlaping);
}

TEST_CASE("Vertex-Vertex Spatial Hash", "[ccd][spatial_hash]")
{
    Eigen::MatrixXd V_t0(4, 2);
    V_t0.row(0) << 1.11111, 0.5;  // edge 0 vertex 0
    V_t0.row(1) << 1.11111, 0.75; // edge 0 vertex 1
    V_t0.row(2) << 1, 0.5;        // edge 1 vertex 0
    V_t0.row(3) << 1, 0.75;       // edge 1 vertex 1

    Eigen::MatrixXd V_t1 = V_t0;
    V_t1.row(0) << 0.888889, 0.5;  // edge 0 vertex 0
    V_t1.row(1) << 0.888889, 0.75; // edge 0 vertex 1

    Eigen::MatrixXi E(2, 2);
    E.row(0) << 1, 0;
    E.row(1) << 2, 3;

    bool ignore_internal_vertices = GENERATE(false, true);

    bool is_valid_step = ipc::is_step_collision_free(
        V_t0, V_t1, E, /*F=*/Eigen::MatrixXi(), ignore_internal_vertices);

    CAPTURE(ignore_internal_vertices);
    CHECK(!is_valid_step);
}

TEST_CASE("Entire 2D Mesh", "[ccd][spatial_hash]")
{
    Eigen::MatrixXd V_t0;
    igl::readCSV(std::string(TEST_DATA_DIR) + "V_t0.txt", V_t0);

    Eigen::MatrixXd V_t1;
    igl::readCSV(std::string(TEST_DATA_DIR) + "V_t1.txt", V_t1);

    Eigen::MatrixXd E_double;
    igl::readCSV(std::string(TEST_DATA_DIR) + "E.txt", E_double);
    Eigen::MatrixXi E = E_double.cast<int>();

    bool ignore_internal_vertices = GENERATE(false, true);

    bool is_valid_step;
    SECTION("2D")
    {
        is_valid_step = ipc::is_step_collision_free(
            V_t0.leftCols(2), V_t1.rightCols(2), E, /*F=*/Eigen::MatrixXi(),
            ignore_internal_vertices);
    }
    SECTION("3D")
    {
        is_valid_step = ipc::is_step_collision_free(
            V_t0, V_t1, E, /*F=*/Eigen::MatrixXi(), ignore_internal_vertices);
    }

    CAPTURE(ignore_internal_vertices);
    CHECK(!is_valid_step);
}

TEST_CASE(
    "Test construct_constraint_set() with codimensional points",
    "[construct_constraint_set][spatial_hash]")
{
    double dhat = 0.1;
    // double dhat = 0.00173123;
    Eigen::MatrixXd V_rest, V;
    igl::readDMAT(
        std::string(TEST_DATA_DIR) + "codim-points/V_rest.dmat", V_rest);
    igl::readDMAT(std::string(TEST_DATA_DIR) + "codim-points/V.dmat", V);
    Eigen::MatrixXi E, F;
    igl::readDMAT(std::string(TEST_DATA_DIR) + "codim-points/E.dmat", E);
    igl::readDMAT(std::string(TEST_DATA_DIR) + "codim-points/F.dmat", F);

    Constraints constraint_set;
    construct_constraint_set(
        V_rest, V, E, F, dhat, constraint_set,
        /*ignore_internal_vertices=*/false);

    std::cout << constraint_set.size() << std::endl;
    std::cout << sqrt(compute_minimum_distance(V, E, F, constraint_set))
              << std::endl;
}

TEST_CASE("Benchmark different spatial hashes", "[!benchmark][spatial_hash]")
{
    using namespace ipc;

    Eigen::MatrixXd V, U;
    Eigen::MatrixXi E, F;
    Eigen::VectorXi group_ids;

    SECTION("Simple")
    {
        V.resize(4, 3);
        V.row(0) << -1, -1, 0;
        V.row(1) << 1, -1, 0;
        V.row(2) << 0, 1, 1;
        V.row(3) << 0, 1, -1;

        E.resize(2, 2);
        E.row(0) << 0, 1;
        E.row(1) << 2, 3;

        SECTION("Without group ids") {}
        SECTION("With group ids")
        {
            group_ids.resize(4);
            group_ids << 0, 0, 1, 1;
        }

        F.resize(0, 3);

        U = Eigen::MatrixXd::Zero(V.rows(), V.cols());
        U.col(1).head(2).setConstant(2);
        U.col(1).tail(2).setConstant(-2);
    }
    SECTION("Complex")
    {
        std::string filename =
            GENERATE(std::string("cube.obj"), std::string("bunny.obj"));
        std::string mesh_path = std::string(TEST_DATA_DIR) + filename;
        bool success = igl::read_triangle_mesh(mesh_path, V, F);
        REQUIRE(success);
        igl::edges(F, E);

        U = Eigen::MatrixXd::Zero(V.rows(), V.cols());
        U.col(1).setOnes();
    }

    HashGrid hashgrid;
    Candidates candidates;

    double inflation_radius = 1e-2; // GENERATE(take(5, random(0.0, 0.1)));

    for (int i = 0; i < 2; i++) {
        // BENCHMARK("IPC") {}
        BENCHMARK("Hash Grid")
        {
            hashgrid.resize(V, V + U, E, inflation_radius);
            hashgrid.addVertices(V, V + U, inflation_radius);
            hashgrid.addEdges(V, V + U, E, inflation_radius);
            hashgrid.addFaces(V, V + U, F, inflation_radius);

            candidates.clear();

            hashgrid.getVertexEdgePairs(E, group_ids, candidates.ev_candidates);
            hashgrid.getEdgeEdgePairs(E, group_ids, candidates.ee_candidates);
            hashgrid.getFaceVertexPairs(F, group_ids, candidates.fv_candidates);
        };

        // Impacts brute_force_impacts;
        // detect_collisions(
        //     vertices, vertices + displacements, edges, faces, group_ids,
        //     CollisionType::EDGE_EDGE | CollisionType::FACE_VERTEX,
        //     brute_force_impacts, DetectionMethod::BRUTE_FORCE);
        // REQUIRE(brute_force_impacts.ev_impacts.size() == 0);
        //
        // Impacts hash_impacts;
        // detect_collisions(
        //     vertices, vertices + displacements, edges, faces, group_ids,
        //     CollisionType::EDGE_EDGE | CollisionType::FACE_VERTEX,
        //     hash_impacts, DetectionMethod::HASH_GRID);
        // REQUIRE(hash_impacts.ev_impacts.size() == 0);
        //
        // REQUIRE(
        //     brute_force_impacts.ee_impacts.size()
        //     == hash_impacts.ee_impacts.size());
        // std::sort(
        //     brute_force_impacts.ee_impacts.begin(),
        //     brute_force_impacts.ee_impacts.end(),
        //     compare_impacts_by_time<EdgeEdgeImpact>);
        // std::sort(
        //     hash_impacts.ee_impacts.begin(), hash_impacts.ee_impacts.end(),
        //     compare_impacts_by_time<EdgeEdgeImpact>);
        // bool is_equal =
        //     brute_force_impacts.ee_impacts == hash_impacts.ee_impacts;
        // CHECK(is_equal);
        // REQUIRE(
        //     brute_force_impacts.fv_impacts.size()
        //     == hash_impacts.fv_impacts.size());
        // std::sort(
        //     brute_force_impacts.fv_impacts.begin(),
        //     brute_force_impacts.fv_impacts.end(),
        //     compare_impacts_by_time<FaceVertexImpact>);
        // std::sort(
        //     hash_impacts.fv_impacts.begin(), hash_impacts.fv_impacts.end(),
        //     compare_impacts_by_time<FaceVertexImpact>);
        // CHECK(brute_force_impacts.fv_impacts == hash_impacts.fv_impacts);

        U.setRandom();
        U *= 3;
    }
}

TEST_CASE("Benchmark slow broadphase CCD", "[!benchmark][spatial_hash][febio]")
{
    using namespace ipc;

    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi E, F;
    Eigen::VectorXi group_ids;

    bool success = igl::read_triangle_mesh(
        std::string(TEST_DATA_DIR) + "slow-broadphase-ccd/0.obj", V0, F);
    REQUIRE(success);
    success = igl::read_triangle_mesh(
        std::string(TEST_DATA_DIR) + "slow-broadphase-ccd/1.obj", V1, F);
    REQUIRE(success);
    igl::edges(F, E);

    HashGrid hashgrid;
    Candidates candidates;

    double inflation_radius = 0; // GENERATE(take(5, random(0.0, 0.1)));

    igl::Timer timer;
    timer.start();

    // BENCHMARK("Hash Grid")
    // {
    hashgrid.resize(V0, V1, E, inflation_radius);
    hashgrid.addVertices(V0, V1, inflation_radius);
    hashgrid.addEdges(V0, V1, E, inflation_radius);
    hashgrid.addFaces(V0, V1, F, inflation_radius);

    candidates.clear();

    hashgrid.getVertexEdgePairs(E, group_ids, candidates.ev_candidates);
    hashgrid.getEdgeEdgePairs(E, group_ids, candidates.ee_candidates);
    hashgrid.getFaceVertexPairs(F, group_ids, candidates.fv_candidates);
    // };

    timer.stop();
    std::cout << timer.getElapsedTime() << " s" << std::endl;
    std::cout << candidates.ev_candidates.size() << " "
              << candidates.ee_candidates.size() << " "
              << candidates.fv_candidates.size() << std::endl;
}

TEST_CASE("Benchmark cell size", "[!benchmark][spatial_hash][cellsize]")
{
    using namespace ipc;

    Eigen::MatrixXd V0, V1;
    Eigen::MatrixXi E, F;
    Eigen::VectorXi group_ids;

    bool success = igl::read_triangle_mesh(
        std::string(TEST_DATA_DIR) + "slow-broadphase-ccd/0.obj", V0, F);
    REQUIRE(success);
    success = igl::read_triangle_mesh(
        std::string(TEST_DATA_DIR) + "slow-broadphase-ccd/1.obj", V1, F);
    REQUIRE(success);
    igl::edges(F, E);

    HashGrid hashgrid;
    Candidates candidates;

    double multiplier =
        GENERATE(map([](double x) { return std::pow(2, x); }, range(-2, 7)));

    double inflation_radius = 0; // GENERATE(take(5, random(0.0, 0.1)));

    std::cout << multiplier << ":" << std::endl;

    igl::Timer timer;
    timer.start();

    hashgrid.resize(V0, V1, E, inflation_radius);
    hashgrid.resize(
        hashgrid.domainMin(), hashgrid.domainMax(),
        multiplier * hashgrid.cellSize());
    hashgrid.addVertices(V0, V1, inflation_radius);
    hashgrid.addEdges(V0, V1, E, inflation_radius);
    hashgrid.addFaces(V0, V1, F, inflation_radius);

    candidates.clear();

    hashgrid.getVertexEdgePairs(E, group_ids, candidates.ev_candidates);
    hashgrid.getEdgeEdgePairs(E, group_ids, candidates.ee_candidates);
    hashgrid.getFaceVertexPairs(F, group_ids, candidates.fv_candidates);

    timer.stop();
    std::cout << timer.getElapsedTime() << " s" << std::endl;
}
