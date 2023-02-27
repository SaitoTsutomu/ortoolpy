import pytest

from ortoolpy.optimization import (
    BinPacking,
    ChinesePostman,
    CombinatorialAuction,
    DijkstraPath,
    FacilityLocation,
    FacilityLocationWithoutCapacity,
    Gap,
    Knapsack,
    MaximumFlow,
    MaximumStableSet,
    MaxMatching,
    MaxWeightMatching,
    MinCostFlow,
    MinimumSpanningTree,
    QuadAssign,
    SetCovering,
    SetPartition,
    ShiftScheduling,
    StableMatching,
    Tsp,
    TwoDimPacking,
    TwoMachineFlowshop,
    Vrp,
)


def test_MinimumSpanningTree(snapshot):
    res = MinimumSpanningTree("data/edge0.csv")
    snapshot.assert_match(res)


def test_MaximumStableSet(snapshot):
    res = MaximumStableSet("data/node0.csv", "data/edge0.csv")
    snapshot.assert_match(res)


def test_DijkstraPath(snapshot):
    res = DijkstraPath("data/edge0.csv", 5, 2)
    snapshot.assert_match(res)


def test_MaximumFlow(snapshot):
    res = MaximumFlow("data/edge0.csv", 5, 2)
    snapshot.assert_match(res)


def test_MinCostFlow(snapshot):
    res = MinCostFlow("data/node0.csv", "data/edge0.csv")
    snapshot.assert_match(res)


def test_Vrp(snapshot):
    res = Vrp("data/node1.csv", "data/edge1.csv", 2, 3)
    snapshot.assert_match(res)


@pytest.mark.skip
def test_Tsp(snapshot):
    res = Tsp("data/node1.csv")
    snapshot.assert_match(res)


def test_ChinesePostman(snapshot):
    res = ChinesePostman("data/edge0.csv")
    snapshot.assert_match(res)


def test_SetCovering(snapshot):
    res = SetCovering("data/subset.csv")
    snapshot.assert_match(res)


def test_SetPartition(snapshot):
    res = SetPartition("data/subset.csv")
    snapshot.assert_match(res)


def test_CombinatorialAuction(snapshot):
    res = CombinatorialAuction("data/auction.csv")
    snapshot.assert_match(res)


def test_TwoMachineFlowshop(snapshot):
    res = TwoMachineFlowshop("data/flowshop.csv")
    snapshot.assert_match(res)


def test_ShiftScheduling(snapshot):
    res = ShiftScheduling(8, 4, "休日夜", ["夜夜", "夜日", "日日日"], {"日": [2] * 8, "夜": [1] * 8})
    snapshot.assert_match(res)


def test_Knapsack(snapshot):
    res = Knapsack("data/knapsack.csv", 100)
    snapshot.assert_match(res)


def test_BinPacking(snapshot):
    res = BinPacking("data/binpacking.csv", 10)
    snapshot.assert_match(res)


def test_TwoDimPacking(snapshot):
    res = TwoDimPacking("data/tdpacking.csv", 500, 300)
    snapshot.assert_match(res)


def test_FacilityLocation(snapshot):
    res = FacilityLocation("data/facility.csv", 2)
    snapshot.assert_match(res)


def test_FacilityLocationWithoutCapacity(snapshot):
    res = FacilityLocationWithoutCapacity("data/facility.csv", 2)
    snapshot.assert_match(res)


def test_QuadAssign(snapshot):
    res = QuadAssign("data/quad_assign_quant.csv", "data/quad_assign_dist.csv")
    snapshot.assert_match(res)


def test_Gap(snapshot):
    res = Gap("data/gap.csv", [2, 1])
    snapshot.assert_match(res)


def test_MaxMatching(snapshot):
    res = MaxMatching("data/edge0.csv")
    snapshot.assert_match(res)


def test_MaxWeightMatching(snapshot):
    res = MaxWeightMatching("data/edge0.csv")
    snapshot.assert_match(res)


def test_StableMatching(snapshot):
    res = StableMatching("data/stable.csv")
    snapshot.assert_match(res)
