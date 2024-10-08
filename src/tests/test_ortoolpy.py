# flake8: noqa: S101
from pathlib import Path

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

DATA_PATH = Path(__file__).parent.parent / "data"


def test_MinimumSpanningTree(snapshot):
    res = MinimumSpanningTree(str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "MinimumSpanningTree.txt")


def test_MaximumStableSet(snapshot):
    res = MaximumStableSet(str(DATA_PATH / "node0.csv"), str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "MaximumStableSet.txt")


def test_DijkstraPath(snapshot):
    res = DijkstraPath(str(DATA_PATH / "edge0.csv"), 5, 2)
    snapshot.assert_match(str(res), "DijkstraPath.txt")


def test_MaximumFlow(snapshot):
    res = MaximumFlow(str(DATA_PATH / "edge0.csv"), 5, 2)
    snapshot.assert_match(str(res), "MaximumFlow.txt")


def test_MinCostFlow(snapshot):
    res = MinCostFlow(str(DATA_PATH / "node0.csv"), str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "MinCostFlow.txt")


def test_Vrp(snapshot):
    res = Vrp(str(DATA_PATH / "node1.csv"), str(DATA_PATH / "edge1.csv"), 2, 3)
    snapshot.assert_match(str(res), "Vrp.txt")


@pytest.mark.skip
def test_Tsp(snapshot):
    res = Tsp(str(DATA_PATH / "node1.csv"))
    snapshot.assert_match(str(res), "Tsp.txt")


def test_ChinesePostman(snapshot):
    res = ChinesePostman(str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "ChinesePostman.txt")


def test_SetCovering(snapshot):
    res = SetCovering(str(DATA_PATH / "subset.csv"))
    snapshot.assert_match(str(res), "SetCovering.txt")


def test_SetPartition(snapshot):
    res = SetPartition(str(DATA_PATH / "subset.csv"))
    snapshot.assert_match(str(res), "SetPartition.txt")


def test_CombinatorialAuction(snapshot):
    res = CombinatorialAuction(str(DATA_PATH / "auction.csv"))
    snapshot.assert_match(str(res), "CombinatorialAuction.txt")


def test_TwoMachineFlowshop(snapshot):
    res = TwoMachineFlowshop(str(DATA_PATH / "flowshop.csv"))
    snapshot.assert_match(str(res), "TwoMachineFlowshop.txt")


def test_ShiftScheduling(snapshot):
    res = ShiftScheduling(8, 4, "休日夜", ["夜夜", "夜日", "日日日"], {"日": [2] * 8, "夜": [1] * 8})
    snapshot.assert_match(str(res), "ShiftScheduling.txt")


def test_Knapsack(snapshot):
    res = Knapsack(str(DATA_PATH / "knapsack.csv"), 100)
    snapshot.assert_match(str(res), "Knapsack.txt")


def test_BinPacking(snapshot):
    res = BinPacking(str(DATA_PATH / "binpacking.csv"), 10)
    snapshot.assert_match(str(res), "BinPacking.txt")


def test_TwoDimPacking(snapshot):
    res = TwoDimPacking(str(DATA_PATH / "tdpacking.csv"), 500, 300)
    snapshot.assert_match(str(res), "TwoDimPacking.txt")


def test_FacilityLocation(snapshot):
    res = FacilityLocation(str(DATA_PATH / "facility.csv"), 2)
    snapshot.assert_match(str(res), "FacilityLocation.txt")


def test_FacilityLocationWithoutCapacity(snapshot):
    res = FacilityLocationWithoutCapacity(str(DATA_PATH / "facility.csv"), 2)
    snapshot.assert_match(str(res), "FacilityLocationWithoutCapacity.txt")


def test_QuadAssign(snapshot):
    res = QuadAssign(str(DATA_PATH / "quad_assign_quant.csv"), str(DATA_PATH / "quad_assign_dist.csv"))
    snapshot.assert_match(str(res), "QuadAssign.txt")


def test_Gap(snapshot):
    res = Gap(str(DATA_PATH / "gap.csv"), [2, 1])
    snapshot.assert_match(str(res), "Gap.txt")


def test_MaxMatching(snapshot):
    res = MaxMatching(str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "MaxMatching.txt")


def test_MaxWeightMatching(snapshot):
    res = MaxWeightMatching(str(DATA_PATH / "edge0.csv"))
    snapshot.assert_match(str(res), "MaxWeightMatching.txt")


def test_StableMatching(snapshot):
    res = StableMatching(str(DATA_PATH / "stable.csv"))
    snapshot.assert_match(str(res), "StableMatching.txt")
