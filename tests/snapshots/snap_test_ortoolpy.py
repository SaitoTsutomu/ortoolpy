# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import GenericRepr, Snapshot


snapshots = Snapshot()

snapshots['test_BinPacking 1'] = GenericRepr('   size  id\n0   8.0   0\n1   7.0   1\n2   3.0   1\n3   5.0   2\n4   3.0   2\n5   2.0   2\n6   4.0   3\n7   6.0   3')

snapshots['test_ChinesePostman 1'] = (
    36,
    GenericRepr('    node1  node2  capacity  weight\n0       0      4         2       2\n1       4      5         2       1\n2       4      5         2       1\n3       3      4         2       4\n4       2      3         2       3\n5       2      3         2       3\n6       0      3         2       2\n7       0      5         2       4\n8       1      5         2       5\n9       1      2         2       5\n10      0      2         2       4\n11      0      1         2       1\n12      0      1         2       1')
)

snapshots['test_CombinatorialAuction 1'] = GenericRepr('   id  price element  buyer\n2   1   10.0       a      1\n4   3   14.0       b      1\n5   3    NaN       c      1')

snapshots['test_DijkstraPath 1'] = GenericRepr('   node1  node2  capacity  weight\n9      4      5         2       1\n3      0      4         2       2\n1      0      2         2       4')

snapshots['test_FacilityLocation 1'] = GenericRepr('   x  y  demand  capacity   id\n0  1  0     1.0       1.0  0.0\n1  0  1     NaN       1.0  NaN\n2  0  1     1.0       NaN  3.0\n3  2  2     1.0       2.0  3.0')

snapshots['test_FacilityLocationWithoutCapacity 1'] = GenericRepr('   x  y  demand  capacity   id\n0  1  0     1.0       1.0  1.0\n1  0  1     NaN       1.0  NaN\n2  0  1     1.0       NaN  1.0\n3  2  2     1.0       2.0  3.0')

snapshots['test_Gap 1'] = GenericRepr('   agent  job  cost  req\n0      0    0     2    1\n1      0    1     2    1\n5      1    2     1    1')

snapshots['test_Knapsack 1'] = GenericRepr('   size  weight\n0    21      22\n1    11      12\n3     9      10\n4    34      35\n5    25      26')

snapshots['test_MaxMatching 1'] = GenericRepr('   node1  node2  capacity  weight\n0      0      5         2       4\n1      1      2         2       5\n2      3      4         2       4')

snapshots['test_MaxWeightMatching 1'] = GenericRepr('   node1  node2  capacity  weight\n0      0      2         2       4\n1      1      5         2       5\n2      3      4         2       4')

snapshots['test_MaximumFlow 1'] = (
    6,
    GenericRepr('   node1  node2  capacity  weight  flow\n0      0      2         2       4     2\n1      0      3         2       2     2\n2      0      4         2       2     2\n3      0      5         2       4     2\n4      1      2         2       5     2\n5      1      5         2       5     2\n6      2      3         2       3     2\n7      4      5         2       1     2')
)

snapshots['test_MaximumStableSet 1'] = GenericRepr('   id  x  y  demand  weight\n1   1  5  8       1       3\n4   4  2  2       1       2')

snapshots['test_MinCostFlow 1'] = GenericRepr('   node1  node2  capacity  weight  flow\n0      0      1         2       1     1\n1      0      3         2       2     1\n2      0      4         2       2     2\n3      4      5         2       1     1')

snapshots['test_MinimumSpanningTree 1'] = GenericRepr('   node1  node2  capacity  weight\n0      0      1         2       1\n1      0      3         2       2\n2      0      4         2       2\n3      2      3         2       3\n4      4      5         2       1')

snapshots['test_QuadAssign 1'] = (
    7,
    GenericRepr('   target  pos\n0       0    0\n1       1    1\n2       2    2')
)

snapshots['test_SetCovering 1'] = GenericRepr('   id  weight element\n0   0     1.0       a\n1   0     NaN       b\n2   1     1.0       a\n3   1     NaN       c\n4   2     1.0       a\n5   2     NaN       d')

snapshots['test_SetPartition 1'] = GenericRepr('   id  weight element\n4   2     1.0       a\n5   2     NaN       d\n6   3     3.0       b\n7   3     NaN       c')

snapshots['test_ShiftScheduling 1'] = GenericRepr('     A  B  C  D  日必要  日計画  夜必要  夜計画\n1日目  日  休  夜  日    2    2    1    1\n2日目  夜  日  休  日    2    2    1    1\n3日目  休  日  日  夜    2    2    1    1\n4日目  日  夜  日  休    2    2    1    1\n5日目  日  休  夜  日    2    2    1    1\n6日目  夜  日  休  日    2    2    1    1\n7日目  休  日  日  夜    2    2    1    1\n8日目  日  夜  日  休    2    2    1    1')

snapshots['test_StableMatching 1'] = GenericRepr('   male  female  pref_male  pref_female\n0     0       0          1            0\n4     1       1          1            2\n8     2       2          1            0')

snapshots['test_Tsp 1'] = (
    24.822702473776797,
    GenericRepr('   id  x  y  demand\n0   0  5  1       0\n4   4  8  0       1\n1   1  8  5       1\n2   2  1  5       1\n5   5  0  4       1\n3   3  1  0       1')
)

snapshots['test_TwoDimPacking 1'] = (
    1.0,
    GenericRepr('   width  height    x    y\n0    240     150    0    0\n1    260     100  240    0\n2    100     200  240  100\n3    240     150    0  150\n4    160     200  340  100')
)

snapshots['test_TwoMachineFlowshop 1'] = (
    GenericRepr('9'),
    GenericRepr('   first  second\n2      1       4\n0      4       3\n1      3       1')
)

snapshots['test_Vrp 1'] = GenericRepr('   car  num  node1  node2  cost\n0    0    0      0      2    10\n1    0    1      5      2     1\n2    0    2      0      3    10\n3    0    3      3      5     1\n4    1    0      0      1    10\n5    1    1      4      1     1\n6    1    2      0      4    10')
