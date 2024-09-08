`ortoolpy` is a package for **Operations Research**.

It is user's responsibility for the use of `ortoolpy`.

```python
from ortoolpy import knapsack

size = [21, 11, 15, 9, 34, 25, 41, 52]
weight = [22, 12, 16, 10, 35, 26, 42, 53]
capacity = 100
knapsack(size, weight, capacity)
>>>
(105.0, [0, 1, 3, 4, 5])
```

## Show Table(in jupyterlab, requires networkx)

```python
import ortoolpy.optimization
%typical_optimization
```

<table style="font-size: smaller;">
<tr><td>典型問題クラス</td><td>典型問題</td><td>関数</td><td>典型問題クラス</td><td>典型問題</td><td>関数</td></tr>
<tr>
 <td rowspan="6">グラフ・ネットワーク問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/3130634debf561608bd9" target="_blank">最小全域木問題</a></td>
  <td>MinimumSpanningTree</td>
 <td rowspan="2">スケジューリング問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/d088959bf884d40b2053" target="_blank">ジョブショップ問題</a></td>
  <td>TwoMachineFlowshop</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/33ad03bcaa421eb2ba17" target="_blank">最大安定集合問題</a><br>
     (<a href="http://qiita.com/SaitoTsutomu/items/179db1bd283ab4d926d1" target="_blank">最小頂点被覆問題</a>)</td>
  <td>MaximumStableSet<br>(MinNodeCover)</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/e79ad9ca61a82d5482fa" target="_blank">勤務スケジューリング問題</a></td>
  <td>ShiftScheduling</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/d716413c4b93b22eaad3" target="_blank">最大カット問題</a></td>
  <td>MaximumCut</td>
 <td rowspan="3">切出し・詰込み問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/d3c07494e7ba992bf19d" target="_blank">ナップサック問題</a></td>
  <td>Knapsack</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/565c59fac36badb6a80c" target="_blank">最短路問題</a></td>
  <td>DijkstraPath</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/36788d01fb3be80564a1" target="_blank">ビンパッキング問題</a></td>
  <td>BinPacking</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/80e70da6717acacefa00" target="_blank">最大流問題</a></td>
  <td>MaximumFlow</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/0ac9bd564ae9f91285d7" target="_blank">n次元詰込み問題</a></td>
  <td>TwoDimPacking</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/41d625df63f1946c7216" target="_blank">最小費用流問題</a></td>
  <td>MinCostFlow</td>
 <td rowspan="2">配置問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/c5055be8144e085274c1" target="_blank">施設配置問題</a></td>
  <td>FacilityLocation</td>
</tr>
<tr>
 <td rowspan="3">経路問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/1126e1493ff601a858c9" target="_blank">運搬経路問題</a></td>
  <td>Vrp</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/0cbd2e9a75ef0ecb3269" target="_blank">容量制約なし施設配置問題</a></td>
  <td>FacilityLocationWithoutCapacity</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/def581796ef079e85d02" target="_blank">巡回セールスマン問題</a></td>
  <td>Tsp</td>
 <td rowspan="5">割当・マッチング問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/3814e0bb137be0c18f02" target="_blank">2次割当問題</a></td>
  <td>QuadAssign</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/6b8e4a9c794ff8be110f" target="_blank">中国人郵便配達問題</a></td>
  <td>ChinesePostman</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/329eb7f49af673a19cb8" target="_blank">一般化割当問題</a></td>
  <td>Gap</td>
</tr>
<tr>
 <td rowspan="3">集合被覆・分割問題</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/b1f3a24aaf50afd93e09" target="_blank">集合被覆問題</a></td>
  <td>SetCovering</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/37262bef6f2cab331e01" target="_blank">最大マッチング問題</a></td>
  <td>MaxMatching</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/22ec0e42999141a0ba1e" target="_blank">集合分割問題</a></td>
  <td>SetPartition</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/bbebc69ebc2549b0d5d2" target="_blank">重みマッチング問題</a></td>
  <td>MaxWeightMatching</td>
</tr>
<tr>
  <td><a href="http://qiita.com/SaitoTsutomu/items/614aa24b4025d3f7cc73" target="_blank">組合せオークション問題</a></td>
  <td>CombinatorialAuction</td>
  <td><a href="http://qiita.com/SaitoTsutomu/items/2ec5f7626054f4b4de63" target="_blank">安定マッチング問題</a></td>
  <td>StableMatching</td>
</tr>
</table>

## Requirements

* Python 3, pandas, pulp, more-itertools

## Features

* This is a sample. So it may not be efficient.
* `ortools_vrp` using Google OR-Tools ( https://developers.google.com/optimization/ ).

## Setup

```sh
$ pip install ortoolpy
```

## History

* 0.0.1 (2015-6-26): first release
