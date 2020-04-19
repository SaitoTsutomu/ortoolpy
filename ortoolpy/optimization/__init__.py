import IPython.core.getipython
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import HTML
from more_itertools import first, pairwise

from .. import *


def typical_optimization_impl(sel):
    return HTML(
        """\
<table>
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
"""
    )


def MinimumSpanningTree(
    dfed, from_label="node1", to_label="node2", weight_label="weight", **kwargs
):
    """
    最小全域木問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
    出力
        選択された辺のDataFrame
        (重みの和は、結果のweight_labelのsum())
    """
    g, _, dfed = graph_from_table(
        None, dfed, from_label=from_label, to_label=to_label, from_to="FrTo_", **kwargs
    )
    t = nx.minimum_spanning_tree(g, weight=weight_label)
    dftmp = pd.DataFrame(
        [f"{min(i,j)}-{max(i,j)}" for i, j in t.edges()], columns=["FrTo_"]
    )
    return pd.merge(dfed, dftmp).drop("FrTo_", 1)


def MaximumStableSet(
    dfnd,
    dfed,
    node_label="id",
    weight_label="weight",
    from_label="node1",
    to_label="node2",
    **kwargs,
):
    """
    最大安定集合問題
    入力
        dfnd: 点のDataFrameもしくはCSVファイル名
        dfed: 辺のDataFrameもしくはCSVファイル名
        node_label: 点IDの属性文字
        weight_label: 点の重みの属性文字
        from_label: 元点の属性文字
        to_label: 先点の属性文字
    出力
        選択された点のDataFrame
        (重みの和は、結果のweight_labelのsum())
    """
    g, dfnd, dfed = graph_from_table(
        dfnd,
        dfed,
        node_label=node_label,
        from_label=from_label,
        to_label=to_label,
        **kwargs,
    )
    lst = maximum_stable_set(g, weight=weight_label)[1]
    return dfnd[dfnd[node_label].isin(set(lst))]


def MaximumCut(
    dfnd,
    dfed,
    node_label="id",
    from_label="node1",
    to_label="node2",
    weight_label="weight",
    **kwargs,
):
    """
    最大カット問題
    入力
        dfnd: 点のDataFrameもしくはCSVファイル名
        dfed: 辺のDataFrameもしくはCSVファイル名
        node_label: 点IDの属性文字
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
    出力
        最大カットと片方の集合の点のDataFrame
    """
    g, dfnd, dfed = graph_from_table(
        dfnd,
        dfed,
        node_label=node_label,
        from_label=from_label,
        to_label=to_label,
        **kwargs,
    )
    r, lst = maximum_cut(g, weight=weight_label)
    return r, dfnd[dfnd[node_label].isin(set(lst))]


def DijkstraPath(
    dfed,
    source,
    target,
    from_label="node1",
    to_label="node2",
    weight_label="weight",
    **kwargs,
):
    """
    最短路問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        source: 開始点
        target: 終了点
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
    出力
        最大カットと片方の集合の点のDataFrame
    """
    g, _, dfed = graph_from_table(
        None, dfed, from_label=from_label, to_label=to_label, from_to="FrTo_", **kwargs
    )
    rt = nx.dijkstra_path(g, source, target, weight=weight_label)
    return pd.concat(
        [dfed[dfed.FrTo_ == f"{min(i,j)}-{max(i,j)}"] for i, j in pairwise(rt)]
    ).drop("FrTo_", 1)


def MaximumFlow(
    dfed,
    source,
    target,
    from_label="node1",
    to_label="node2",
    capacity_label="capacity",
    flow_label="flow",
    **kwargs,
):
    """
    最大流問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        source: 開始点
        target: 終了点
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        capacity_label: 辺の容量の属性文字
        flow_label: 辺の流量の属性文字(結果)
    出力
        最大流と辺のDataFrame
    """
    g, _, dfed = graph_from_table(
        None, dfed, from_label=from_label, to_label=to_label, from_to="FrTo_", **kwargs
    )
    r, t = nx.maximum_flow(g, source, target, capacity=capacity_label)
    dftmp = pd.DataFrame(
        [
            (f"{min(i,j)}-{max(i,j)}", f)
            for i, d in t.items()
            for j, f in d.items()
            if f
        ],
        columns=["FrTo_", flow_label],
    )
    return r, pd.merge(dfed, dftmp).drop("FrTo_", 1)


def MinCostFlow(
    dfnd,
    dfed,
    node_label="id",
    demand_label="demand",
    from_label="node1",
    to_label="node2",
    weight_label="weight",
    capacity_label="capacity",
    flow_label="flow",
    **kwargs,
):
    """
    最小費用流問題
    入力
        dfnd: 点のDataFrameもしくはCSVファイル名
        dfed: 辺のDataFrameもしくはCSVファイル名(有向)
        node_label: 点IDの属性文字
        demand_label: 需要の属性文字
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
        capacity_label: 辺の容量の属性文字
        flow_label: 辺の流量の属性文字(結果)
    出力
        辺のDataFrame
    """
    g, dfnd, dfed = graph_from_table(
        dfnd,
        dfed,
        True,
        node_label=node_label,
        from_label=from_label,
        to_label=to_label,
        **kwargs,
    )
    t = nx.min_cost_flow(
        g, demand=demand_label, capacity=capacity_label, weight=weight_label
    )
    dftmp = pd.DataFrame(
        [(i, j, f) for i, d in t.items() for j, f in d.items() if f],
        columns=[from_label, to_label, flow_label],
    )
    return pd.merge(dfed, dftmp)


def Vrp(
    dfnd,
    dfed,
    nv,
    capa,
    node_label="id",
    demand_label="demand",
    from_label="node1",
    to_label="node2",
    cost_label="cost",
    vehicle_label="car",
    series_label="num",
    **kwargs,
):
    """
    運搬経路問題
    入力
        dfnd: 点のDataFrameもしくはCSVファイル名
        dfed: 辺のDataFrameもしくはCSVファイル名(無向)
        nv: 運搬車数
        capa: 運搬車容量
        node_label: 点IDの属性文字
        demand_label: 需要の属性文字
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        cost_label: 費用(per 量)の属性文字
        vehicle_label: 車両の属性文字(結果)
        series_label: 1車両内順番の属性文字(結果)
    出力
        辺のDataFrame
    """
    g, dfnd, dfed = graph_from_table(
        dfnd,
        dfed,
        node_label=node_label,
        from_label=from_label,
        to_label=to_label,
        from_to="FrTo_",
        **kwargs,
    )
    t = vrp(g.to_directed(), nv, capa, demand=demand_label, cost=cost_label)
    dftmp = pd.DataFrame(
        [
            (f"{min(i,j)}-{max(i,j)}", h, k)
            for h, l in enumerate(t)
            for k, (i, j) in enumerate(l)
        ],
        columns=["FrTo_", vehicle_label, series_label],
    )
    return pd.merge(dftmp, dfed).drop("FrTo_", 1)


def Tsp(dfnd, x_label="x", y_label="y", **kwargs):
    """
    巡回セールスマン問題
    入力
        dfnd: 点のDataFrameもしくはCSVファイル名
        x_label: X座標の属性文字
        y_label: Y座標の属性文字
    出力
        総距離と点のDataFrame
    """
    dfnd = graph_from_table(dfnd, None, no_graph=True, **kwargs)[1]
    pos = [p for p in zip(dfnd[x_label], dfnd[y_label])]
    r, t = tsp(pos)
    return r, dfnd.iloc[t]


def ChinesePostman(
    dfed, from_label="node1", to_label="node2", weight_label="weight", **kwargs
):
    """
    中国人郵便配達問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
    出力
        重みの和と選択された辺のDataFrame(複数辺は出力されない)
        (重みの和は、結果のweight_labelのsum())
    """
    g, _, dfed = graph_from_table(
        None,
        dfed,
        multi=True,
        from_label=from_label,
        to_label=to_label,
        from_to="FrTo_",
        **kwargs,
    )
    r, t = chinese_postman(g, weight=weight_label)
    dftmp = pd.DataFrame([(f"{min(i,j)}-{max(i,j)}",) for i, j in t], columns=["FrTo_"])
    return r, pd.merge(dftmp, dfed).drop("FrTo_", 1)


def SetCovering(
    df,
    id_label="id",
    weight_label="weight",
    element_label="element",
    is_partition=False,
    **kwargs,
):
    """
    集合被覆問題
    入力
        df: 候補のDataFrameもしくはCSVファイル名
        id_label: 候補番号の属性文字
        weight_label: 重みの属性文字
        element_label: 要素の属性文字
        is_partition: 集合分割問題かどうか
    出力
        選択された候補リストの番号リスト
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    g = df.groupby(id_label)
    r = set_covering(
        len(g),
        [(r[weight_label].iloc[0], r[element_label].tolist()) for _, r in g],
        is_partition=is_partition,
    )
    df = df[df[id_label].isin(r)].copy()
    df.loc[df[id_label] == df[id_label].shift(1), weight_label] = np.nan
    return df


def SetPartition(
    df, id_label="id", weight_label="weight", element_label="element", **kwargs
):
    """
    集合分割問題
    入力
        df: 候補のDataFrameもしくはCSVファイル名
        id_label: 候補番号の属性文字
        weight_label: 重みの属性文字
        element_label: 要素の属性文字
    出力
        選択された候補リストの番号リスト
    """
    return SetCovering(
        df, id_label, weight_label, element_label, is_partition=True, **kwargs
    )


def CombinatorialAuction(
    df,
    id_label="id",
    price_label="price",
    element_label="element",
    buyer_label="buyer",
    limit=-1,
    **kwargs,
):
    """
    組合せオークション問題
        要素を重複売却せず、購入者ごとの候補数上限を超えないように売却金額を最大化
    入力
        df: 候補のDataFrameもしくはCSVファイル名
        id_label: 候補番号の属性文字
        price_label: 金額の属性文字
        element_label: 要素の属性文字
        buyer_label: 購入者番号の属性文字
        limit: 購入者ごとの候補数上限。-1なら無制限。購入者IDをキーにした辞書可
    出力
        選択された候補リストの番号リスト
        (priceは、同一id内で1つ以外NaNと変える)
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    df = df.sort_values(id_label)
    g = df.groupby(id_label)
    r = combinatorial_auction(
        len(df[element_label].unique()),
        [
            (
                v[price_label].iloc[0],
                v[element_label].tolist(),
                v.iloc[0].get(buyer_label, 0),
            )
            for _, v in g
        ],
        limit=limit,
    )
    df = df[df[id_label].isin(r)].copy()
    df.loc[df[id_label] == df[id_label].shift(1), price_label] = np.nan
    return df


def TwoMachineFlowshop(
    df, first_machine_label="first", second_machine_label="second", **kwargs
):
    """
    入力
        df: ジョブのDataFrameもしくはCSVファイル名
        first_machine_label: 前工程処理時間の属性文字
        second_machine_label: 後工程処理時間の属性文字
    出力
        処理時間と処理順のDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    r, t = two_machine_flowshop(df[[first_machine_label, second_machine_label]].values)
    return r, df.iloc[t]


def ShiftScheduling(ndy, nst, shift, proh, need):
    """
    勤務スケジューリング問題
    入力
        ndy: 日数
        nst: スタッフ数
        shift: シフト(1文字)のリスト
        proh: 禁止パターン(シフトの文字列)のリスト
        need: シフトごとの必要人数リスト(日ごと)
    出力
        日ごとスタッフごとのシフトのDataFrame
    """
    r = shift_scheduling(ndy, nst, shift, proh, need)
    df = pd.DataFrame(
        np.vectorize(lambda i: shift[i])(r),
        columns=[chr(65 + i) for i in range(nst)],
        index=["%d日目" % i for i in range(1, ndy + 1)],
    )
    for sft, lst in need.items():
        df["%s必要" % sft] = lst
        df["%s計画" % sft] = (df.iloc[:, :4] == sft).sum(1)
    return df


def Knapsack(df, capacity, size_label="size", weigh_labelt="weight", **kwargs):
    """
    ナップサック問題
        価値の最大化
    入力
        df: 荷物のDataFrameもしくはCSVファイル名
        capacity: 容量
        size_label: 荷物の大きさの属性文字
        weigh_labelt: 荷物の価値の属性文字
    出力
        選択した荷物のDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    t = knapsack(df[size_label], df[weigh_labelt], capacity)[1]
    return df.reset_index(drop=True).iloc[t]


def BinPacking(df, capacity, size_label="size", bin_label="id", **kwargs):
    """
    ビンパッキング問題
        列生成法で解く(近似解法)
    入力
        df: 荷物のDataFrameもしくはCSVファイル名
        capacity: 容量
        size_label: 荷物の大きさの属性文字
        bin_label: 容器のIDの属性文字(結果)
    出力
        選択した荷物のDataFrame
    """

    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    df = df.reset_index(drop=True)
    t = binpacking(capacity, df[size_label])
    res = []
    for i, tt in enumerate(t):
        for w in tt:
            j = df[df[size_label] == w].index[0]
            res.append(dict(df.loc[j].to_dict(), **{bin_label: i}))
            df.loc[j, size_label] = np.nan
    return pd.DataFrame(res)


def TwoDimPacking(
    df,
    width,
    height,
    width_label="width",
    height_label="height",
    x_label="x",
    y_label="y",
    **kwargs,
):
    """
    2次元パッキング問題
        ギロチンカットで元板からアイテムを切り出す(近似解法)
    入力
        df: アイテムのDataFrameもしくはCSVファイル名
        width: 元板の横
        height: 元板の縦
        width_label: アイテムの横の属性文字
        height_label: アイテムの縦の属性文字
        x_label: アイテムのX座標の属性文字(結果)
        y_label: アイテムのY座標の属性文字(結果)
    出力
        容積率と入ったアイテムのDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    df = df.reset_index(drop=True)
    dt = df[[width_label, height_label]].values
    r, t = TwoDimPackingClass(width, height, dt).solve()
    t = np.array(t)
    df = df.iloc[t[:, 0]]
    df[x_label] = t[:, 3]
    df[y_label] = t[:, 4]
    return r, df


def FacilityLocation(
    df,
    p,
    x_label="x",
    y_label="y",
    demand_label="demand",
    capacity_label="capacity",
    result_label="id",
    func=None,
    **kwargs,
):
    """
    施設配置問題
        P-メディアン問題：総距離×量の和の最小化
    入力
        df: ポイントのDataFrameもしくはCSVファイル名
        p: 施設数上限
        x_label: X座標の属性文字
        y_label: Y座標の属性文字
        demand_label: 需要の属性文字(空なら需要なし)
        capacity_label: 容量の属性文字(空なら候補でない)
        result_label: 施設IDの属性文字(結果,利用時はdropna().astype(int))
        func: 顧客位置index,施設候補indexを引数とする重み関数
    出力
        ポイントのDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    df = df.reset_index(drop=True)
    df1 = df.dropna(subset=[demand_label])[[x_label, y_label, demand_label]]
    df2 = df.dropna(subset=[capacity_label])[[x_label, y_label, capacity_label]]
    t = facility_location(p, df1.values, df2.values, func=func)
    df.loc[df1.index, result_label] = df2.iloc[t].index
    return df


def FacilityLocationWithoutCapacity(
    df,
    p,
    x_label="x",
    y_label="y",
    user_label="demand",
    facility_label="capacity",
    result_label="id",
    func=None,
    **kwargs,
):
    """
    施設配置問題
        P-メディアン問題：総距離×量の和の最小化
    入力
        df: ポイントのDataFrameもしくはCSVファイル名
        p: 施設数上限
        x_label: X座標の属性文字
        y_label: Y座標の属性文字
        user_label: 空でない場合顧客を表す属性文字
        facility_label: 空でない場合施設を表す属性文字
        result_label: 施設IDの属性文字(結果,利用時はdropna().astype(int))
        func: 顧客位置index,施設候補indexを引数とする重み関数
    出力
        ポイントのDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    df = df.reset_index(drop=True)
    df1 = df.dropna(subset=[user_label])[[x_label, y_label]]
    df2 = df.dropna(subset=[facility_label])[[x_label, y_label]]
    t = facility_location_without_capacity(p, df1.values, df2.values, func=func)
    df.loc[df1.index, result_label] = df2.iloc[t].index
    return df


def QuadAssign(
    dfqu,
    dfdi,
    from_label="from",
    to_label="to",
    quant_label="quant",
    dist_label="dist",
    target_label="target",
    pos_label="pos",
    **kwargs,
):
    """
    2次割当問題
        全探索
    入力
        dfqu: 対象間の輸送量のDataFrameもしくはCSVファイル名
        dfdi: 割当先間の距離のDataFrameもしくはCSVファイル名
        from_label: 輸送元番号の属性文字
        to_label: X輸送先番号の属性文字
        quant_label: 必要輸送量の属性文字
        dist_label: 距離の属性文字
        target_label: 対象の属性文字(結果)
        pos_label: 位置の属性文字(結果)
    出力
        評価値と割当
    """
    dfqu = graph_from_table(dfqu, None, no_graph=True, **kwargs)[1]
    dfdi = graph_from_table(dfdi, None, no_graph=True, **kwargs)[1]
    tmp = dfdi.copy()
    tmp[[to_label, from_label]] = tmp[[from_label, to_label]]
    dfdi = pd.concat([dfdi, tmp]).drop_duplicates([from_label, to_label])
    r = range(
        max(
            dfqu[[from_label, to_label]].max().max(),
            dfdi[[from_label, to_label]].max().max(),
        )
        + 1
    )
    q = [
        [
            first(dfqu[(dfqu[from_label] == i) & (dfqu[to_label] == j)][quant_label], 0)
            for j in r
        ]
        for i in r
    ]
    d = [
        [
            first(
                dfdi[(dfdi[from_label] == i) & (dfdi[to_label] == j)][dist_label],
                np.inf,
            )
            for j in r
        ]
        for i in r
    ]
    r, t = quad_assign(q, d)
    return r, pd.DataFrame([i for i in enumerate(t)], columns=[target_label, pos_label])


def Gap(
    df,
    capacity,
    agent_label="agent",
    job_label="job",
    cost_label="cost",
    req_label="req",
    **kwargs,
):
    """
    一般化割当問題
        費用最小の割当を解く
    入力
        df: DataFrameもしくはCSVファイル名
        capacity: エージェントの容量のリスト
        agent_label: エージェントの属性文字
        job_label: ジョブの属性文字
        cost_label: 費用の属性文字
        req_label: 要求量の属性文字
    出力
        選択されたDataFrame
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    a = range(df[agent_label].max() + 1)
    j = range(df[job_label].max() + 1)
    c = [
        [
            first(df[(df[agent_label] == i) & (df[job_label] == k)][cost_label], 0)
            for k in j
        ]
        for i in a
    ]
    r = [
        [
            first(df[(df[agent_label] == i) & (df[job_label] == k)][req_label], 1e6)
            for k in j
        ]
        for i in a
    ]
    t = gap(c, r, capacity)
    return pd.concat(
        [df[(df[agent_label] == i) & (df[job_label] == k)] for k, i in enumerate(t)]
    )


def MaxMatching(dfed, from_label="node1", to_label="node2", **kwargs):
    """
    最大マッチング問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        from_label: 元点の属性文字
        to_label: 先点の属性文字
    出力
        選択された辺のDataFrame
    """
    return MaxWeightMatching(
        dfed, from_label=from_label, to_label=to_label, weight_label="", **kwargs
    )


def MaxWeightMatching(
    dfed, from_label="node1", to_label="node2", weight_label="weight", **kwargs
):
    """
    最大重みマッチング問題
    入力
        dfed: 辺のDataFrameもしくはCSVファイル名
        from_label: 元点の属性文字
        to_label: 先点の属性文字
        weight_label: 辺の重みの属性文字
    出力
        選択された辺のDataFrame
    """
    g, _, dfed = graph_from_table(
        None, dfed, from_label=from_label, to_label=to_label, from_to="FrTo_", **kwargs
    )
    for i, j in g.edges():
        g.adj[i][j]["weight"] = g.adj[i][j].get(weight_label, 1)
    t = nx.max_weight_matching(g)
    dftmp = pd.DataFrame([f"{min(i,j)}-{max(i,j)}" for i, j in t], columns=["FrTo_"])
    return pd.merge(dfed, dftmp).drop("FrTo_", 1)


def StableMatching(
    df,
    male_label="male",
    female_label="female",
    pref_male_label="pref_male",
    pref_female_label="pref_female",
    **kwargs,
):
    """
    安定マッチング問題
    入力
        df: 選好のDataFrameもしくはCSVファイル名
        male_label: 男性の属性文字
        female_label: 女性の属性文字
        pref_male_label: 男性から見た順番の属性文字
        pref_female_label: 女性から見た順番の属性文字
    出力
        マッチングのDataFrame(男性優先)
    """
    df = graph_from_table(df, None, no_graph=True, **kwargs)[1]
    m = range(df[male_label].max() + 1)
    f = range(df[female_label].max() + 1)
    prefm = [
        [
            first(
                df[(df[male_label] == i) & (df[pref_male_label] == j)][female_label],
                1e6,
            )
            for j in f
        ]
        for i in m
    ]
    preff = [
        [
            first(
                df[(df[female_label] == i) & (df[pref_female_label] == j)][male_label],
                1e6,
            )
            for j in m
        ]
        for i in f
    ]
    t = stable_matching(prefm, preff)
    return pd.concat(
        [df[(df[female_label] == i) & (df[male_label] == j)] for i, j in t.items()]
    ).sort_values(male_label)


_ip = IPython.core.getipython.get_ipython()
if _ip:
    _ip.register_magic_function(
        typical_optimization_impl, magic_kind="line", magic_name="typical_optimization"
    )

