"""
Copyright: 2015-2020 Saito Tsutomu
License: Python Software Foundation License
"""
from collections import defaultdict
from math import ceil, sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from more_itertools import always_iterable, pairwise

# fmt: off
from pulp import (LpBinary, LpInteger, LpMaximize, LpMinimize, LpProblem,
                  LpVariable, lpDot, lpSum, value)

# fmt: on
iterable = lambda a: isinstance(a, Iterable)


def L(s, i=[-1]):
    i[0] += 1
    return "%s_%s" % (s, i[0])


def always_dataframe_list(df):
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        return [df]
    return df


def _obj_expr(df, objs):
    return (
        lpDot(-df[obj[1:]] if obj[0] == "-" else df[obj], df.Var)
        for obj in objs
        if obj.lstrip("-") in df
    )


class LpProblemEx(LpProblem):
    def __init__(self, *n, df=None, dfb=None, dfi=None, **ad):
        super().__init__(*n, **ad)
        self._df = always_dataframe_list(df)
        self._dfb = always_dataframe_list(dfb)
        self._dfi = always_dataframe_list(dfi)
        if self._df:
            addvars(*self._df)
        if self._dfb:
            addbinvars(*self._dfb)
        if self._dfi:
            addintvars(*self._dfi)

    def solve(self, *n, objs=None, **ad):
        dfs = self._df + self._dfb + self._dfi
        if objs:
            objs = list(always_iterable(objs))
            self += lpSum(_obj_expr(df, objs) for df in dfs if "Var" in df)
        super().solve(*n, **ad)
        if self.status == 1:
            for df in dfs:
                if "Var" in df:
                    addvals(df)
        else:
            for df in dfs:
                if "Val" in df:
                    df.drop("Val", 1, inplace=True)
        return self.status


def model(*n, ismin=True, df=None, dfb=None, dfi=None, **ad):
    """最小化のモデル"""
    ad["sense"] = LpMinimize if ismin else LpMaximize
    m = LpProblemEx(*n, **ad, df=df, dfb=dfb, dfi=dfi)
    return m


def model_min(*n, df=None, dfb=None, dfi=None, **ad):
    """最小化のモデル"""
    return model(*n, **ad, ismin=True, df=df, dfb=dfb, dfi=dfi)


def model_max(*n, df=None, dfb=None, dfi=None, **ad):
    """最大化のモデル"""
    return model(*n, **ad, ismin=False, df=df, dfb=dfb, dfi=dfi)


def addvals(*n, Var="Var", Val="Val"):
    """結果の列を作成"""
    for df in n:
        df[Val] = df[Var].apply(value)
    return n


def addvar(name=None, *, var_count=[0], lowBound=0, format="v%.6d", **kwargs):
    """変数作成用ユーティリティ"""
    if not name:
        var_count[0] += 1
        name = format % var_count[0]
    if "lowBound" not in kwargs:
        kwargs["lowBound"] = lowBound
    return LpVariable(name, **kwargs)


def addvars(*n, **ad):
    """配列変数作成用ユーティリティ"""
    if n and all(isinstance(df, pd.DataFrame) for df in n):
        s = ad.pop("Var", "Var")
        for df in n:
            df[s] = addvars(len(df), **ad)
        return n
    va = []
    _addvarsRec(va, *n, **ad)
    return va


def addbinvar(*n, **ad):
    """0-1変数作成用ユーティリティ"""
    return addvar(*n, cat=LpBinary, **ad)


def addbinvars(*n, **ad):
    """0-1配列変数作成用ユーティリティ"""
    return addvars(*n, cat=LpBinary, **ad)


def addintvar(*n, **ad):
    """整数変数作成用ユーティリティ"""
    return addvar(*n, cat=LpInteger, **ad)


def addintvars(*n, **ad):
    """整数配列変数作成用ユーティリティ"""
    return addvars(*n, cat=LpInteger, **ad)


def _addvarsRec(va, *n, **ad):
    if n == ():
        return None
    b = len(n) == 1
    for i in range(n[0]):
        if b:
            va.append(addvar(**ad))
        else:
            nva = []
            _addvarsRec(nva, *n[1:], **ad)
            va.append(nva)


def addline(m, p1, p2, x, y, upper=True):
    """2点直線制約"""
    from pulp import LpConstraint, LpConstraintGE, LpConstraintLE

    dx = p2[0] - p1[0]
    if dx != 0:
        m += LpConstraint(
            y - (p2[1] - p1[1]) / dx * x - (p2[0] * p1[1] - p1[0] * p2[1]) / dx,
            LpConstraintGE if upper else LpConstraintLE,
        )


def addlines_conv(m, curve, x, y, upper=True):
    """区分線形制約(凸)"""
    for p1, p2 in pairwise(curve):
        addline(m, p1, p2, x, y, upper)


def addlines(m, curve, x, y):
    """区分線形制約(非凸)"""
    n = len(curve)
    w = addvars(n - 1)
    z = addbinvars(n - 2)
    a = [p[0] for p in curve]
    b = [p[1] for p in curve]
    m += x == a[0] + lpSum(w)
    c = [(b[i + 1] - b[i]) / (a[i + 1] - a[i]) for i in range(n - 1)]
    m += y == b[0] + lpDot(c, w)
    for i in range(n - 1):
        if i < n - 2:
            m += (a[i + 1] - a[i]) * z[i] <= w[i]
        m += w[i] <= (a[i + 1] - a[i]) * (1 if i == 0 else z[i - 1])


def value_or_zero(x):
    """value or 0"""
    v = value(x) if x else None
    return v if v is not None else 0


def random_model(nv, nc, sense=1, seed=1, rtfr=0, rtco=0.1, rteq=0, feasible=False):
    """
    ランダムなLP作成
    nv,nc:変数数、制約条件数
    sense:種類(LpMinimize=1)
    seed:乱数シード
    rtfr:自由変数割合
    rtco:係数行列非零割合
    rteq:制約条件等号割合
    feasible:実行可能かどうか
    """
    from pulp import LpConstraint, LpStatusOptimal

    rtco = max(1e-3, rtco)
    if seed:
        np.random.seed(seed)
    var_count = [0]
    x = np.array(addvars(nv, var_count=var_count))
    while True:
        m = LpProblem(sense=sense)
        m += lpDot(np.random.rand(nv) * 2 - 1, x)
        for v in x[np.random.rand(nv) < rtfr]:
            v.lowBound = None
        rem = nc
        while rem > 0:
            a = np.random.rand(nv) * 2 - 1
            a[np.random.rand(nv) >= rtco] = 0
            if (a == 0).all():
                continue
            if (a <= 0).all():
                a = -a
            se = int(np.random.rand() >= rteq) * np.random.choice([-1, 1])
            m += LpConstraint(lpDot(a, x), se, rhs=np.random.rand())
            rem -= 1
        if not feasible or m.solve() == LpStatusOptimal:
            return m


def dual_model(m, ignoreUpBound=False):
    """双対問題作成"""
    coe = 1 if m.sense == LpMinimize else -1
    nwmdl = LpProblem(sense=LpMaximize if m.sense == LpMinimize else LpMinimize)
    ccs = [[], [], []]
    for cns in m.constraints.values():
        ccs[cns.sense + 1].append(cns)
    for v in m.variables():
        if v.lowBound is not None:
            ccs[2].append(v >= v.lowBound)
        if not ignoreUpBound and v.upBound is not None:
            ccs[0].append(v <= v.upBound)
    var_count = [0]
    vvs = [
        addvars(len(cc), var_count=var_count, lowBound=0 if i != 1 else None)
        for i, cc in enumerate(ccs)
    ]
    nwmdl += coe * lpSum(
        k * lpDot([c.constant for c in cc], vv)
        for k, cc, vv in zip([1, -1, -1], ccs, vvs)
    )
    for w in m.variables():
        nwmdl += lpSum(
            k * c.get(w) * v
            for k, cc, vv in zip([-1, 1, 1], ccs, vvs)
            for c, v in zip(cc, vv)
            if c.get(w)
        ) == coe * m.objective.get(w, 0)
    return nwmdl


def dual_model_nonzero(m, ignoreUpBound=False):
    """双対問題作成(ただし、全て非負変数)"""
    assert all(v.lowBound == 0 for v in m.variables()), "Must be lowBound==0"
    if not ignoreUpBound:
        assert all(v.upBound is None for v in m.variables())
    coe = 1 if m.sense == LpMinimize else -1
    nwmdl = LpProblem(sense=LpMaximize if m.sense == LpMinimize else LpMinimize)
    ccs = [[], [], []]
    for cns in m.constraints.values():
        ccs[cns.sense + 1].append(cns)
    var_count = [0]
    vvs = [
        addvars(len(cc), var_count=var_count, lowBound=0 if i != 1 else None)
        for i, cc in enumerate(ccs)
    ]
    nwmdl += coe * lpSum(
        k * lpDot([c.constant for c in cc], vv)
        for k, cc, vv in zip([1, -1, -1], ccs, vvs)
    )
    for w in m.variables():
        nwmdl += lpSum(
            k * c.get(w) * v
            for k, cc, vv in zip([-1, 1, 1], ccs, vvs)
            for c, v in zip(cc, vv)
            if c.get(w)
        ) <= coe * m.objective.get(w, 0)
    return nwmdl


def satisfy(m, eps=1e-6):
    """モデルが実行可能かどうか"""
    for x in m.variables():
        v = value(x)
        if x.lowBound is not None and v < x.lowBound - eps:
            return False
        if x.upBound is not None and v > x.upBound + eps:
            return False
        if x.cat == LpInteger and ceil(v - eps * 2) - v > eps:
            return False
    for e in m.constraints.values():
        v = value(e)
        if e.sense >= 0 and v < -eps:
            return False
        if e.sense <= 0 and v > eps:
            return False
    return True


def graph_from_table(
    dfnd,
    dfed,
    directed=False,
    multi=False,
    node_label="id",
    from_label="node1",
    to_label="node2",
    from_to=None,
    no_graph=False,
    **kwargs
):
    """
    表からグラフを作成
    Excelの場合：[Excelファイル名]シート名
    from_to: 'from-to'となる列を追加（ただしfrom < to）
    """
    import re
    import networkx as nx

    if isinstance(dfnd, str):
        m = re.match(r"\[([^]]+)](\w+)", dfnd)
        if m:
            dfnd = pd.read_excel(m.group(1), m.group(2), **kwargs)
        else:
            dfnd = pd.read_csv(dfnd, **kwargs)
    if isinstance(dfed, str):
        m = re.match(r"\[([^]]+)](\w+)", dfed)
        if m:
            dfed = pd.read_excel(m.group(1), m.group(2), **kwargs)
        else:
            dfed = pd.read_csv(dfed, **kwargs)
    g = None
    if not no_graph:
        if multi:
            g = nx.MultiDiGraph() if directed else nx.MultiGraph()
        else:
            g = nx.DiGraph() if directed else nx.Graph()
        if dfnd is not None:
            for r in dfnd.itertuples(False):
                dc = r._asdict()
                g.add_node(dc[node_label], **dc)
        if from_to:
            dfft = dfed[[from_label, to_label]]
            dfed[from_to] = dfft.min(1).astype(str) + "-" + dfft.max(1).astype(str)
        for r in dfed.itertuples(False):
            dc = r._asdict()
            g.add_edge(dc[from_label], dc[to_label], **dc)
    return g, dfnd, dfed


def networkx_draw(g, dcpos=None, node_label="id", x_label="x", y_label="y", **kwargs):
    """グラフを描画"""
    import networkx as nx

    if not dcpos:
        dcpos = {r[node_label]: (r[x_label], r[y_label]) for i, r in g.nodes.items()}
    nx.draw_networkx_nodes(g, dcpos, **kwargs)
    nx.draw_networkx_edges(g, dcpos)
    nx.draw_networkx_labels(g, dcpos)
    return dcpos


def maximum_stable_set(g, weight="weight"):
    """
    最大安定集合問題
    入力
        g: グラフ(node:weight)
        weight: 重みの属性文字
    出力
        最大安定集合の重みの合計と頂点番号リスト
    """
    m = LpProblem(sense=LpMaximize)
    v = [addvar(cat=LpBinary) for _ in g.nodes()]
    for i, j in g.edges():
        m += v[i] + v[j] <= 1
    m += lpDot([g.nodes[i].get(weight, 1) for i in g.nodes()], v)
    if m.solve() != 1:
        return None
    return value(m.objective), [i for i, x in enumerate(v) if value(x) > 0.5]


def min_node_cover(g, weight="weight"):
    """
    最小頂点被覆問題
    入力
        g: グラフ
        weight: 重みの属性文字
    出力
        頂点リスト
    """
    return list(set(g.nodes()) - set(maximum_stable_set(g, weight)[1]))


def maximum_cut(g, weight="weight"):
    """
    最大カット問題
    入力
        g: グラフ(node:weight)
        weight: 重みの属性文字
    出力
        カットの重みの合計と片方の頂点番号リスト
    """
    m = LpProblem(sense=LpMaximize)
    v = [addvar(cat=LpBinary) for _ in g.nodes()]
    u = []
    for i in range(g.number_of_nodes()):
        for j in range(i + 1, g.number_of_nodes()):
            w = g.get_edge_data(i, j, {weight: None}).get(weight, 1)
            if w:
                t = addvar()
                u.append(w * t)
                m += t <= v[i] + v[j]
                m += t <= 2 - v[i] - v[j]
    m += lpSum(u)
    if m.solve() != 1:
        return None
    return value(m.objective), [i for i, x in enumerate(v) if value(x) > 0.5]


def get_route(manager, routing, solution, vehicle_id=0):
    index = routing.Start(vehicle_id)
    while not routing.IsEnd(index):
        yield manager.IndexToNode(index)
        index = solution.Value(routing.NextVar(index))
    yield manager.IndexToNode(index)


def get_callback_index1(manager, routing, dct):
    def _callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return dct[from_node]

    return routing.RegisterUnaryTransitCallback(_callback)


def get_callback_index2(manager, routing, dct):
    if isinstance(dct, list):

        def _callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dct[from_node][to_node]

    else:

        def _callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dct[from_node, to_node]

    return routing.RegisterTransitCallback(_callback)


def vrp(g, nv, capa, demand="demand", cost="cost", method=None):
    """
    運搬経路問題
    入力
        g: グラフ(node:demand, edge:cost)
        nv: 運搬車数
        capa: 運搬車容量
        demand: 需要の属性文字
        cost: 費用の属性文字
        method: 計算方法(ex. 'ortools')
    出力
        運搬車ごとの頂点対のリスト
    """
    if method == "ortools":
        import networkx as nx

        dd = nx.to_numpy_matrix(g, weight=cost, nonedge=999999).astype(int)
        dem = nx.get_node_attributes(g, demand)
        routes = ortools_vrp(len(dd), dd, nv, capa, dem)
        return [list(pairwise(route)) for route in routes]
    from itertools import islice

    rv = range(nv)
    m = LpProblem()
    x = [{(i, j): addvar(cat=LpBinary) for i, j in g.edges()} for _ in rv]
    w = [[addvar() for i in g.nodes()] for _ in rv]
    m += lpSum(g.adj[i][j][cost] * lpSum(x[v][i, j] for v in rv) for i, j in g.edges())
    for v in rv:
        xv, wv = x[v], w[v]
        m += lpSum(xv[0, j] for j in g.nodes() if j) == 1
        for h in g.nodes():
            m += wv[h] <= capa
            m += lpSum(xv[i, j] for i, j in g.edges() if i == h) == lpSum(
                xv[i, j] for i, j in g.edges() if j == h
            )
        for i, j in g.edges():
            if i == 0:
                m += wv[j] >= g.nodes[j][demand] - capa * (1 - xv[i, j])
            else:
                m += wv[j] >= wv[i] + g.nodes[j][demand] - capa * (1 - xv[i, j])
    for h in islice(g.nodes(), 1, None):
        m += lpSum(x[v][i, j] for v in rv for i, j in g.edges() if i == h) == 1
    if m.solve() != 1:
        return None
    return [[(i, j) for i, j in g.edges() if value(x[v][i, j]) > 0.5] for v in rv]


def ortools_vrp(nn, dist, nv=1, capa=1000, demands=None, depo=0, limit_time=0):
    """
    運搬経路問題
    入力
        nn: 地点数
        dist: (i, j)をキー、距離を値とした辞書
        nv: 運搬車数
        capa: 運搬車容量
        demands: 需要
        depo: デポ
        limit_time: 計算時間制限（use GUIDED_LOCAL_SEARCH）
    出力
        運搬車ごとのルート
    """
    assert isinstance(dist[0, 1], (int, np.int32, np.int64)), "Distance must be int."
    try:
        from ortools.constraint_solver import routing_enums_pb2
        from ortools.constraint_solver import pywrapcp
    except ModuleNotFoundError:
        print('Please "pip install ortools"')
        raise
    manager = pywrapcp.RoutingIndexManager(nn, nv, depo)
    routing = pywrapcp.RoutingModel(manager)
    routing.SetArcCostEvaluatorOfAllVehicles(
        get_callback_index2(manager, routing, dist)
    )
    if demands is not None:
        routing.AddDimension(
            get_callback_index1(manager, routing, demands), 0, capa, True, "Capacity"
        )
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    if limit_time:
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.guided_local_search_lambda_coefficient = 0.1
        search_parameters.time_limit.seconds = limit_time
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None
    return [list(get_route(manager, routing, solution, i)) for i in range(nv)]


def tsp(nodes, dist=None, method=None):
    """
    巡回セールスマン問題
    入力
        nodes: 点(dist未指定時は、座標)のリスト
        dist: (i, j)をキー、距離を値とした辞書
        method: 計算方法(ex. 'ortools')
    出力
        距離と点番号リスト
    """
    if dist is None:
        from scipy.spatial import distance

        dist = distance.cdist(nodes, nodes)
    if method == "ortools":  # Approximate solution
        route = ortools_vrp(len(dist), dist)[0]
        return sum(dist[i, j] for i, j in pairwise(route)), route[:-1]
    return tsp1(nodes, dist)


def tsp1(nodes, dist):
    """
    巡回セールスマン問題
    入力
        nodes: 点(dist未指定時は、座標)のリスト
        dist: (i,j)をキー、距離を値とした辞書
    出力
        距離と点番号リスト
    """
    from more_itertools import iterate, take

    n = len(nodes)
    df = pd.DataFrame(
        [(i, j, dist[i, j]) for i in range(n) for j in range(n) if i != j],
        columns=["NodeI", "NodeJ", "Dist"],
    )
    m = LpProblem()
    df["VarIJ"] = addbinvars(len(df))
    df["VarJI"] = df.sort_values(["NodeJ", "NodeI"]).VarIJ.values
    u = [0] + addvars(n - 1)
    m += lpDot(df.Dist, df.VarIJ)
    for _, v in df.groupby("NodeI"):
        m += lpSum(v.VarIJ) == 1  # 出次数制約
        m += lpSum(v.VarJI) == 1  # 入次数制約
    for i, j, _, vij, vji in df.query("NodeI!=0 & NodeJ!=0").itertuples(False):
        m += u[i] + 1 - (n - 1) * (1 - vij) + (n - 3) * vji <= u[j]  # 持ち上げポテンシャル制約(MTZ)
    for _, j, _, v0j, vj0 in df.query("NodeI==0").itertuples(False):
        m += 1 + (1 - v0j) + (n - 3) * vj0 <= u[j]  # 持ち上げ下界制約
    for i, _, _, vi0, v0i in df.query("NodeJ==0").itertuples(False):
        m += u[i] <= (n - 1) - (1 - vi0) - (n - 3) * v0i  # 持ち上げ上界制約
    m.solve()
    df["ValIJ"] = df.VarIJ.apply(value)
    dc = df[df.ValIJ > 0.5].set_index("NodeI").NodeJ.to_dict()
    return value(m.objective), list(take(n, iterate(lambda k: dc[k], 0)))


def tsp2(pos):
    """
    巡回セールスマン問題
    入力
        pos: 座標のリスト
    出力
        距離と点番号リスト
    """
    pos = np.array(pos)
    N = len(pos)
    m = LpProblem()
    v = {}
    for i in range(N):
        for j in range(i + 1, N):
            v[i, j] = v[j, i] = LpVariable("v%d%d" % (i, j), cat=LpBinary)
    m += lpDot(
        [np.linalg.norm(pos[i] - pos[j]) for i, j in v if i < j],
        [x for (i, j), x in v.items() if i < j],
    )
    for i in range(N):
        m += lpSum(v[i, j] for j in range(N) if i != j) == 2
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                m += v[i, j] + v[j, k] + v[k, i] <= 2
    st = set()
    while True:
        m.solve()
        u = unionfind(N)
        for i in range(N):
            for j in range(i + 1, N):
                if value(v[i, j]) > 0:
                    u.unite(i, j)
        gg = u.groups()
        if len(gg) == 1:
            break
        for g_ in gg:
            g = tuple(g_)
            if g not in st:
                st.add(g)
                m += (
                    lpSum(
                        v[i, j]
                        for i in range(N)
                        for j in range(i + 1, N)
                        if (i in g and j not in g) or (i not in g and j in g)
                    )
                    >= 1
                )
                break
    cn = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if value(v[i, j]) > 0:
                if i or cn[i] == 0:
                    cn[i] += j
                cn[j] += i
    p, q, r = cn[0], 0, [0]
    while p != 0:
        r.append(p)
        q, p = p, cn[p] - q
    return value(m.objective), r


def tsp3(point):
    from itertools import permutations

    n = len(point)
    bst, mn = None, 1e100
    for d in permutations(range(1, n)):
        e = [point[i] for i in [0] + list(d) + [0]]
        s = sum(
            sqrt((e[i][0] - e[i + 1][0]) ** 2 + (e[i][1] - e[i + 1][1]) ** 2)
            for i in range(n)
        )
        if s < mn:
            mn = s
            bst = [0] + list(d)
    return mn, bst


def chinese_postman(g, weight="weight"):
    """
    中国人郵便配達問題
    入力
        g: 無向グラフ
        weight: 重みの属性文字
    出力
        距離と頂点リスト
    """
    import networkx as nx
    from itertools import combinations

    assert not g.is_directed()
    g = nx.MultiGraph(g)
    subnd = [nd for nd, dg in g.degree() if dg % 2 == 1]  # 奇数次数ノード群
    dd = nx.floyd_warshall(g, weight=weight)  # 完全距離表
    mx = max(d for dc in dd.values() for d in dc.values())  # 最大距離
    h = nx.Graph()
    for i, j in combinations(subnd, 2):
        h.add_edge(i, j, weight=mx - dd[i][j])
    for i, j in nx.max_weight_matching(h, True):  # 最大重み最大マッチング問題
        g.add_edge(i, j, weight=dd[i][j])
    return (
        sum(d[weight] for (i, j, _), d in g.edges.items()),
        list(nx.eulerian_circuit(g)),
    )


def set_covering(n, cand, is_partition=False):
    """
    集合被覆問題
    入力
        n: 要素数
        cand: (重み, 部分集合)の候補リスト
        is_partition: 集合分割問題かどうか
    出力
        選択された候補リストの番号リスト
    """
    ad = AutoCountDict()
    m = LpProblem()
    vv = [addvar(cat=LpBinary) for _ in cand]
    m += lpDot([w for w, _ in cand], vv)  # obj func
    ee = [[] for _ in range(n)]
    for v, (_, c) in zip(vv, cand):
        for k in c:
            ee[ad[k]].append(v)
    for e in ee:
        if e:
            if is_partition:
                m += lpSum(e) == 1
            else:
                m += lpSum(e) >= 1
    if m.solve() != 1:
        return None
    return [i for i, v in enumerate(vv) if value(v) > 0.5]


def set_partition(n, cand):
    """
    集合分割問題
    入力
        n: 要素数
        cand: (重み, 部分集合)の候補リスト
    出力
        選択された候補リストの番号リスト
    """
    return set_covering(n, cand, True)


def combinatorial_auction(n, cand, limit=-1):
    """
    組合せオークション
      要素を重複売却せず、購入者ごとの候補数上限を超えないように売却金額を最大化
    入力
        n: 要素数
        cand: (金額, 部分集合, 購入者ID)の候補リスト。購入者IDはなくてもよい
        limit: 購入者ごとの候補数上限。-1なら無制限。購入者IDをキーにした辞書可
    出力
        選択された候補リストの番号リスト
    """
    dcv = defaultdict(list)  # buyer別候補変数
    dcc = defaultdict(list)  # item別候補
    isdgt = isinstance(limit, int)
    # buyer別候補数上限
    dcl = defaultdict(lambda: limit if isdgt else -1, {} if isdgt else limit)
    m = LpProblem(sense=LpMaximize)
    x = addbinvars(len(cand))  # 候補を選ぶかどうか
    m += lpDot([ca[0] for ca in cand], x)  # 目的関数
    for v, ca in zip(x, cand):
        bid = ca[2] if len(ca) > 2 else 0
        dcv[bid].append(v)
        for k in ca[1]:
            dcc[k].append(v)
    for k, v in dcv.items():
        ll = dcl[k]
        if ll >= 0:
            m += lpSum(v) <= ll  # 購入者ごとの上限
    for v in dcc.values():
        m += lpSum(v) <= 1  # 要素の売却先は1まで
    if m.solve() != 1:
        return None
    return [i for i, v in enumerate(x) if value(v) > 0.5]


def two_machine_flowshop(p):
    """
    2機械フローショップ問題
        2台のフローショップ型のジョブスケジュールを求める(ジョンソン法)
    入力
        p: (前工程処理時間, 後工程処理時間)の製品ごとのリスト
    出力
        処理時間と処理順のリスト
    """

    def proctime(p, ll):
        n = len(p)
        t = [[0, 0] for _ in range(n + 1)]
        for i in range(1, n + 1):
            t1, t2 = p[ll[i - 1]]
            t[i][0] = t[i - 1][0] + t1
            t[i][1] = max(t[i - 1][1], t[i][0]) + t2
        return t[n][1]

    a, l1, l2 = np.array(p, dtype=float).flatten(), [], []
    for _ in range(a.size // 2):
        j = a.argmin()
        k = j // 2
        if j % 2 == 0:
            l1.append(k)
        else:
            l2.append(k)
        a[2 * k] = a[2 * k + 1] = np.inf
    ll = l1 + l2[::-1]
    return proctime(p, ll), ll


def shift_scheduling(ndy, nst, shift, proh, need):
    """
    勤務スケジューリング問題
    入力
        ndy: 日数
        nst: スタッフ数
        shift: シフト(1文字)のリスト
        proh: 禁止パターン(シフトの文字列)のリスト
        need: シフトごとの必要人数リスト(日ごと)
    出力
        日ごとスタッフごとのシフトの番号のテーブル
    """
    nsh = len(shift)
    rdy, rst, rsh = range(ndy), range(nst), range(nsh)
    dsh = {sh: k for k, sh in enumerate(shift)}
    m = LpProblem()
    v = [[[addvar(cat=LpBinary) for _ in rsh] for _ in rst] for _ in rdy]
    for i in rdy:
        for j in rst:
            m += lpSum(v[i][j]) == 1
        for sh, dd in need.items():
            m += lpSum(v[i][j][dsh[sh]] for j in rst) >= dd[i]
    for prh in proh:
        n, pr = len(prh), [dsh[sh] for sh in prh]
        for j in rst:
            for i in range(ndy - n + 1):
                m += lpSum(v[i + h][j][pr[h]] for h in range(n)) <= n - 1
    if m.solve() != 1:
        return None
    return [[int(value(lpDot(rsh, v[i][j]))) for j in rst] for i in rdy]


def knapsack(size, weight, capacity):
    """
    ナップサック問題
        価値の最大化
    入力
        size: 荷物の大きさのリスト
        weight: 荷物の価値のリスト
        capacity: 容量
    出力
        価値の総和と選択した荷物番号リスト
    """
    m = LpProblem(sense=LpMaximize)
    v = [addvar(cat=LpBinary) for _ in size]
    m += lpDot(weight, v)
    m += lpDot(size, v) <= capacity
    if m.solve() != 1:
        return None
    return value(m.objective), [i for i in range(len(size)) if value(v[i]) > 0.5]


def binpacking(c, w):
    """
    ビンパッキング問題
        列生成法で解く(近似解法)
    入力
        c: ビンの大きさ
        w: 荷物の大きさのリスト
    出力
        ビンごとの荷物の大きさリスト
    """
    from pulp import LpAffineExpression

    n = len(w)
    rn = range(n)
    mkp = LpProblem("knapsack", LpMaximize)  # 子問題
    mkpva = [addvar(cat=LpBinary) for _ in rn]
    mkp.addConstraint(lpDot(w, mkpva) <= c)
    mdl = LpProblem("dual", LpMaximize)  # 双対問題
    mdlva = [addvar() for _ in rn]
    for i, v in enumerate(mdlva):
        v.w = w[i]
    mdl.setObjective(lpSum(mdlva))
    for i in rn:
        mdl.addConstraint(mdlva[i] <= 1)
    while True:
        mdl.solve()
        mkp.setObjective(lpDot([value(v) for v in mdlva], mkpva))
        mkp.solve()
        if mkp.status != 1 or value(mkp.objective) < 1 + 1e-6:
            break
        mdl.addConstraint(lpDot([value(v) for v in mkpva], mdlva) <= 1)
    nwm = LpProblem("primal", LpMinimize)  # 主問題
    nm = len(mdl.constraints)
    rm = range(nm)
    nwmva = [addvar(cat=LpBinary) for _ in rm]
    nwm.setObjective(lpSum(nwmva))
    dict = {}
    for v, q in mdl.objective.items():
        dict[v] = LpAffineExpression() >= q
    const = list(mdl.constraints.values())
    for i, q in enumerate(const):
        for v in q:
            dict[v].addterm(nwmva[i], 1)
    for q in dict.values():
        nwm.addConstraint(q)
    nwm.solve()
    if nwm.status != 1:
        return None
    w0, result = list(w), [[] for _ in range(len(const))]
    for i, va in enumerate(nwmva):
        if value(va) < 0.5:
            continue
        for v in const[i]:
            if v.w in w0:
                w0.remove(v.w)
                result[i].append(v.w)
    return [r for r in result if r]


class TwoDimPackingClass:
    """
    2次元パッキング問題
        ギロチンカットで元板からアイテムを切り出す(近似解法)
    入力
        width, height: 元板の大きさ
        items: アイテムの(横,縦)のリスト
    出力
        容積率と入ったアイテムの(id,横,縦,x,y)のリスト
    """

    def __init__(self, width, height, items=None):
        self.width = width
        self.height = height
        self.items = items

    @staticmethod
    def calc(pp, w, h):
        plw, plh, ofw, ofh = pp
        if w > plw or h > plh:
            return None
        if w * (plh - h) <= h * (plw - w):
            return (
                w * (plh - h),
                (w, plh - h, ofw, ofh + h),
                (plw - w, plh, ofw + w, ofh),
            )
        else:
            return (
                h * (plw - w),
                (plw - w, h, ofw + w, ofh),
                (plw, plh - h, ofw, ofh + h),
            )

    def solve(self, iters=100, seed_=1):
        from random import shuffle, seed

        bst, self.pos = 0, []
        seed(seed_)
        for cnt in range(iters):
            tmp, szs = [], [(k, w, h) for k, (w, h) in enumerate(self.items)]
            plates = [(self.width, self.height, 0, 0)]
            shuffle(szs)
            while len(szs) > 0 and len(plates) > 0:
                mni, mnr, (k, w, h), szs = -1, [1e9], szs[0], szs[1:]
                for i in range(len(plates)):
                    res = TwoDimPackingClass.calc(plates[i], w, h)
                    if res and res[0] < mnr[0]:
                        mni, mnr = i, res
                if mni >= 0:
                    tmp.append((k, w, h) + plates[i][2:])
                    plates[i : i + 1] = [p for p in mnr[1:3] if p[0] * p[1] > 0]
            sm = sum(r[1] * r[2] for r in tmp)
            if sm > bst:
                bst, self.result = sm, tmp
        self.rate = bst / self.width / self.height
        return self.rate, self.result


def ordered_binpacking_sub(w, c, return_index=False):
    r, n = 0, 1
    if return_index:
        idx = [0]
    for i, v in enumerate(w):
        if not r or r + v <= c:
            r += v
        else:
            r, n = v, n + 1
            if return_index:
                idx.append(i)
    return (n, idx) if return_index else n


def ordered_binpacking(n, w, tol=1):
    """
    順番を維持したビンパッキング平準化問題
    入力
        n: ビン数
        w: 荷物の大きさのリスト
    出力
        区切り位置（n + 1個）
    """
    w = [ceil(v / tol) for v in w]
    cl, cu = 1, sum(w)
    if ordered_binpacking_sub(w, cl) <= n:
        cu = cl
    while cu - cl > 1:
        mid = (cu + cl + 1) // 2
        m = ordered_binpacking_sub(w, mid)
        if m > n:
            cl = mid
        else:
            cu = mid
    idx = ordered_binpacking_sub(w, cu, True)[1]
    while len(idx) <= n:
        idx.append(len(w))
    return idx


def facility_location(p, point, cand, func=None):
    """
    施設配置問題
        p-メディアン問題：総距離×量の和の最小化
    入力
        p: 施設数上限
        point: 顧客位置と量のリスト
        cand: 施設候補位置と容量のリスト
        func: 顧客位置index,施設候補indexを引数とする重み関数
    出力
        顧客ごとの施設番号リスト
    """
    if not func:
        func = lambda i, j: sqrt(
            (point[i][0] - cand[j][0]) ** 2 + (point[i][1] - cand[j][1]) ** 2
        )
    rp, rc = range(len(point)), range(len(cand))
    m = LpProblem()
    x = addbinvars(len(point), len(cand))
    y = addbinvars(len(cand))
    m += lpSum(x[i][j] * point[i][2] * func(i, j) for i in rp for j in rc)
    m += lpSum(y) <= p
    for i in rp:
        m += lpSum(x[i]) == 1
    for j in rc:
        m += lpSum(point[i][2] * x[i][j] for i in rp) <= cand[j][2] * y[j]
    if m.solve() != 1:
        return None
    return [int(value(lpDot(rc, x[i]))) for i in rp]


def facility_location_without_capacity(p, point, cand=None, func=None):
    """
    容量制約なし施設配置問題
        p-メディアン問題：総距離の和の最小化
    入力
        p: 施設数上限
        point: 顧客位置のリスト
        cand: 施設候補位置のリスト(Noneの場合、pointと同じ)
        func: 顧客位置index,施設候補indexを引数とする重み関数
    出力
        顧客ごとの施設番号リスト
    """
    if cand is None:
        cand = point
    if not func:
        func = lambda i, j: sqrt(
            (point[i][0] - cand[j][0]) ** 2 + (point[i][1] - cand[j][1]) ** 2
        )
    rp, rc = range(len(point)), range(len(cand))
    m = LpProblem()
    x = addbinvars(len(point), len(cand))
    y = addbinvars(len(cand))
    m += lpSum(x[i][j] * func(i, j) for i in rp for j in rc)
    m += lpSum(y) <= p
    for i in rp:
        m += lpSum(x[i]) == 1
        for j in rc:
            m += x[i][j] <= y[j]
    if m.solve() != 1:
        return None
    return [int(value(lpDot(rc, x[i]))) for i in rp]


def quad_assign(quant, dist):
    """
    2次割当問題
        全探索
    入力
        quant: 対象間の輸送量
        dist: 割当先間の距離
    出力
        評価値と対象ごとの割当先番号リスト
    """
    from itertools import permutations

    n = len(quant)
    bst, mn, r = None, 1e100, range(n)
    for d in permutations(r):
        s = sum(quant[i][j] * dist[d[i]][d[j]] for i in r for j in r if j != i)
        if s < mn:
            mn = s
            bst = d
    return mn, bst


def gap(cst, req, cap):
    """
    一般化割当問題
        費用最小の割当を解く
    入力
        cst: エージェントごと、ジョブごとの費用のテーブル
        req: エージェントごと、ジョブごとの要求量のテーブル
        cap: エージェントの容量のリスト
    出力
        ジョブごとのエージェント番号リスト
    """
    na, nj = len(cst), len(cst[0])
    m = LpProblem()
    v = [[addvar(cat=LpBinary) for _ in range(nj)] for _ in range(na)]
    m += lpSum(lpDot(cst[i], v[i]) for i in range(na))
    for i in range(na):
        m += lpDot(req[i], v[i]) <= cap[i]
    for j in range(nj):
        m += lpSum(v[i][j] for i in range(na)) == 1
    if m.solve() != 1:
        return None
    return [
        int(value(lpDot(range(na), [v[i][j] for i in range(na)]))) for j in range(nj)
    ]


def stable_matching(prefm, preff):
    """
    安定マッチング問題
    入力
        prefm: 選好(男性の順位別の女性)
        preff: 選好(女性の順位別の男性)
    出力
        マッチング(男性優先,key=女性,value=男性)
    """
    res, n = {}, len(prefm)
    pos, freem = [0] * n, list(range(n - 1, -1, -1))
    while freem:
        m, freem = freem[-1], freem[:-1]
        if pos[m] == n:
            continue
        f, pos[m] = prefm[m][pos[m]], pos[m] + 1
        if f in res:
            if preff[f].index(res[f]) < preff[f].index(m):
                freem.append(m)
                continue
            else:
                freem.append(res[f])
        res[f] = m
    return res


def logistics_network(
    tbde,
    tbdi,
    tbfa,
    dep="需要地",
    dem="需要",
    fac="工場",
    prd="製品",
    tcs="輸送費",
    pcs="生産費",
    lwb="下限",
    upb="上限",
):
    """
    ロジスティクスネットワーク問題を解く
    tbde: 需要地 製品 需要
    tbdi: 需要地 工場 輸送費
    tbfa: 工場 製品 生産費 (下限) (上限)
    出力: 解の有無, 輸送表, 生産表
    """
    facprd = [fac, prd]
    m = LpProblem()
    tbpr = tbfa[facprd].sort_values(facprd).drop_duplicates()
    tbdi2 = pd.merge(tbdi, tbpr, on=fac)
    tbdi2["VarX"] = addvars(tbdi2.shape[0])
    tbfa["VarY"] = addvars(tbfa.shape[0])
    tbsm = pd.concat(
        [tbdi2.groupby(facprd).VarX.sum(), tbfa.groupby(facprd).VarY.sum()], 1
    )
    tbde2 = pd.merge(tbde, tbdi2.groupby([dep, prd]).VarX.sum().reset_index())
    m += lpDot(tbdi2[tcs], tbdi2.VarX) + lpDot(tbfa[pcs], tbfa.VarY)
    tbsm.apply(lambda r: m.addConstraint(r.VarX <= r.VarY), 1)
    tbde2.apply(lambda r: m.addConstraint(r.VarX >= r[dem]), 1)
    if lwb in tbfa:

        def flwb(r):
            r.VarY.lowBound = r[lwb]

        tbfa[tbfa[lwb] > 0].apply(flwb, 1)
    if upb in tbfa:

        def fupb(r):
            r.VarY.upBound = r[upb]

        tbfa[tbfa[upb] != np.inf].apply(fupb, 1)
    m.solve()
    if m.status == 1:
        tbdi2["ValX"] = tbdi2.VarX.apply(value)
        tbfa["ValY"] = tbfa.VarY.apply(value)
    return m.status == 1, tbdi2, tbfa


def sudoku(s, checkOnlyOne=False):
    """
    sudoku(
    '4 . . |. . . |1 . . '
    '. 5 . |. 3 . |. . 8 '
    '2 . . |7 . 8 |. 9 . '
    '------+------+------'
    '. 4 5 |6 . . |8 . 1 '
    '. . 3 |. 5 . |. . . '
    '. 2 . |1 . 3 |. . . '
    '------+------+------'
    '8 . . |. . 5 |. . . '
    '. . 4 |. . . |. . . '
    '. 1 . |. 6 4 |3 . 9 ')[0]
    """
    import re
    from itertools import product

    data = re.sub(r"[^\d.]", "", s)
    assert len(data) == 81
    r = range(9)
    a = pd.DataFrame(
        [
            (i, j, (i // 3) * 3 + j // 3, k + 1, c == str(k + 1))
            for (i, j), c in zip(product(r, r), data)
            for k in r
        ],
        columns=["行", "列", "_3x3", "数", "固"],
    )
    a["Var"] = addbinvars(len(a))
    m = LpProblem()
    for cl in [["行", "列"], ["行", "数"], ["列", "数"], ["_3x3", "数"]]:
        for _, v in a.groupby(cl):
            m += lpSum(v.Var) == 1
    for _, r in a[a.固 == True].iterrows():  # noqa
        m += r.Var == 1
    m.solve()
    if m.status != 1:
        return None, None
    a["Val"] = a.Var.apply(value)
    res = a[a.Val > 0.5].数.values.reshape(9, 9).tolist()
    if checkOnlyOne:
        fr = a[(a.Val > 0.5) & (a.固 != True)].Var  # noqa
        m += lpSum(fr) <= len(fr) - 1
        return res, m.solve() != 1
    return res, None


def groupbys(df1, df2, by=None, left_by=None, right_by=None, allow_right_empty=False):
    """
    df1: Left pandas.DataFrame
    df2: Right pandas.DataFrame
    by: "by" of groupby. Use intersection of each columns, if it is None.
    left_by: This or "by" is used as "by" of df1.groupby.
    right_by: This or "by" is used as "by" of df2.groupby.
    allow_right_empty: Output right, if it is empty.
    """
    if by is None:
        by = df1.columns.intersection(df2.columns).tolist()
    g1 = df1.groupby(left_by if left_by else by)
    g2 = df2.groupby(right_by if right_by else by)
    for k, v1 in g1:
        v2 = df2.iloc[g2.indices.get(k, [])]
        if allow_right_empty or len(v2):
            yield k, v1, v2


class AutoDict:
    """キーをメンバとして使える辞書"""

    def __init__(self, dct):
        for k, v in dct.items():
            self.__dict__[k] = AutoDict(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        return self.__dict__.__getitem__(k)

    def __setitem__(self, k, v):
        self.__dict__.__setitem__(k, v)

    def get(self, k, df=None):
        return self.__dict__.get(k, df)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def clear(self):
        self.__dict__.clear()

    def update(self, d):
        self.__dict__.update(d)

    def copy(self):
        return AutoDict(self.__dict__)

    def pop(self, k, df=None):
        return self.__dict__.pop(k, df)

    def popitem(self):
        return self.__dict__.popitem()

    def setdefault(self, k, df=None):
        return self.__dict__.setdefault(k, df)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, k):
        return self.__dict__.__contains__(k)

    def __delitem__(self, k):
        self.__dict__.__delitem__(k)

    def __repr__(self):
        return repr(self.__dict__)

    @staticmethod
    def fromkeys(iterable, value=None):
        return AutoDict(dict.fromkeys(iterable, value))


class AutoCountDict:
    """新規キーにintを割り当てる辞書"""

    def __init__(self, a=None):
        self._dct = {} if a is None else dict(a)

    def __getitem__(self, k):
        return self.setdefault(k, len(self._dct))

    def __setitem__(self, k, v):
        self._dct.__setitem__(k, v)

    def get(self, k, df=None):
        return self._dct.get(k, df)

    def items(self):
        return self._dct.items()

    def keys(self):
        return self._dct.keys()

    def values(self):
        return self._dct.values()

    def clear(self):
        self._dct.clear()

    def update(self, d):
        self._dct.update(d)

    def copy(self):
        return AutoCountDict(self._dct)

    def pop(self, k, df=None):
        return self._dct.pop(k, df)

    def popitem(self):
        return self._dct.popitem()

    def setdefault(self, k, df=None):
        return self._dct.setdefault(k, df)

    def __iter__(self):
        return iter(self._dct)

    def __len__(self):
        return len(self._dct)

    def __contains__(self, k):
        return self._dct.__contains__(k)

    def __delitem__(self, k):
        self._dct.__delitem__(k)

    def __repr__(self):
        return repr(self._dct)

    @staticmethod
    def fromkeys(iterable, value=None):
        return AutoCountDict(dict.fromkeys(iterable, value))


class CacheDict:
    """Cached Dictionary"""

    from functools import lru_cache

    def __init__(self, a=None):
        self._dct = {} if a is None else dict(a)

    @lru_cache(maxsize=2048)
    def __getitem__(self, k):
        return self._dct.__getitem__(k)

    def __setitem__(self, k, v):
        if self.__contains__(k):
            raise PermissionError()
        self._dct.__setitem__(k, v)

    def get(self, k, df=None):
        return self[k] if self.__contains__(k) else df

    def items(self):
        return self._dct.items()

    def keys(self):
        return self._dct.keys()

    def values(self):
        return self._dct.values()

    def clear(self):
        raise PermissionError()

    def update(self, d):
        for k, v in d.items():
            self[k] = v

    def copy(self):
        return CacheDict(self._dct)

    def pop(self, k, df=None):
        raise PermissionError()

    def popitem(self):
        raise PermissionError()

    def setdefault(self, k, df=None):
        return self._dct.setdefault(k, df)

    def __iter__(self):
        return iter(self._dct)

    def __len__(self):
        return len(self._dct)

    def __contains__(self, k):
        return self._dct.__contains__(k)

    def __delitem__(self, k):
        raise PermissionError()

    def __repr__(self):
        return repr(self._dct)

    @staticmethod
    def fromkeys(iterable, value=None):
        return CacheDict(dict.fromkeys(iterable, value))


class MultiKeyDict:
    """
    子要素が親要素を含む親子キーによる辞書
    from functools import partial
    MKD = partial(MultiKeyDict, iskey=lambda x: x.startswith('key'))
    MKD({'key1':{'k1':1, 'key2':{'k2':2}}})['key1','key2']
    >>>
    {('key1', 'key2', 'k1'): 1, ('key1', 'key2', 'k2'): 2}
    """

    ekey = tuple()  # 空キー

    def __init__(
        self, dc={}, conv=None, iskey=None, dtype=dict, extend=False, cache=2048
    ):
        """
        親子情報ペア(「（キー,値)の配列」と「子キーごとの要素の辞書」のタプル)
        dc: 元辞書
        conv: 要素の変換関数
        iskey: 子キーかどうか
        dtype: 型
        """
        from functools import lru_cache

        self._lst, self._dct = [], {}
        self._conv, self._iskey, self._dtype, self._extend = conv, iskey, dtype, extend
        self.__getitem__ = lru_cache(cache)(self.__getitem__)
        if dtype == dict and isinstance(dc, dict):
            for k, v in dc.items():
                if iskey and iskey(k):
                    self._dct[k] = MultiKeyDict(v, conv, iskey, dtype, extend, cache)
                else:
                    self._lst.append((k, conv(v) if conv else v))
        else:
            if dtype == list and iterable(dc):
                d = (conv(v) if conv else v for v in dc)
                if not extend:
                    d = list(d)
            else:
                d = conv(dc) if conv else dc
            if extend:
                self._lst.extend(d)
            else:
                self._lst.append(d)

    def getitem_iter(self, keys, prkey=None):
        """
        該当木の全直系親、直系子の情報
        keys: キーリスト
        """
        if prkey is None:
            prkey = MultiKeyDict.ekey
        for k, v in self._lst:
            yield prkey + keys + (k,), v
        if not keys:
            for k, v in self._dct.items():
                yield from v.getitem_iter(keys, prkey + (k,))
        elif keys[0] in self._dct:
            yield from self._dct[keys[0]].getitem_iter(keys[1:], prkey + (keys[0],))

    def __getitem__(self, keys):
        """
        該当木の全直系親、直系子の情報
        keys: キーリスト
        """
        return dict(self.getitem_iter(keys if isinstance(keys, tuple) else (keys,)))

    def get_list(self, keys, onlylastkey=False, extend=False):
        d = defaultdict(list)
        for k, v in self.getitem_iter(keys if isinstance(keys, tuple) else (keys,)):
            e = d[k[-1] if onlylastkey else k]
            if extend:
                e.extend(v)
            else:
                e.append(v)
        return dict(d)

    def list(self, key=None):
        return self._lst if key is None else [(k, i) for k, i in self._lst if k == key]

    def dict(self, key=None):
        return (
            self._dct
            if key is None
            else {k: v for k, v in self._dct.items() if k == key}
        )

    def items(self, func=None):
        d = self[MultiKeyDict.ekey].items()
        return d if func is None else (i for i in d if func(i))

    def __iter__(self):
        return iter(self[MultiKeyDict.ekey])

    def __repr__(self):
        """表現"""
        return repr((self._lst, self._dct))


class unionfind:
    """指定された二つの要素key1, key2が同じグループに含まれるかどうかを調べる"""

    def __init__(self, *args):
        self._dict = {}  # key -> index
        self._parent = []  # index -> gid（同じ値はグループの代表）

    def find(self, key, index=None):
        """グループ代表"""
        if index is not None:
            par = self._parent[index]
        elif key not in self._dict:
            self._dict[key] = index = par = len(self._parent)
            self._parent.append(index)
        else:
            index = self._dict[key]
            par = self._parent[index]
        if par != index:
            par = self._parent[index] = self.find(None, par)
        return par

    def unite(self, keyi, keyj):
        """同じグループに"""
        gidi = self.find(keyi)
        gidj = self.find(keyj)
        if gidi != gidj:
            self._parent[gidi] = gidj

    def issame(self, keyi, keyj):
        """同じグループか"""
        return self.find(keyi) == self.find(keyj)

    def groups(self):
        """全グループ"""
        d = defaultdict(list)
        for key in self._dict:
            d[self.find(key)].append(key)
        return list(d.values())

    @staticmethod
    def isconnected(ll, u=None):
        nw, nh = len(ll), len(ll[0])
        rw, rh = range(nw), range(nh)
        if not u:
            u = unionfind()
        f = -1
        for i in rw:
            for j in rh:
                if not ll[i][j]:
                    continue
                if f < 0:
                    f = i + j * nw
                if j > 0 and ll[i][j] == ll[i][j - 1]:
                    u.unite(i + j * nw, i + j * nw - nw)
                if i > 0 and ll[i][j] == ll[i - 1][j]:
                    u.unite(i + j * nw, i + j * nw - 1)
        return f >= 0 and all(
            [u.issame(f, i + j * nw) for i in rw for j in rh if ll[i][j]]
        )

    @staticmethod
    def isconnectedlist(nw, nh, lst):
        ll = [[False] * nw for j in range(nh)]
        for i, j in lst:
            ll[i][j] = True
        return unionfind.isconnected(ll)
