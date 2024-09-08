import pickle
from math import sqrt
from pathlib import Path

import numpy as np

_path = Path(__file__).parent / "cover_by_rect.pkl"


def get_cache(maxsize=100):
    if not _path.exists():
        cache = {}
        for i in range(1, int(sqrt(maxsize)) + 1):
            for j in range(1, maxsize // i + 1):
                cache[i, j] = 1 + (i + j) / i / j * 1e-3, np.array([[0, 0, i, j]])
                cache[j, i] = 1 + (j + i) / j / i * 1e-3, np.array([[0, 0, j, i]])
        save_cache(cache)
        return cache
    with open(_path, "rb") as fp:
        return pickle.load(fp)


def save_cache(cache):
    with open(_path, "wb") as fp:
        pickle.dump(cache, fp)


class CoverByRect:
    cache = get_cache()

    @staticmethod
    def get(x, y):
        """
        長方形を上限ありの四角形でカバーする
        入力
            x, y: サイズ
        出力
            カバーする[l, t, r, b]のリスト
        """
        res = CoverByRect.cache.get((x, y))
        if res is not None:
            return res[1]
        if (x != y and x != y - 1) or x < 256 or y < 256:
            res = CoverByRect._calc(x, y)
            save_cache(CoverByRect.cache)
            return res[1]
        x1 = x // 2
        x2 = x - x1
        y1 = y // 2
        y2 = y - y1
        r11 = CoverByRect.get(x1, y1)
        r12 = [a + [0, y1, 0, y1] for a in CoverByRect.get(x1, y2)]
        r21 = [a[[1, 0, 3, 2]] + [x1, 0, x1, 0] for a in CoverByRect.get(y1, x2)]
        r22 = [a + [x1, y1, x1, y1] for a in CoverByRect.get(x2, y2)]
        res = np.r_[r11, r12, r21, r22]
        CoverByRect.cache[x, y] = len(res), res
        save_cache(CoverByRect.cache)
        return res

    @staticmethod
    def _calc(x, y):
        if x * y == 0:
            return 0, []
        r = CoverByRect.cache.get((x, y))
        if r:
            return r
        if x < y:
            n, r = CoverByRect._calc(y, x)
            bst = n, np.array([i[[1, 0, 3, 2]] for i in r])
        else:
            bst = x * y, None
            for i in range(x // 2, 0, -1):
                n1, r1 = CoverByRect._calc(i, y)
                n2, r2 = CoverByRect._calc(x - i, y)
                if n1 + n2 < bst[0]:
                    bst = n1 + n2, np.r_[r1, r2 + [i, 0, i, 0]]
            for i in range(y // 2, 0, -1):
                n1, r1 = CoverByRect._calc(x, i)
                n2, r2 = CoverByRect._calc(x, y - i)
                if n1 + n2 < bst[0]:
                    bst = n1 + n2, np.r_[r1, r2 + [0, i, 0, i]]
            if x == y:
                for i in range(2 - x % 2, x, 2):
                    j = (x - i) // 2
                    n1, r1 = CoverByRect._calc(i, i)
                    n2, r2 = CoverByRect._calc(j, x - j)
                    if n1 + n2 * 4 < bst[0]:
                        bst = (
                            n1 + n2 * 4,
                            np.r_[
                                r2,
                                r2 + [x - j, j, x - j, j],
                                r1 + [j, j, j, j],
                                [k[[1, 0, 3, 2]] + [j, 0, j, 0] for k in r2],
                                [k[[1, 0, 3, 2]] + [0, x - j, 0, x - j] for k in r2],
                            ],
                        )
        CoverByRect.cache[x, y] = bst
        return bst
