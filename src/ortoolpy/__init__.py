"""
# オペレーションズ・リサーチとは

- オペレーションズ・リサーチ(Operations Research: OR)とは数理的アプローチに基づく問題解決学です。
  - 数学を使って社会の課題を解決します。
  - 数理最適化、グラフ理論、待ち行列、確率、統計、シミュレーション、PERT、AHP、データマイニング、ゲーム理論、ロジスティクス、サプライチェーン・マネジメントなどがあります。
  - 数理最適化は、ORの中でも研究者がたくさんいます。最近では、ソフトウェア性能向上とハードの性能向上の相乗効果により、これまで解けなかったような問題も解けるようになってきています。

 - 公益社団法人日本オペレーションズ・リサーチ学会: http://www.orsj.org/
 - オペレーションズ・リサーチとは: https://orsj.org/?page_id=420
"""

from importlib.metadata import metadata

from .etc import *  # noqa: F401 F403 RUF100

# from .optimization import * # networkx等が必要なのでデフォルトではimportしない  # noqa: ERA001

_package_metadata = metadata(__package__)
__version__ = _package_metadata["Version"]
__author__ = _package_metadata.get("Author-email", "")
