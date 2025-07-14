from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        """
        self.n = n
        self.adj: DefaultDict[int, List[int]] = defaultdict(list)

    def add_edge(self, u: int, v: int) -> None:
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)
        self.adj[u].sort()
        self.adj[v].sort()

    def dfs(self, start: int) -> list[int]:
        visited = [False] * (self.n + 1)
        result = []
        stack = [start]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                result.append(node)
                # 인접 노드를 내림차순으로 넣어야 스택에서 오름차순 방문
                for neighbor in sorted(self.adj[node], reverse=True):
                    if not visited[neighbor]:
                        stack.append(neighbor)
        return result

    def bfs(self, start: int) -> list[int]:
        visited = [False] * (self.n + 1)
        result = []
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in sorted(self.adj[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        return result
    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))
