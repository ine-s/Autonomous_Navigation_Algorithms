import heapq
import math
import os
import time

import networkx as nx

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_shortest_path(graph, path, title="Shortest path", save_path=None):
    """Visualise un graphe NetworkX et le chemin trouvé."""
    if not HAS_MATPLOTLIB:
        print("[INFO] matplotlib non installé -> pas de visualisation.")
        return

    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = nx.random_layout(graph, seed=42)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=25, node_color="lightgray", alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edge_color="lightgray", alpha=0.4, width=0.7)

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="red", width=2.0)

        start_node, end_node = path[0], path[-1]
        mid_nodes = path[1:-1]

        if mid_nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=mid_nodes, node_size=55, node_color="red")
        nx.draw_networkx_nodes(graph, pos, nodelist=[start_node], node_size=100, node_color="green")
        nx.draw_networkx_nodes(graph, pos, nodelist=[end_node], node_size=100, node_color="blue")

    plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight")
        print(f"[IMAGE] {save_path}")
    else:
        plt.show()
    plt.close()


def ensure_weighted_graph(graph):
    for u, v in graph.edges():
        graph[u][v]["weight"] = graph[u][v].get("weight", 1.0)


def largest_component_subgraph(graph):
    if nx.is_connected(graph):
        return graph
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def pick_start_goal(graph):
    nodes = list(graph.nodes())
    start = nodes[0]
    goal = nodes[-1]
    return start, goal


def reconstruct_path(prev, end):
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def dijkstra_with_stats(graph, start, goal):
    """Dijkstra classique (unidirectionnel)."""
    t0 = time.perf_counter()

    dist = {start: 0.0}
    prev = {start: None}
    heap = [(0.0, start)]
    visited = set()
    explored = 0

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        explored += 1

        if u == goal:
            break

        for v in graph.neighbors(u):
            w = graph[u][v].get("weight", 1.0)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    duration = time.perf_counter() - t0
    if goal not in prev and goal != start:
        return None, explored, duration, float("inf")

    path = reconstruct_path(prev, goal)
    return path, explored, duration, dist.get(goal, 0.0)


def bidirectional_dijkstra_with_stats(graph, start, goal):
    """Bidirectional Dijkstra (maison), version optimale (coût identique à Dijkstra)."""
    t0 = time.perf_counter()

    f_dist = {start: 0.0}
    b_dist = {goal: 0.0}
    f_prev = {start: None}
    b_prev = {goal: None}

    f_heap = [(0.0, start)]
    b_heap = [(0.0, goal)]

    f_seen = set()  # nœuds définitivement traités côté start
    b_seen = set()  # nœuds définitivement traités côté goal
    explored = 0
    meet = None
    mu = float("inf")

    def pop_valid(heap, dist, seen):
        while heap:
            d, u = heapq.heappop(heap)
            if u in seen:
                continue
            if d > dist.get(u, float("inf")):
                continue
            return d, u
        return None, None

    def peek_valid_dist(heap, dist, seen):
        while heap:
            d, u = heap[0]
            if u in seen or d > dist.get(u, float("inf")):
                heapq.heappop(heap)
                continue
            return d
        return float("inf")

    while f_heap and b_heap:
        df, uf = pop_valid(f_heap, f_dist, f_seen)
        if uf is None:
            break
        f_seen.add(uf)
        explored += 1
        if uf in b_seen:
            cand = f_dist[uf] + b_dist[uf]
            if cand < mu:
                mu = cand
                meet = uf
        for vf in graph.neighbors(uf):
            w = graph[uf][vf].get("weight", 1.0)
            nd = df + w
            if nd < f_dist.get(vf, float("inf")):
                f_dist[vf] = nd
                f_prev[vf] = uf
                heapq.heappush(f_heap, (nd, vf))
            if vf in b_seen:
                cand = f_dist.get(vf, nd) + b_dist[vf]
                if cand < mu:
                    mu = cand
                    meet = vf

        db, ub = pop_valid(b_heap, b_dist, b_seen)
        if ub is None:
            break
        b_seen.add(ub)
        explored += 1
        if ub in f_seen:
            cand = f_dist[ub] + b_dist[ub]
            if cand < mu:
                mu = cand
                meet = ub
        for vb in graph.neighbors(ub):
            w = graph[ub][vb].get("weight", 1.0)
            nd = db + w
            if nd < b_dist.get(vb, float("inf")):
                b_dist[vb] = nd
                b_prev[vb] = ub
                heapq.heappush(b_heap, (nd, vb))
            if vb in f_seen:
                cand = b_dist.get(vb, nd) + f_dist[vb]
                if cand < mu:
                    mu = cand
                    meet = vb

        # Condition d'arrêt correcte pour bidirectional Dijkstra
        if meet is not None:
            if peek_valid_dist(f_heap, f_dist, f_seen) + peek_valid_dist(b_heap, b_dist, b_seen) >= mu:
                break

    duration = time.perf_counter() - t0
    if meet is None:
        return None, explored, duration, float("inf")

    # start -> meet
    left = []
    cur = meet
    while cur is not None:
        left.append(cur)
        cur = f_prev.get(cur)
    left.reverse()

    # meet -> goal
    right = []
    cur = b_prev.get(meet)
    while cur is not None:
        right.append(cur)
        cur = b_prev.get(cur)

    path = left + right
    cost = mu
    return path, explored, duration, cost


def bidirectional_astar_with_stats(graph, start, goal):
    """A* bidirectionnel (heuristique euclidienne si positions, sinon spring layout)."""
    t0 = time.perf_counter()

    pos = nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = nx.random_layout(graph, seed=42)

    def h(a, b):
        ax, ay = pos[a]
        bx, by = pos[b]
        return math.hypot(ax - bx, ay - by)

    gf = {start: 0.0}
    gb = {goal: 0.0}
    pf = {start: None}
    pb = {goal: None}

    open_f = [(h(start, goal), start)]
    open_b = [(h(goal, start), goal)]

    seen_f = set()
    seen_b = set()
    explored = 0
    meet = None

    while open_f and open_b:
        _, uf = heapq.heappop(open_f)
        if uf not in seen_f:
            seen_f.add(uf)
            explored += 1
            if uf in seen_b:
                meet = uf
                break
            for vf in graph.neighbors(uf):
                w = graph[uf][vf].get("weight", 1.0)
                nd = gf[uf] + w
                if nd < gf.get(vf, float("inf")):
                    gf[vf] = nd
                    pf[vf] = uf
                    heapq.heappush(open_f, (nd + h(vf, goal), vf))

        _, ub = heapq.heappop(open_b)
        if ub not in seen_b:
            seen_b.add(ub)
            explored += 1
            if ub in seen_f:
                meet = ub
                break
            for vb in graph.neighbors(ub):
                w = graph[ub][vb].get("weight", 1.0)
                nd = gb[ub] + w
                if nd < gb.get(vb, float("inf")):
                    gb[vb] = nd
                    pb[vb] = ub
                    heapq.heappush(open_b, (nd + h(vb, start), vb))

    duration = time.perf_counter() - t0
    if meet is None:
        return None, explored, duration, float("inf")

    left = []
    cur = meet
    while cur is not None:
        left.append(cur)
        cur = pf.get(cur)
    left.reverse()

    right = []
    cur = pb.get(meet)
    while cur is not None:
        right.append(cur)
        cur = pb.get(cur)

    path = left + right
    cost = gf[meet] + gb[meet]
    return path, explored, duration, cost


def nx_bidirectional_dijkstra(graph, start, goal):
    """Référence NetworkX."""
    t0 = time.perf_counter()
    dist, path = nx.bidirectional_dijkstra(graph, start, goal, weight="weight")
    duration = time.perf_counter() - t0
    return path, dist, duration


def format_result(name, path, explored, duration, cost):
    length = len(path) - 1 if path else None
    if path is None:
        return f"{name:<34} | path: NONE | explored: {explored:<6} | time: {duration:.4f}s | cost: inf"
    return f"{name:<34} | path_len: {length:<4} | explored: {explored:<6} | time: {duration:.4f}s | cost: {cost:.3f}"


def log_info(message):
    print(f"[INFO] {message}")


def log_ok(message):
    print(f"[OK]   {message}")


def log_warn(message):
    print(f"[WARN] {message}")


def compare_on_graph(graph, graph_name, image_tag):
    ensure_weighted_graph(graph)
    graph = largest_component_subgraph(graph)
    start, goal = pick_start_goal(graph)

    print()
    log_info(
        f"Graph={graph_name} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} start={start} goal={goal}"
    )

    path_d, explored_d, t_d, cost_d = dijkstra_with_stats(graph, start, goal)
    path_bd, explored_bd, t_bd, cost_bd = bidirectional_dijkstra_with_stats(graph, start, goal)
    path_ba, explored_ba, t_ba, cost_ba = bidirectional_astar_with_stats(graph, start, goal)

    try:
        path_nx, dist_nx, t_nx = nx_bidirectional_dijkstra(graph, start, goal)
    except nx.NetworkXNoPath:
        path_nx, dist_nx, t_nx = None, float("inf"), 0.0

    print("  " + format_result("Dijkstra classique", path_d, explored_d, t_d, cost_d))
    print("  " + format_result("Dijkstra bidirectionnel (maison)", path_bd, explored_bd, t_bd, cost_bd))
    print("  " + format_result("A* bidirectionnel (maison)", path_ba, explored_ba, t_ba, cost_ba))
    nx_len = len(path_nx) - 1 if path_nx else None
    print(
        "  "
        + f"{'NetworkX bidirectional_dijkstra':<34} | path_len: {nx_len if nx_len is not None else 'NONE':<4} | time: {t_nx:.4f}s | cost: {dist_nx if dist_nx != float('inf') else 'inf'}"
    )

    # Vérifications de l'énoncé
    q1 = path_bd is not None
    q3 = path_bd is not None and path_nx is not None and abs(cost_bd - dist_nx) < 1e-4
    q4 = path_d is not None and path_bd is not None and explored_bd <= explored_d
    q5 = path_ba is not None
    q6_comparable = path_bd is not None and path_ba is not None

    log_ok(f"Q1 Bi-Dijkstra implémenté: {'PASS' if q1 else 'FAIL'}")
    if q3:
        log_ok("Q3 coût Bi-Dijkstra == NetworkX: PASS")
    else:
        log_warn("Q3 coût Bi-Dijkstra != NetworkX: FAIL")
    log_ok(f"Q4 Bi-Dijkstra explore <= Dijkstra: {'PASS' if q4 else 'FAIL'}")
    log_ok(f"Q5 Bi-A* implémenté: {'PASS' if q5 else 'FAIL'}")
    if q6_comparable:
        if explored_ba < explored_bd:
            q6_msg = "Bi-A* explore moins de nœuds"
        elif explored_ba == explored_bd:
            q6_msg = "Bi-A* et Bi-Dijkstra explorent pareil"
        else:
            q6_msg = "Bi-A* explore plus de nœuds"
    else:
        q6_msg = "Comparaison impossible (pas de chemin)"
    log_info(f"Q6 comparaison Bi-Dijkstra/Bi-A*: {q6_msg}")

    # Images des solutions
    if path_bd:
        visualize_shortest_path(
            graph,
            path_bd,
            title=f"{graph_name} - Dijkstra bidirectionnel",
            save_path=os.path.join(OUTPUT_DIR, f"tp2_{image_tag}_bidijkstra.png"),
        )
    if path_ba:
        visualize_shortest_path(
            graph,
            path_ba,
            title=f"{graph_name} - A* bidirectionnel",
            save_path=os.path.join(OUTPUT_DIR, f"tp2_{image_tag}_biastar.png"),
        )


def main():
    # Q2 : tests sur graphes de plus grande taille (énoncé)
    g_geo = nx.random_geometric_graph(2000, 0.05, seed=42)

    # pondération euclidienne pour geometric graph
    pos_geo = nx.get_node_attributes(g_geo, "pos")
    for u, v in g_geo.edges():
        x1, y1 = pos_geo[u]
        x2, y2 = pos_geo[v]
        g_geo[u][v]["weight"] = math.hypot(x1 - x2, y1 - y2)

    g_er = nx.erdos_renyi_graph(1000, 0.01, seed=42)
    g_ba = nx.barabasi_albert_graph(1000, 5, seed=42)

    compare_on_graph(g_geo, "Random geometric graph", "geo")
    compare_on_graph(g_er, "Erdos-Renyi graph", "er")
    compare_on_graph(g_ba, "Barabasi-Albert graph", "ba")

    print()
    log_ok("Run terminé")
    log_info("Images: output_images/tp2_*_bidijkstra.png et output_images/tp2_*_biastar.png")


if __name__ == "__main__":
    main()
