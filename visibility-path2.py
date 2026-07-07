import sys
import time
import math
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
import triangle as tr
import osmnx as ox

# --- CONFIGURATIONS ---
sys.setrecursionlimit(20000)
ox.settings.use_cache = True
ox.settings.timeout = 300

try:
    from shapely.strtree import STRTree
except ImportError:
    from shapely.strtree import STRtree as STRTree

# ==============================================================================
# 1. GEOMETRY KERNEL (Unchanged)
# ==============================================================================
epsilon = 1.1102230246251565e-16
splitter = 134217729.0

def two_sum(a, b):
    x = a + b
    b_virtual = x - a
    a_virtual = x - b_virtual
    b_roundoff = b - b_virtual
    a_roundoff = a - a_virtual
    return x, a_roundoff + b_roundoff

def split(a):
    c = splitter * a
    a_big = c - a
    a_hi = c - a_big
    a_lo = a - a_hi
    return a_hi, a_lo

def two_product(a, b):
    x = a * b
    a_hi, a_lo = split(a)
    b_hi, b_lo = split(b)
    err1 = x - (a_hi * b_hi)
    err2 = err1 - (a_lo * b_hi)
    err3 = err2 - (a_hi * b_lo)
    y = a_lo * b_lo - err3
    return x, y

def orient2d_adapt(pa, pb, pc):
    acx = pa[0] - pc[0]; bcx = pb[0] - pc[0]
    acy = pa[1] - pc[1]; bcy = pb[1] - pc[1]
    detleft, detleft_err = two_product(acx, bcy)
    detright, detright_err = two_product(acy, bcx)
    det, det_err = two_sum(detleft, -detright)
    b_virtual = det - detleft
    a_virtual = det - b_virtual
    b_roundoff = -detright - b_virtual
    a_roundoff = detleft - a_virtual
    det_err += a_roundoff + b_roundoff
    return det_err

def orient2d(pa, pb, pc):
    det = (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pa[1] - pc[1]) * (pb[0] - pc[0])
    if det != 0.0:
        det_bound = (abs(pa[0] - pc[0]) + abs(pb[0] - pc[0])) * (abs(pa[1] - pc[1]) + abs(pb[1] - pc[1]))
        if abs(det) >= epsilon * det_bound: return det
    return orient2d_adapt(pa, pb, pc)

def build_halfedges(triangles_flat):
    num_triangles = len(triangles_flat) // 3
    halfedges = -np.ones(num_triangles * 3, dtype=int)
    edge_map = {}
    for t_idx in range(num_triangles):
        for i in range(3):
            idx1 = triangles_flat[t_idx * 3 + i]
            idx2 = triangles_flat[t_idx * 3 + (i + 1) % 3]
            edge = tuple(sorted((idx1, idx2)))
            if edge not in edge_map: edge_map[edge] = []
            edge_map[edge].append(t_idx * 3 + i)
    for hes in edge_map.values():
        if len(hes) == 2:
            he1, he2 = hes
            halfedges[he1], halfedges[he2] = he2, he1
    return halfedges

def next_edge(e): return e - 2 if e % 3 == 2 else e + 1
def prev_edge(e): return e + 2 if e % 3 == 0 else e - 1
def is_left_of(x1, y1, x2, y2, px, py): return orient2d((x1, y1), (x2, y2), (px, py)) > 0
def is_right_of(x1, y1, x2, y2, px, py): return orient2d((x1, y1), (x2, y2), (px, py)) < 0

def order_angles(qx, qy, p1x, p1y, p2x, p2y):
    seg_left = is_left_of(qx, qy, p2x, p2y, p1x, p1y)
    lx, ly = (p1x, p1y) if seg_left else (p2x, p2y)
    rx, ry = (p2x, p2y) if seg_left else (p1x, p1y)
    return [lx, ly, rx, ry]

def order_del_angles(d, qx, qy, p1_idx, p2_idx):
    coords = d['coords']
    p1x, p1y = coords[p1_idx]; p2x, p2y = coords[p2_idx]
    return order_angles(qx, qy, p1x, p1y, p2x, p2y)

def is_within_cone(px, py, slx, sly, srx, sry, rlx, rly, rrx, rry):
    if is_left_of(px, py, slx, sly, rrx, rry): return False
    if is_left_of(px, py, rlx, rly, srx, sry): return False
    return True

def restrict_angles(px, py, slx, sly, srx, sry, rlx, rly, rrx, rry):
    nlx, nly, res_left = (slx, sly, True) if is_right_of(px, py, rlx, rly, slx, sly) else (rlx, rly, False)
    nrx, nry, res_right = (srx, sry, True) if is_left_of(px, py, rrx, rry, srx, sry) else (rrx, rry, False)
    return ([nlx, nly, nrx, nry], res_left, res_right)

def seg_intersect_ray(s1x, s1y, s2x, s2y, r1x, r1y, r2x, r2y):
    rdx, rdy = r2x - r1x, r2y - r1y
    sdx, sdy = s2x - s1x, s2y - s1y
    denominator = sdx * rdy - sdy * rdx
    if denominator == 0: return float('inf')
    t2 = (rdx * (s1y - r1y) + rdy * (r1x - s1x)) / denominator
    if rdx != 0: t1 = (s1x + sdx * t2 - r1x) / rdx
    else: t1 = (s1y + sdy * t2 - r1y) / rdy if rdy != 0 else float('inf')
    if t1 < -1e-9 or t2 < -1e-9 or t2 > 1.0 + 1e-9: return float('inf')
    return t1

def containing_triangle(d, qx, qy):
    coords = d['coords']
    triangles = d['triangles']
    q = (qx, qy)
    # Query the STRtree with a point to get candidate triangles whose bounding
    # boxes contain (qx, qy), then do exact orient2d checks only on those.
    pt = Point(qx, qy)
    for t_idx in d['tri_strtree'].query(pt):
        p_indices = triangles[t_idx*3 : t_idx*3+3]
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    return -1

def triangular_expansion(d, qx, qy, obstructs):
    memo = {}
    triangles = d['triangles']
    coords = d['coords']
    halfedges = d['halfedges']
    
    def expand(edg_in, rlx, rly, rrx, rry):
        key = (edg_in, rlx, rly, rrx, rry)
        if key in memo: return memo[key]
        ret = []
        edges = [next_edge(edg_in), prev_edge(edg_in)]
        for edg in edges:
            p1_idx, p2_idx = triangles[edg], triangles[next_edge(edg)]
            adj_out = halfedges[edg]
            slx, sly, srx, sry = order_del_angles(d, qx, qy, p1_idx, p2_idx)
            if not is_within_cone(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry): continue
            [nlx, nly, nrx, nry], res_l, res_r = restrict_angles(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry)
            if orient2d((qx, qy), (nrx, nry), (nlx, nly)) <= 0.0: continue
            if adj_out != -1 and not obstructs(edg):
                ret.extend(expand(adj_out, nlx, nly, nrx, nry))
                continue
            if not res_l:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rlx, rly)
                if inter != float('inf'): slx, sly = qx + inter * (rlx-qx), qy + inter * (rly-qy)
            if not res_r:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rrx, rry)
                if inter != float('inf'): srx, sry = qx + inter * (rrx-qx), qy + inter * (rry-qy)
            ret.append((slx, sly, srx, sry))
        memo[key] = ret
        return ret

    tri_start = containing_triangle(d, qx, qy)
    if tri_start == -1: return []
    ret = []
    p_indices = triangles[tri_start*3 : tri_start*3+3]
    points = coords[p_indices]
    points_sorted = sorted(points.tolist(), key=lambda p: np.arctan2(p[1] - qy, p[0] - qx))
    for i in range(3):
        p_start, p_end = points_sorted[i], points_sorted[(i + 1) % 3]
        rlx, rly, rrx, rry = order_angles(qx, qy, p_start[0], p_start[1], p_end[0], p_end[1])
        for edg in [tri_start * 3, tri_start * 3 + 1, tri_start * 3 + 2]:
            p1_idx, p2_idx = triangles[edg], triangles[next_edge(edg)]
            p1c, p2c = coords[p1_idx], coords[p2_idx]
            if (np.allclose(p1c, p_start) and np.allclose(p2c, p_end)) or (np.allclose(p1c, p_end) and np.allclose(p2c, p_start)):
                adj = halfedges[edg]
                if adj == -1 or obstructs(edg):
                    ret.append(order_angles(qx, qy, p1c[0], p1c[1], p2c[0], p2c[1]))
                else:
                    ret.extend(expand(adj, rlx, rly, rrx, rry))
                break
    return ret

def get_visibility_polygon(viewpoint, d, obstructs_func):
    segs = triangular_expansion(d, viewpoint[0], viewpoint[1], obstructs_func)
    if not segs: return Polygon()
    qx, qy = viewpoint
    # Collect all boundary endpoints and sort by angle around viewpoint.
    # The visibility polygon is star-shaped so this directly gives the boundary
    # without needing to build and merge individual triangles via unary_union.
    pts = {}
    for s in segs:
        for px, py in [(s[0], s[1]), (s[2], s[3])]:
            angle = math.atan2(py - qy, px - qx)
            pts[angle] = (px, py)
    if len(pts) < 2: return Polygon()
    ordered = [pts[a] for a in sorted(pts)]
    return Polygon(ordered)

# ==============================================================================
# 2. ENVIRONMENT
# ==============================================================================

class Environment2D:
    def __init__(self, gdf, bounds=1600):
        self.bounds = bounds
        self.boundary_poly = box(-bounds, -bounds, bounds, bounds)
        self.boundary_geom = self.boundary_poly.boundary
        minx, miny, maxx, maxy = gdf.total_bounds
        cx, cy = (minx+maxx)/2, (miny+maxy)/2
        raw = []
        for poly in gdf.geometry:
            if poly.geom_type == 'MultiPolygon': poly = max(poly.geoms, key=lambda a: a.area)
            if poly.geom_type != 'Polygon': continue
            trans_poly = Polygon([(p[0]-cx, p[1]-cy) for p in poly.exterior.coords])
            if trans_poly.area > 5.0 and trans_poly.intersects(self.boundary_poly):
                raw.append(trans_poly)
        merged = unary_union(raw)
        self.obstacles = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
        if merged.geom_type == 'Polygon': self.obstacles = [merged]
        self.tree = STRTree(self.obstacles) if self.obstacles else STRTree([Polygon()])
        self.d = self._build_mesh()
        
    def _build_mesh(self):
        vertices, segments, holes = [], [], []
        def add_ring(coords, is_hole=False):
            if coords[0] == coords[-1]: coords = coords[:-1]
            start = len(vertices)
            for x, y in coords: vertices.append([x, y])
            for i in range(len(coords)): segments.append([start + i, start + (i+1)%len(coords)])
            if is_hole:
                try: holes.append([Polygon(coords).representative_point().x, Polygon(coords).representative_point().y])
                except: pass
        add_ring(list(self.boundary_poly.exterior.coords))
        for obs in self.obstacles: add_ring(list(obs.exterior.coords), is_hole=True)
        B = tr.triangulate({'vertices': vertices, 'segments': segments, 'holes': holes}, 'p')
        triangles_flat = B['triangles'].flatten()
        coords = B['vertices']
        halfedges = build_halfedges(triangles_flat)
        triangles_2d = B['triangles']
        wall_edges = set()
        if 'segments' in B:
             seg_set = set(tuple(sorted((s[0], s[1]))) for s in B['segments'])
             for t_idx, tri in enumerate(triangles_2d):
                 for i in range(3):
                     u, v = tri[i], tri[(i+1)%3]
                     if tuple(sorted((u, v))) in seg_set: wall_edges.add(t_idx * 3 + i)
        self.wall_edges = wall_edges

        # Build a spatial index over triangle bounding boxes for fast point location
        num_triangles = len(triangles_flat) // 3
        tri_boxes = []
        for t_idx in range(num_triangles):
            p1, p2, p3 = coords[triangles_flat[t_idx*3 : t_idx*3+3]]
            minx = min(p1[0], p2[0], p3[0]); maxx = max(p1[0], p2[0], p3[0])
            miny = min(p1[1], p2[1], p3[1]); maxy = max(p1[1], p2[1], p3[1])
            tri_boxes.append(box(minx, miny, maxx, maxy))
        tri_strtree = STRTree(tri_boxes)

        return {'triangles': triangles_flat, 'coords': coords, 'halfedges': halfedges,
                'tri_strtree': tri_strtree, 'tri_boxes': tri_boxes}

    def obstructs(self, edge_idx): return edge_idx in self.wall_edges

    def get_windows(self, vis_poly):
        boundary = vis_poly.boundary
        if boundary.is_empty: return []
        parts = list(boundary.geoms) if boundary.geom_type == 'MultiLineString' else [boundary]
        windows = []
        for part in parts:
            coords = list(part.coords)
            for i in range(len(coords)-1):
                p1, p2 = coords[i], coords[i+1]
                if math.dist(p1, p2) < 1.0: continue
                mx, my = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
                mid_pt = Point(mx, my)
                if self.boundary_geom.distance(mid_pt) < 0.1: continue
                candidates_idx = self.tree.query(box(mx - 0.1, my - 0.1, mx + 0.1, my + 0.1))
                is_wall = any(self.obstacles[idx].distance(mid_pt) < 0.1 for idx in candidates_idx)
                if not is_wall: windows.append(LineString([p1, p2]))
        return windows

# ==============================================================================
# 3. BIDIRECTIONAL SOLVER
# ==============================================================================

class MinLinkSolver:
    def __init__(self, env):
        self.env = env
        self.history = []

    def solve(self, start_pt, end_pt):
        start, end = tuple(start_pt), tuple(end_pt)
        if containing_triangle(self.env.d, start[0], start[1]) == -1: return None

        # Two search trees: pos -> {'parent': pos, 'poly': Polygon, 'id': int}
        tree_A = {start: {'parent': None, 'poly': None, 'id': 0, 'side': 'A'}}
        tree_B = {end: {'parent': None, 'poly': None, 'id': 1, 'side': 'B'}}

        frontier_A = [start]
        frontier_B = [end]

        # Flat lists of (pos, poly) for each side, mirroring insertion order
        # so that STRtree query indices map back to these lists.
        spatial_A = []  # [(pos, poly), ...]
        spatial_B = []
        strtree_A = None  # built after A expands, used by B in the same round
        strtree_B = None  # built after B expands, used by A in the next round

        # Pruning: skip windows already seen or already inside explored territory
        seen_windows_A = set()  # rounded midpoint keys of all windows ever found on side A
        seen_windows_B = set()

        node_counter = 2
        self.history = []

        # --- Timers ---
        timers = {
            'visibility':    0.0,  # get_visibility_polygon
            'intersection':  0.0,  # strtree query + intersects check
            'windows':       0.0,  # get_windows
            'strtree_build': 0.0,  # rebuilding spatial indices
            'pruning':       0.0,  # seen_windows + inside-visibility checks
        }
        solve_start = time.perf_counter()

        for current_g in range(15): # Max Link Depth
            # --- Expand SIDE A ---
            new_frontier_A = []
            for curr in frontier_A:
                t0 = time.perf_counter()
                vis_poly = get_visibility_polygon(curr, self.env.d, self.env.obstructs)
                timers['visibility'] += time.perf_counter() - t0
                tree_A[curr]['poly'] = vis_poly
                spatial_A.append((curr, vis_poly))

                # Spatial candidate filter, then precise intersection check
                # Done before get_windows so we skip window computation on termination
                t0 = time.perf_counter()
                if strtree_B is not None:
                    for idx in strtree_B.query(vis_poly):
                        pos_B, poly_B = spatial_B[idx]
                        if vis_poly.intersects(poly_B):
                            timers['intersection'] += time.perf_counter() - t0
                            self._print_timers(timers, time.perf_counter() - solve_start)
                            inter = vis_poly.intersection(poly_B)
                            bridge_pt = (inter.centroid.x, inter.centroid.y)
                            return self._reconstruct(tree_A, tree_B, curr, pos_B, bridge_pt)
                timers['intersection'] += time.perf_counter() - t0

                # Only compute windows if we didn't terminate above
                t0 = time.perf_counter()
                windows = self.env.get_windows(vis_poly)
                timers['windows'] += time.perf_counter() - t0

                step_data = {'curr_pos': curr, 'vis_poly': vis_poly, 'side': 'A', 'children': []}
                for win in windows:
                    mx, my = win.centroid.x, win.centroid.y
                    win_key = (round(mx, 4), round(my, 4))

                    t0 = time.perf_counter()
                    if win_key in seen_windows_A:
                        timers['pruning'] += time.perf_counter() - t0
                        continue
                    seen_windows_A.add(win_key)
                    timers['pruning'] += time.perf_counter() - t0

                    mid_pt = (mx, my)
                    if mid_pt not in tree_A:
                        tree_A[mid_pt] = {'parent': curr, 'poly': None, 'id': node_counter, 'side': 'A'}
                        new_frontier_A.append(mid_pt)
                        step_data['children'].append({'pos': mid_pt, 'id': node_counter})
                        node_counter += 1
                self.history.append(step_data)

            frontier_A = new_frontier_A

            # Rebuild strtree_A so B can query A's current-round polygons too
            t0 = time.perf_counter()
            if spatial_A:
                strtree_A = STRTree([p for _, p in spatial_A])
            timers['strtree_build'] += time.perf_counter() - t0

            # --- Expand SIDE B ---
            new_frontier_B = []
            for curr in frontier_B:
                t0 = time.perf_counter()
                vis_poly = get_visibility_polygon(curr, self.env.d, self.env.obstructs)
                timers['visibility'] += time.perf_counter() - t0
                tree_B[curr]['poly'] = vis_poly
                spatial_B.append((curr, vis_poly))

                t0 = time.perf_counter()
                if strtree_A is not None:
                    for idx in strtree_A.query(vis_poly):
                        pos_A, poly_A = spatial_A[idx]
                        if vis_poly.intersects(poly_A):
                            timers['intersection'] += time.perf_counter() - t0
                            self._print_timers(timers, time.perf_counter() - solve_start)
                            inter = vis_poly.intersection(poly_A)
                            bridge_pt = (inter.centroid.x, inter.centroid.y)
                            return self._reconstruct(tree_A, tree_B, pos_A, curr, bridge_pt)
                timers['intersection'] += time.perf_counter() - t0

                # Only compute windows if we didn't terminate above
                t0 = time.perf_counter()
                windows = self.env.get_windows(vis_poly)
                timers['windows'] += time.perf_counter() - t0

                step_data = {'curr_pos': curr, 'vis_poly': vis_poly, 'side': 'B', 'children': []}
                for win in windows:
                    mx, my = win.centroid.x, win.centroid.y
                    win_key = (round(mx, 4), round(my, 4))

                    t0 = time.perf_counter()
                    if win_key in seen_windows_B:
                        timers['pruning'] += time.perf_counter() - t0
                        continue
                    seen_windows_B.add(win_key)
                    timers['pruning'] += time.perf_counter() - t0

                    mid_pt = (mx, my)
                    if mid_pt not in tree_B:
                        tree_B[mid_pt] = {'parent': curr, 'poly': None, 'id': node_counter, 'side': 'B'}
                        new_frontier_B.append(mid_pt)
                        step_data['children'].append({'pos': mid_pt, 'id': node_counter})
                        node_counter += 1
                self.history.append(step_data)

            frontier_B = new_frontier_B

            # Rebuild strtree_B for A to use in the next round
            t0 = time.perf_counter()
            if spatial_B:
                strtree_B = STRTree([p for _, p in spatial_B])
            timers['strtree_build'] += time.perf_counter() - t0

        self._print_timers(timers, time.perf_counter() - solve_start)
        return None

    def _print_timers(self, timers, total):
        print("\n--- Solver Timing Breakdown ---")
        accounted = sum(timers.values())
        for name, t in timers.items():
            pct = 100 * t / total if total > 0 else 0
            print(f"  {name:<20s} {t:.4f}s  ({pct:.1f}%)")
        print(f"  {'other':<20s} {total - accounted:.4f}s  ({100*(total-accounted)/total:.1f}%)")
        print(f"  {'TOTAL':<20s} {total:.4f}s")
        print("--------------------------------\n")

    def _reconstruct(self, tree_A, tree_B, meet_A, meet_B, bridge_pt):
        path_A = []
        curr = meet_A
        while curr:
            path_A.append(curr)
            curr = tree_A[curr]['parent']
        path_A = path_A[::-1]
        
        path_B = []
        curr = meet_B
        while curr:
            path_B.append(curr)
            curr = tree_B[curr]['parent']
            
        return path_A + [bridge_pt] + path_B

# ==============================================================================
# 4. ANIMATION & RUNNER
# ==============================================================================



class AnimationController:
    def __init__(self, fig, env, history, path=None):
        # Left subplot: animation
        self.ax = fig.add_subplot(1, 2, 1)
        self.ax.set_aspect('equal', adjustable='box')

        # Right subplot: final path shown immediately
        self.ax2 = fig.add_subplot(1, 2, 2)
        self.ax2.set_aspect('equal', adjustable='box')

        self.env, self.history, self.path = env, history, path

        for ax in [self.ax, self.ax2]:
            ax.set_xlim(-env.bounds, env.bounds)
            ax.set_ylim(-env.bounds, env.bounds)
            for obs in env.obstacles:
                ax.add_patch(MplPolygon(list(obs.exterior.coords), facecolor='#2c3e50', zorder=10))

        # Draw final path instantly on right subplot
        if path:
            xs, ys = zip(*path)
            self.ax2.plot(xs, ys, 'g-', linewidth=3, zorder=100, label='Min-Link Path')
            self.ax2.plot(xs[0], ys[0], 'ro', markersize=8, zorder=110)
            self.ax2.plot(xs[-1], ys[-1], 'bo', markersize=8, zorder=110)
            for pt in path[1:-1]:
                self.ax2.plot(pt[0], pt[1], 'g^', markersize=7, zorder=105)
            self.ax2.legend(loc='upper right')

        self.scat_A = self.ax.scatter([], [], c='red', marker='x', s=40, label='Frontier A', zorder=25)
        self.scat_B = self.ax.scatter([], [], c='blue', marker='x', s=40, label='Frontier B', zorder=25)

    def update(self, frame):
        if frame >= len(self.history):
            if self.path and frame == len(self.history):
                xs, ys = zip(*self.path)
                self.ax.plot(xs, ys, 'g-', linewidth=3, zorder=100, label='Min Link Path')
                self.ax.legend(loc='upper right')
            return
        
        data = self.history[frame]
        color = 'yellow' if data['side'] == 'A' else 'cyan'
        vis_poly = data['vis_poly']

        # Handling MultiPolygon to fix the AttributeError
        if vis_poly.geom_type == 'Polygon':
            polys_to_draw = [vis_poly]
        elif vis_poly.geom_type == 'MultiPolygon':
            polys_to_draw = list(vis_poly.geoms)
        else:
            polys_to_draw = []

        for poly in polys_to_draw:
            if not poly.is_empty:
                patch = MplPolygon(list(poly.exterior.coords), closed=True, 
                                   facecolor=color, alpha=0.15, edgecolor=color, zorder=5)
                self.ax.add_patch(patch)
        
        pts = [c['pos'] for c in data['children']]
        if pts:
            if data['side'] == 'A': 
                self.scat_A.set_offsets(pts)
            else: 
                self.scat_B.set_offsets(pts)

def run():
    print("Loading Map...")
    gdf = ox.features_from_point((29.7604, -95.3698), {"building": True}, dist=1600).to_crs(epsg=32615)
    env = Environment2D(gdf, bounds=1600)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-env.bounds, env.bounds)
    ax.set_ylim(-env.bounds, env.bounds)

    for obs in env.obstacles:
        ax.fill(*obs.exterior.xy, color='#2c3e50', zorder=10)

    clicks = []

    def on_click(event):
        if event.xdata is None or len(clicks) >= 2: return
        clicks.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro' if len(clicks) == 1 else 'bo', markersize=10, zorder=30)
        fig.canvas.draw()

        if len(clicks) == 2:
            solver = MinLinkSolver(env)
            path = solver.solve(clicks[0], clicks[1])

            fig.clf()
            fig.set_size_inches(18, 9)
            ctrl = AnimationController(fig, env, solver.history, path)

            # Persist start/end markers on both subplots
            for subplot_ax in [ctrl.ax, ctrl.ax2]:
                subplot_ax.plot(clicks[0][0], clicks[0][1], 'ro', markersize=8, zorder=35)
                subplot_ax.plot(clicks[1][0], clicks[1][1], 'bo', markersize=8, zorder=35)

            global ani
            ani = animation.FuncAnimation(
                fig,
                ctrl.update,
                frames=len(solver.history) + 5,
                interval=150,
                repeat=False
            )
            plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
if __name__ == "__main__":
    run()