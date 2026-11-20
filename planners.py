"""Planning in 7D joint-angle space; collision checking occurs in 3D."""
from dataclasses import dataclass
import heapq
from typing import Optional
import numpy as np
from robot7dof import Robot7DOF

@dataclass
class Plan:
    path: Optional[np.ndarray]
    nodes: int
    note: str = ''

def _check(robot,start,goal):
    if not robot.valid(start)[0]: raise ValueError('Start configuration collides or violates joint limits')
    if not robot.valid(goal)[0]: raise ValueError('Goal configuration collides or violates joint limits')

def _trace(points,parents,i):
    path=[]
    while i>=0:
        path.append(points[i].copy())
        i=parents[i]
    return path[::-1]

def _steer(a,b,step):
    d=b-a
    n=float(np.linalg.norm(d))
    return b.copy() if n<=step else a+d*(step/n)

def rrt(robot:Robot7DOF,start,goal,rng,iterations=1700,step=.32,goal_bias=.16):
    _check(robot,start,goal)
    if robot.edge_free(start,goal):return Plan(np.stack([start,goal]),2,'direct collision-free edge')
    points=np.empty((iterations+2,7))
    parents=np.full(iterations+2,-1,dtype=int)
    points[0]=start
    n=1
    for _ in range(iterations):
        target=goal if rng.random()<goal_bias else robot.sample(rng)
        nearest=int(np.argmin(np.sum((points[:n]-target)**2,axis=1)))
        new=_steer(points[nearest],target,step)
        if np.linalg.norm(new-points[nearest])<1e-9 or not robot.edge_free(points[nearest],new):continue
        points[n],parents[n]=new,nearest
        n+=1
        if np.linalg.norm(new-goal)<step*2.3 and robot.edge_free(new,goal):
            points[n],parents[n]=goal,n-1
            n+=1
            return Plan(np.asarray(_trace(points,parents,n-1)),n)
    return Plan(None,n,'iteration budget exhausted')

class _Tree:
    def __init__(self,root,label):
        self.points=[np.array(root,dtype=float)]
        self.parent=[-1]
        self.label=label
    def extend(self,robot,target,step):
        arr=np.asarray(self.points)
        closest=int(np.argmin(np.sum((arr-target)**2,axis=1)))
        if np.linalg.norm(arr[closest]-target)<1e-10:return 'reached',closest
        new=_steer(arr[closest],target,step)
        if not robot.edge_free(arr[closest],new):return 'trapped',-1
        self.points.append(new)
        self.parent.append(closest)
        i=len(self.points)-1
        return ('reached' if np.linalg.norm(new-target)<1e-9 else 'advanced'),i

def rrt_connect(robot:Robot7DOF,start,goal,rng,iterations=1700,step=.32,goal_bias=.12):
    _check(robot,start,goal)
    if robot.edge_free(start,goal):return Plan(np.stack([start,goal]),2,'direct collision-free edge')
    ta,tb=_Tree(start,'start'),_Tree(goal,'goal')
    for _ in range(iterations):
        target=tb.points[0] if rng.random()<goal_bias else robot.sample(rng)
        status,ia=ta.extend(robot,target,step)
        if status!='trapped':
            while True:
                status_b,ib=tb.extend(robot,ta.points[ia],step)
                if status_b!='advanced':break
            if status_b=='reached':
                if ta.label=='start':
                    a=_trace(ta.points,ta.parent,ia)
                    b=_trace(tb.points,tb.parent,ib)
                else:
                    a=_trace(tb.points,tb.parent,ib)
                    b=_trace(ta.points,ta.parent,ia)
                return Plan(np.asarray(a+b[-2::-1]),len(ta.points)+len(tb.points))
        ta,tb=tb,ta
    return Plan(None,len(ta.points)+len(tb.points),'iteration budget exhausted')

def rrt_star(robot:Robot7DOF,start,goal,rng,iterations=1700,step=.32,goal_bias=.16,near_radius=.78,goal_radius=.72):
    """Rewiring sampling planner (fixed radius; asymptotic optimality not claimed)."""
    _check(robot,start,goal)
    if robot.edge_free(start,goal):return Plan(np.stack([start,goal]),2,'direct collision-free edge')
    p=np.zeros((iterations+1,7))
    parents=np.full(iterations+1,-1,dtype=int)
    cost=np.zeros(iterations+1)
    children=[set() for _ in range(iterations+1)]
    p[0]=start
    n=1
    for _ in range(iterations):
        target=goal if rng.random()<goal_bias else robot.sample(rng)
        nearest=int(np.argmin(np.sum((p[:n]-target)**2,axis=1)))
        new=_steer(p[nearest],target,step)
        if np.linalg.norm(new-p[nearest])<1e-9 or not robot.edge_free(p[nearest],new):continue
        distances=np.linalg.norm(p[:n]-new,axis=1)
        nearby=np.where(distances<near_radius)[0]
        best_parent=nearest
        best_cost=cost[nearest]+distances[nearest]
        for j in nearby[np.argsort(cost[nearby]+distances[nearby])]:
            v=cost[j]+distances[j]
            if v>=best_cost-1e-8:break
            if robot.edge_free(p[j],new):
                best_parent,best_cost=int(j),float(v)
                break
        i=n
        p[i],parents[i],cost[i]=new,best_parent,best_cost
        children[best_parent].add(i)
        n+=1
        for j_ in nearby:
            j=int(j_)
            if j==best_parent or j==0:continue
            candidate_cost=cost[i]+distances[j]
            if candidate_cost<cost[j]-1e-8 and robot.edge_free(new,p[j]):
                children[parents[j]].discard(j)
                parents[j]=i
                children[i].add(j)
                delta=candidate_cost-cost[j]
                cost[j]+=delta
                stack=list(children[j])
                while stack:
                    k=stack.pop()
                    cost[k]+=delta
                    stack.extend(children[k])
    d_goal=np.linalg.norm(p[:n]-goal,axis=1)
    candidates=np.where(d_goal<=goal_radius)[0]
    best_j=None
    best_total=np.inf
    for j in candidates[np.argsort(cost[candidates]+d_goal[candidates])]:
        estimated=cost[j]+d_goal[j]
        if estimated>=best_total:break
        if robot.edge_free(p[j],goal):
            best_total=estimated
            best_j=int(j)
            break
    if best_j is None:return Plan(None,n,'no verified connection to goal')
    path=_trace(p,parents,best_j)
    if np.linalg.norm(path[-1]-goal)>1e-10:path.append(goal.copy())
    return Plan(np.asarray(path),n)

def astar_limited(robot:Robot7DOF,start,goal,max_expansions=550,grid_step=.42):
    """Lazy budget-limited 7D lattice search; not globally optimal/complete."""
    _check(robot,start,goal)
    if robot.edge_free(start,goal):return Plan(np.stack([start,goal]),2,'direct collision-free edge')
    neighbors=[(i,s) for i in range(7) for s in (-1,1)]
    zero=(0,)*7
    open_set=[(float(np.linalg.norm(start-goal)),0.,zero)]
    g={zero:0.}
    parents={zero:None}
    closed=set()
    while open_set and len(closed)<max_expansions:
        _,gpop,current=heapq.heappop(open_set)
        if current in closed or gpop>g[current]+1e-10:continue
        closed.add(current)
        q=start+grid_step*np.asarray(current)
        if np.linalg.norm(q-goal)<=grid_step*1.25 and robot.edge_free(q,goal):
            states=[]
            idx=current
            while idx is not None:
                states.append(start+grid_step*np.asarray(idx))
                idx=parents[idx]
            states.reverse()
            if np.linalg.norm(states[-1]-goal)>1e-10:states.append(goal.copy())
            return Plan(np.asarray(states),len(closed),'budgeted 7D lattice')
        for axis,direction in neighbors:
            nb=list(current);nb[axis]+=direction;nb=tuple(nb)
            if nb in closed:continue
            q2=start+grid_step*np.asarray(nb)
            if np.any(q2<robot.limits[:,0]) or np.any(q2>robot.limits[:,1]):continue
            if not robot.edge_free(q,q2):continue
            cost=g[current]+grid_step
            if cost<g.get(nb,np.inf)-1e-10:
                g[nb],parents[nb]=cost,current
                heuristic=float(np.linalg.norm(q2-goal))
                heapq.heappush(open_set,(cost+heuristic,cost,nb))
    return Plan(None,len(closed),'expansion budget exhausted or no grid path')

def shorten(robot:Robot7DOF,path:np.ndarray,rng,attempts=120)->np.ndarray:
    if path is None:return path
    points=[x.copy() for x in path]
    for _ in range(attempts):
        if len(points)<3:break
        a,b=sorted(rng.choice(len(points),2,replace=False))
        if b>a+1 and robot.edge_free(points[a],points[b]):
            points=points[:a+1]+points[b:]
    return np.asarray(points)

def densify(path,max_delta=.04):
    out=[np.asarray(path[0],float)]
    for a,b in zip(path[:-1],path[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/max_delta)))
        out.extend(a+(b-a)*t for t in np.linspace(0.,1.,n+1)[1:])
    return np.asarray(out)

def compute_metrics(robot,path):
    dense=densify(path)
    hands=robot.fk_batch(dense)[:,-1,:]
    return {
        'waypoints':int(len(path)),
        'joint_path_deg':float(np.rad2deg(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())),
        'joint_total_variation_deg':float(np.rad2deg(np.abs(np.diff(path,axis=0)).sum())),
        'ee_path_m':float(np.linalg.norm(np.diff(hands,axis=0),axis=1).sum()),
        'min_obstacle_clearance_m':float(robot.obstacle_clearance(dense).min()),
        'collision_checked':bool(all(robot.edge_free(a,b) for a,b in zip(path[:-1],path[1:]))),
    }
