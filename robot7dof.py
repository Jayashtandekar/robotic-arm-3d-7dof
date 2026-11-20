"""Educational seven-joint serial arm, 3D forward kinematics and collision model."""
from dataclasses import dataclass, field
import numpy as np

AXES = ("z", "y", "x", "y", "x", "y", "x")
LINK_LENGTHS = np.array([.34, .30, .28, .25, .23, .18, .12])
JOINT_LIMITS_DEG = np.array([[-170,170],[-120,120],[-170,170],[-120,120],[-170,170],[-120,120],[-175,175]], dtype=float)

def rot_batch(axis, a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3), float)
    if axis == "x":
        R[:,0,0] = 1
        R[:,1,1], R[:,1,2], R[:,2,1], R[:,2,2] = c, -s, s, c
    elif axis == "y":
        R[:,1,1] = 1
        R[:,0,0], R[:,0,2], R[:,2,0], R[:,2,2] = c, s, -s, c
    elif axis == "z":
        R[:,2,2] = 1
        R[:,0,0], R[:,0,1], R[:,1,0], R[:,1,1] = c, -s, s, c
    else:
        raise ValueError(axis)
    return R

def segment_distance_batch(p1, q1, p2, q2):
    """Closest distance of two line segments, including endpoints (batched Mx3)."""
    u, v, w = q1-p1, q2-p2, p1-p2
    aa = np.einsum("ij,ij->i",u,u)
    bb = np.einsum("ij,ij->i",u,v)
    cc = np.einsum("ij,ij->i",v,v)
    dd = np.einsum("ij,ij->i",u,w)
    ee = np.einsum("ij,ij->i",v,w)
    D = aa*cc-bb*bb
    eps=1e-12
    sN, tN = bb*ee-cc*dd, aa*ee-bb*dd
    sD, tD = D.copy(), D.copy()
    parallel=D<eps
    sN=np.where(parallel,0.,sN)
    sD=np.where(parallel,1.,sD)
    tN=np.where(parallel,ee,tN)
    tD=np.where(parallel,cc,tD)
    low=(~parallel)&(sN<0)
    sN=np.where(low,0.,sN)
    tN=np.where(low,ee,tN)
    tD=np.where(low,cc,tD)
    high=(~parallel)&(sN>sD)
    sN=np.where(high,sD,sN)
    tN=np.where(high,ee+bb,tN)
    tD=np.where(high,cc,tD)
    low_t=tN<0
    tN=np.where(low_t,0.,tN)
    sN=np.where(low_t&(-dd<0),0.,sN)
    sN=np.where(low_t&(-dd>aa),sD,sN)
    middle=low_t&(-dd>=0)&(-dd<=aa)
    sN=np.where(middle,-dd,sN)
    sD=np.where(middle,aa,sD)
    high_t=tN>tD
    tN=np.where(high_t,tD,tN)
    expr=-dd+bb
    sN=np.where(high_t&(expr<0),0.,sN)
    sN=np.where(high_t&(expr>aa),sD,sN)
    middle=high_t&(expr>=0)&(expr<=aa)
    sN=np.where(middle,expr,sN)
    sD=np.where(middle,aa,sD)
    sc=np.divide(sN,sD,out=np.zeros_like(sN),where=np.abs(sD)>eps)
    tc=np.divide(tN,tD,out=np.zeros_like(tN),where=np.abs(tD)>eps)
    return np.linalg.norm(w+sc[:,None]*u-tc[:,None]*v,axis=1)

@dataclass
class Robot7DOF:
    lengths: np.ndarray = field(default_factory=lambda:LINK_LENGTHS.copy())
    limits: np.ndarray = field(default_factory=lambda:np.deg2rad(JOINT_LIMITS_DEG))
    obstacles: np.ndarray = field(default_factory=lambda:np.array([[.10,.20,1.16,.12],[-.29,.20,.85,.11]],float))
    link_radius: float = .026
    safety_margin: float = .024
    self_collision: bool = True

    def fk(self,q):
        return self.fk_batch(np.asarray(q,float)[None,:])[0]

    def fk_batch(self,q):
        q=np.atleast_2d(np.asarray(q,float))
        if q.shape[1]!=7: raise ValueError("Expected (7,) or (M,7)")
        m=len(q)
        pts=np.zeros((m,8,3),float)
        R=np.broadcast_to(np.eye(3),(m,3,3)).copy()
        for i,(axis,length) in enumerate(zip(AXES,self.lengths)):
            R=R@rot_batch(axis,q[:,i])
            pts[:,i+1]=pts[:,i]+length*R[:,:,2]
        return pts

    def obstacle_clearance(self,q,points=None):
        pts=self.fk_batch(q) if points is None else points
        if not len(self.obstacles):return np.full(len(pts),np.inf)
        a,b=pts[:,:-1,:],pts[:,1:,:]
        ab=b-a
        centers,radii=self.obstacles[:,:3],self.obstacles[:,3]
        ac=centers[None,None,:,:]-a[:,:,None,:]
        proj=np.sum(ac*ab[:,:,None,:],axis=-1)
        segnorm=np.sum(ab**2,axis=-1)[:,:,None]
        t=np.clip(proj/np.maximum(segnorm,1e-12),0.,1.)
        nearest=a[:,:,None,:]+t[...,None]*ab[:,:,None,:]
        dist=np.linalg.norm(nearest-centers[None,None,:,:],axis=-1)
        return (dist-radii[None,None,:]-self.link_radius).min(axis=(1,2))

    def self_clearance(self,q,points=None):
        pts=self.fk_batch(q) if points is None else points
        pairs=[(i,j) for i in range(7) for j in range(i+2,7)]
        left=np.array([i for i,_ in pairs],int)
        right=np.array([j for _,j in pairs],int)
        m=len(pts)
        p1,q1=pts[:,left],pts[:,left+1]
        p2,q2=pts[:,right],pts[:,right+1]
        gap=segment_distance_batch(p1.reshape(-1,3),q1.reshape(-1,3),p2.reshape(-1,3),q2.reshape(-1,3))
        return gap.reshape(m,-1).min(axis=1)-2*self.link_radius

    def valid(self,q):
        q=np.atleast_2d(np.asarray(q,float))
        in_limits=np.all((q>=self.limits[:,0]-1e-10)&(q<=self.limits[:,1]+1e-10),axis=1)
        points=self.fk_batch(q)
        collision_free=self.obstacle_clearance(q,points)>self.safety_margin
        if self.self_collision and np.any(in_limits&collision_free):
            eligible=in_limits&collision_free
            ok=self.self_clearance(q[eligible],points[eligible])>self.safety_margin
            collision_free[eligible]&=ok
        return in_limits&collision_free

    def edge_free(self,a,b,step=.055):
        a,b=np.asarray(a,float),np.asarray(b,float)
        n=max(2,int(np.ceil(np.max(np.abs(b-a))/step))+1)
        qq=a[None,:]+np.linspace(0,1,n)[:,None]*(b-a)[None,:]
        return bool(np.all(self.valid(qq)))

    def sample(self,rng):
        return rng.uniform(self.limits[:,0],self.limits[:,1])
