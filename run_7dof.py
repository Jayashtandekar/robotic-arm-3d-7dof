"""Benchmark and plot 3D 7-DOF motion-planning algorithms."""
import argparse
import csv
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from planners import astar_limited,compute_metrics,densify,rrt,rrt_connect,rrt_star,shorten
from robot7dof import Robot7DOF,AXES
from test_scenes import START,GOAL

METHODS=('astar','rrt','rrt_connect','rrt_star')
DISPLAY={'astar':'A* (capped)','rrt':'RRT','rrt_connect':'RRT-Connect','rrt_star':'RRT*'}
COLORS={'astar':'#8e44ad','rrt':'#e67e22','rrt_connect':'#1976d2','rrt_star':'#00897b'}

def config(robot,args):
    return {
        'robot':'Illustrative 7-revolute-joint chain, NOT a named industrial robot',
        'joint_axes':list(AXES),'link_lengths_m':robot.lengths.tolist(),
        'joint_limits_deg':np.rad2deg(robot.limits).tolist(),
        'start_deg':np.rad2deg(START).tolist(),'goal_deg':np.rad2deg(GOAL).tolist(),
        'obstacle_spheres_xy_z_r_m':robot.obstacles.tolist(),
        'link_radius_m':robot.link_radius,'safety_margin_m':robot.safety_margin,
        'collision_sampling_max_per_joint_rad':.055,
        'self_collision_checked':robot.self_collision,
        'methods':args.methods,'iterations_per_method':args.iterations,
        'astar_max_expansions':args.astar_limit,'trials':args.trials,
        'seed':args.seed,'shortcut_postprocessing':args.shortcut,
    }

def make_sphere(ax,center,radius):
    u=np.linspace(0,2*np.pi,22)
    v=np.linspace(0,np.pi,12)
    x=center[0]+radius*np.outer(np.cos(u),np.sin(v))
    y=center[1]+radius*np.outer(np.sin(u),np.sin(v))
    z=center[2]+radius*np.outer(np.ones_like(u),np.cos(v))
    ax.plot_surface(x,y,z,color='#e74c3c',alpha=.25,linewidth=0,shade=False)

def decorate(ax,robot,name):
    ax.set_title(name,fontsize=12,weight='bold')
    for ob in robot.obstacles:make_sphere(ax,ob[:3],ob[3]+robot.link_radius+robot.safety_margin)
    ax.set_xlabel('X (m)',labelpad=1)
    ax.set_ylabel('Y (m)',labelpad=1)
    ax.set_zlabel('Z (m)',labelpad=0)
    ax.set_xlim(-.95,.95);ax.set_ylim(-.95,.95);ax.set_zlim(0.,1.85)
    ax.set_box_aspect((1.9,1.9,1.85))
    ax.view_init(elev=24,azim=-67)

def draw_arm(ax,xyz,color,alpha=.9,lw=3,label=None):
    ax.plot(xyz[:,0],xyz[:,1],xyz[:,2],'-o',color=color,lw=lw,ms=3.8,alpha=alpha,label=label)

def plot_paths(robot,records,outdir):
    paths={}
    for r in records:
        if r['trial']==1 and r['method'] not in paths:paths[r['method']]=r.get('_path')
    fig=plt.figure(figsize=(13.8,10.2))
    first,last=robot.fk(START),robot.fk(GOAL)
    for i,method in enumerate(METHODS):
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        p=paths.get(method)
        decorate(ax,robot,DISPLAY[method]+(f' | {len(p)} waypoints' if p is not None else ' | budget exhausted'))
        draw_arm(ax,first,'#2ecc71',alpha=.35,lw=2.5,label='Start')
        draw_arm(ax,last,'#2446a1',alpha=.35,lw=2.5,label='Goal')
        if p is not None:
            all_points=robot.fk_batch(densify(p,.06))
            hands=all_points[:,-1]
            ax.plot(hands[:,0],hands[:,1],hands[:,2],color=COLORS[method],lw=2.5,label='End-effector path')
            for k in [int(len(all_points)*.25),int(len(all_points)*.5),int(len(all_points)*.75)]:
                draw_arm(ax,all_points[min(k,len(all_points)-1)],COLORS[method],alpha=.25,lw=1.7)
            ax.scatter(*hands[-1],marker='*',s=75,c=COLORS[method])
        if i==0:ax.legend(loc='upper left',fontsize=7)
    fig.suptitle('7-DOF robotic arm in 3D | same scene for all planners\nTranslucent red spheres = obstacles with collision buffer',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.95])
    target=outdir/'arm_3d_paths.png'
    fig.savefig(target,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return target

def aggregate(records):
    summary=[]
    for method in METHODS:
        rr=[r for r in records if r['method']==method]
        if not rr:continue
        ok=[r for r in rr if r['success']]
        avg=lambda k:float(np.mean([r[k] for r in ok])) if ok else None
        summary.append({
            'method':DISPLAY[method],'trials':len(rr),'successes':len(ok),
            'success_rate_pct':round(100.*len(ok)/len(rr),1),
            'mean_runtime_s':round(float(np.mean([r['runtime_s'] for r in rr])),4),
            'mean_successful_joint_path_deg':None if not ok else round(avg('joint_path_deg'),2),
            'mean_successful_ee_path_m':None if not ok else round(avg('ee_path_m'),4),
            'mean_successful_min_clearance_m':None if not ok else round(avg('min_obstacle_clearance_m'),4),
            'all_successful_paths_collision_checked':all(r['collision_checked'] for r in ok),
        })
    return summary

def plot_summary(summary,outdir):
    names=[r['method'] for r in summary]
    ss=[r['success_rate_pct'] for r in summary]
    times=[r['mean_runtime_s'] for r in summary]
    dist=[r['mean_successful_joint_path_deg'] or 0. for r in summary]
    clears=[r['mean_successful_min_clearance_m'] or 0. for r in summary]
    palette=[COLORS[m] for m in METHODS if DISPLAY[m] in names]
    fig,axes=plt.subplots(2,2,figsize=(12.8,9))
    panels=(
        (axes[0,0],ss,'Success rate (%)','Higher is better',0),
        (axes[0,1],times,'Mean planning time (s)','Includes unsuccessful trials',None),
        (axes[1,0],dist,'Mean joint-space path length (deg)','Successful trials only',None),
        (axes[1,1],clears,'Min observed obstacle clearance (m)','Successful trials only',0),
    )
    for ax,values,ylabel,note,bottom in panels:
        xx=np.arange(len(names))
        ax.bar(xx,values,color=palette,alpha=.9)
        ax.set_xticks(xx,names,rotation=15,ha='right')
        ax.set_ylabel(ylabel);ax.set_title(note,fontsize=10)
        ax.grid(axis='y',alpha=.2)
        if bottom is not None:ax.set_ylim(bottom=bottom)
        for i,v in enumerate(values):
            ax.text(i,v,f'{v:.2f}' if v<10 else f'{v:.1f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('7-DOF path planning comparison (same geometry and per-method budgets)',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95])
    target=outdir/'planner_comparison.png'
    fig.savefig(target,dpi=155,bbox_inches='tight')
    plt.close(fig)
    return target

def save_gif(robot,path,outdir,label='RRT-Connect',frames=90):
    if path is None:return None
    dense=densify(path,.03)
    arclen=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
    t=np.linspace(0.,1.,frames)
    ease=10*t**3-15*t**4+6*t**5
    angles=np.column_stack([np.interp(ease*arclen[-1],arclen,dense[:,j]) for j in range(7)])
    poses=robot.fk_batch(angles)
    fig=plt.figure(figsize=(7.2,6.5))
    ax=fig.add_subplot(111,projection='3d')
    decorate(ax,robot,f'{label}: 7-DOF path animation')
    arm,=ax.plot([],[],[],'-o',lw=4,ms=5,color='#0b7285')
    hand,=ax.plot([],[],[],'-',lw=2,color='#f39c12')
    def update(i):
        pts=poses[i]
        arm.set_data_3d(pts[:,0],pts[:,1],pts[:,2])
        hand.set_data_3d(poses[:i+1,-1,0],poses[:i+1,-1,1],poses[:i+1,-1,2])
        return arm,hand
    anim=FuncAnimation(fig,update,frames=frames,interval=60,blit=False)
    target=outdir/'arm_motion_3d.gif'
    anim.save(target,writer=PillowWriter(fps=16),dpi=72)
    plt.close(fig)
    return target

def write_csv(path,data):
    if not data:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)

def run(args):
    robot=Robot7DOF()
    if not robot.valid(START)[0] or not robot.valid(GOAL)[0]:
        raise ValueError('Benchmark start or goal pose is invalid')
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    with (outdir/'experiment_config.json').open('w',encoding='utf-8') as f:
        json.dump(config(robot,args),f,indent=2)
    records=[]
    for trial in range(1,args.trials+1):
        for ix,method in enumerate(args.methods):
            rng=np.random.default_rng(args.seed+1000*trial+ix)
            t=time.perf_counter()
            if method=='astar':
                result=astar_limited(robot,START,GOAL,max_expansions=args.astar_limit)
            else:
                result={'rrt':rrt,'rrt_connect':rrt_connect,'rrt_star':rrt_star}[method](
                    robot,START,GOAL,rng,iterations=args.iterations)
            elapsed=time.perf_counter()-t
            if result.path is not None and args.shortcut:result.path=shorten(robot,result.path,rng)
            metrics=compute_metrics(robot,result.path) if result.path is not None else {}
            entry={
                'trial':trial,'method':method,'success':result.path is not None,
                'runtime_s':round(elapsed,6),'nodes_visited':result.nodes,
                'waypoints':metrics.get('waypoints'),
                'joint_path_deg':metrics.get('joint_path_deg'),
                'joint_total_variation_deg':metrics.get('joint_total_variation_deg'),
                'ee_path_m':metrics.get('ee_path_m'),
                'min_obstacle_clearance_m':metrics.get('min_obstacle_clearance_m'),
                'collision_checked':metrics.get('collision_checked'),
                'note':result.note,'_path':result.path,
            }
            if result.path is not None and not metrics['collision_checked']:
                raise RuntimeError(f'Final path has sampled collision: {method}, trial {trial}')
            records.append(entry)
            flag='PASS' if entry['success'] else 'FAIL'
            print(f'Trial {trial:02d} {DISPLAY[method]:<13} {flag:<4} t={elapsed:.3f}s nodes={result.nodes:4d}',flush=True)
            if trial==1 and result.path is not None:
                np.savetxt(outdir/f'path_{method}_deg.csv',np.rad2deg(result.path),delimiter=',',
                    header=','.join(f'q{i+1}_deg' for i in range(7)),comments='')
    summary=aggregate(records)
    write_csv(outdir/'trials.csv',[{k:v for k,v in r.items() if k!='_path'} for r in records])
    write_csv(outdir/'summary.csv',summary)
    plot_paths(robot,records,outdir)
    plot_summary(summary,outdir)
    if args.gif:
        choice=next((r for r in records if r['trial']==1 and r['method']=='rrt_connect' and r['success']),None)
        if choice is None:choice=next((r for r in records if r['trial']==1 and r['success']),None)
        if choice:print('Animated GIF:',save_gif(robot,choice['_path'],outdir,DISPLAY[choice['method']]))
        else:print('No successful path available for GIF')
    print('\nMethod          Success  Mean time(s)  Mean joint path(deg)  Mean clearance(m)')
    for r in summary:
        print(f"{r['method']:<15} {r['successes']:>2}/{r['trials']:<5}"
              f"{r['mean_runtime_s']:>10.3f}{str(r['mean_successful_joint_path_deg']):>20}"
              f"{str(r['mean_successful_min_clearance_m']):>20}")
    print('\nSaved files to',outdir.resolve())
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=1500)
    parser.add_argument('--astar-limit',type=int,default=450)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',default='results')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--shortcut',action='store_true')
    parser.add_argument('--gif',action='store_true')
    args=parser.parse_args()
    if args.trials<1 or args.iterations<1 or args.astar_limit<1:
        parser.error('trials, iterations and astar-limit must all be positive')
    run(args)
