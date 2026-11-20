import unittest
import numpy as np
from robot7dof import Robot7DOF,segment_distance_batch
from test_scenes import START,GOAL
from planners import astar_limited,compute_metrics,rrt,rrt_connect,rrt_star

class MotionPlanningTests(unittest.TestCase):
    def setUp(self):
        self.robot=Robot7DOF()

    def test_dimensions_and_segment_lengths(self):
        pose=self.robot.fk(START)
        self.assertEqual(pose.shape,(8,3))
        self.assertTrue(np.allclose(pose[0],0.))
        self.assertTrue(np.allclose(np.linalg.norm(np.diff(pose,axis=0),axis=1),self.robot.lengths))
        self.assertFalse(np.allclose(pose[:,1],0.),'not a 2D planar robot')

    def test_scene_endpoints_and_colliding_direct_edge(self):
        self.assertTrue(self.robot.valid(START)[0])
        self.assertTrue(self.robot.valid(GOAL)[0])
        self.assertFalse(self.robot.edge_free(START,GOAL))

    def test_sphere_hits_link_even_when_tip_clear(self):
        robot=Robot7DOF(obstacles=np.array([[0.,0.,.17,.10]]))
        q=np.zeros(7)
        self.assertLess(robot.obstacle_clearance(q)[0],0)
        self.assertGreater(np.linalg.norm(robot.fk(q)[-1]-robot.obstacles[0,:3]),1.)
        self.assertFalse(robot.valid(q)[0])

    def test_parallel_and_crossing_segment_distance(self):
        p=np.array([[0.,0.,0.]])
        q=np.array([[1.,0.,0.]])
        d=segment_distance_batch(p,q,np.array([[0.,1.,0.]]),np.array([[1.,1.,0.]]))
        self.assertAlmostEqual(d[0],1.)
        d=segment_distance_batch(p,q,np.array([[.5,-1.,0.]]),np.array([[.5,1.,0.]]))
        self.assertAlmostEqual(d[0],0.)

    def test_joint_limit_violations(self):
        bad=START.copy()
        bad[1]=self.robot.limits[1,1]+.01
        self.assertFalse(self.robot.valid(bad)[0])

    def _check_plan(self,plan):
        self.assertIsNotNone(plan.path)
        self.assertTrue(np.allclose(plan.path[0],START))
        self.assertTrue(np.allclose(plan.path[-1],GOAL))
        m=compute_metrics(self.robot,plan.path)
        self.assertTrue(m['collision_checked'])
        self.assertGreater(m['min_obstacle_clearance_m'],self.robot.safety_margin)

    def test_rrt(self):
        self._check_plan(rrt(self.robot,START,GOAL,np.random.default_rng(1008),iterations=1500))

    def test_rrt_connect(self):
        self._check_plan(rrt_connect(self.robot,START,GOAL,np.random.default_rng(1009),iterations=1500))

    def test_rrt_star(self):
        self._check_plan(rrt_star(self.robot,START,GOAL,np.random.default_rng(1010),iterations=1500))

    def test_astar_obeys_expansion_limit(self):
        plan=astar_limited(self.robot,START,GOAL,max_expansions=30)
        self.assertLessEqual(plan.nodes,30)

if __name__=='__main__':
    unittest.main()
