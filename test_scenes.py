"""Shared start/goal and geometry for the standard 7-DOF benchmark."""
import numpy as np
START_DEG = np.array([0.,20.,-35.,25.,10.,-25.,10.])
GOAL_DEG = np.array([90.,-30.,30.,-30.,45.,35.,-25.])
START = np.deg2rad(START_DEG)
GOAL = np.deg2rad(GOAL_DEG)
