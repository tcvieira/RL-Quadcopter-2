import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""    
             
        ## ATTEMPT 1 ##
        #current_position = self.sim.pose[:3]
        #target_position = self.target_pos[:3]
        
        # Penalize the copter when it is far away from its target position. 
        # The closer the copter gets to its target, the smaller this 
        # penalty becomes.
        #reward = 1 - .1*((abs(current_position - target_position)).sum())**2 
        #return reward
        
        ## ATEMPT 2 ##
        #curr_pos = self.sim.pose[3:6]
        #euler_angles = self.sim.pose[3:6]
        #distance_from_target = np.sqrt(((curr_pos - self.target_pos)**2).sum())
        
        # Add penalty term for large euler angles and distance from target
        #penalty = 10 * abs(euler_angles).sum() + 0.3 * distance_from_target

        # Reward per step (long flights get rewarded)
        #reward = 300
        
        #if distance_from_target < 10:
        #    reward += 10000
            
        #return reward - penalty
        
        ## ATEMPT 3 ##
        #distance = abs(self.sim.pose[:3]- self.target_pos).sum()
        #reward = np.exp(-0.0001*distance)
        
        ## ATEMPT 4 ##
        reward = 1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            if done:
                reward += 10
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
