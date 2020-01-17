import numpy as np
import pybullet as p
import itertools

class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []
        
        self.time = 0.0
        self.dt = 1./250.


    def reset(self):
        """
        Moves the robot back to its initial position 
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        """
        returns a list of received messages, each element of the list is a tuple (a,b)
        where a= id of the sending robot and b= message (can be any object, list, etc chosen by user)
        Note that the message will only be received if the robot is a neighbor (i.e. is close enough)
        """
        return self.messages_received
        
    def send_message(self, robot_id, message):
        """
        sends a message to robot with id number robot_id, the message can be any object, list, etc
        """
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        """
        returns a list of neighbors (i.e. robots within 2m distance) to which messages can be sent
        """
        return self.neighbors
    
    def compute_controller(self):
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """

        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position]) and relative desired position

        # get info from neighbors
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()

        # send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, pos)
        dx = 0.
        dy = 0.

        # I use time to decide which control law to use

        # Task 1: Make a square formation in the room
        if self.time < 2:
            # Set desired position for robots, each with index corresponding to robot id
            des_pos = [[1, -0.5], [1, 0.5], [1.5, -0.5], [1.5, 0.5], [2, -0.5], [2, 0.5]]

            # Get desired position for current robot
            id = self.id
            des_pos_x = des_pos[id][0]
            des_pos_y = des_pos[id][1]

            if messages:
                for m in messages:
                    dx += m[1][0] - pos[0] + (des_pos_x - des_pos[m[0]][0])
                    dy += m[1][1] - pos[1] + (des_pos_y - des_pos[m[0]][1])

                # Set a K for the control law to make it converge faster:
                dx *= 10.
                dy *= 10.

        # Task 2: Get all the robot out of the room and make a circle formation outside

        # First, Let robots get out of the room in a line
        elif self.time < 9:
            id = self.id

            if messages:
                for m in messages:
                    # Find robot 1 for robot 0, robot 3 for robot 1 etc... except robot 5, which is leader robot
                    if (id == 0 and m[0] == 1) or (id == 1 and m[0] == 3) or (id == 3 and m[0] == 5) \
                            or (id == 2 and m[0] == 0) or (id == 4 and m[0] == 2):
                        r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                        dx = - (pos[0] - m[1][0]) * ((50 / r_ij) - 50 * 0.3 / (r_ij ** 2)) + 5.
                        dy = - (pos[1] - m[1][1]) * ((50 / r_ij) - 50 * 0.3 / (r_ij ** 2)) + 5.
                if id == 5:
                    # leader 5 follows a virtual potential field
                    h_ij = np.linalg.norm([pos[0] - 2.8, pos[1] - 6])
                    if h_ij > 1:
                        dx = - (pos[0] - 2.8) * ((10 / h_ij) - 10 * 0.5 / (h_ij ** 2)) + 5.
                        dy = - (pos[1] - 6) * ((10 / h_ij) - 10 * 0.5 / (h_ij ** 2)) + 15.

        # Second, Let robots make a circle formation
        elif self.time < 12:

            if messages:
                for m in messages:
                    # control the distance between robots, radius is 1.2
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    if r_ij < 1.5: # r_1
                        dx += - (pos[0] - m[1][0]) * ((50 / r_ij) - 50 * 1.2 / (r_ij ** 2))
                        dy += - (pos[1] - m[1][1]) * ((50 / r_ij) - 50 * 1.2 / (r_ij ** 2))
                # And follows a virtual potential field, with radius = 0.8
                h_ij = np.linalg.norm([pos[0] - 2, pos[1] - 4])
                if h_ij < 1.5: # d_1
                    dx += - (pos[0] - 2) * ((250 / h_ij) - 250 * 0.8 / (h_ij ** 2))
                    dy += - (pos[1] - 4) * ((250 / h_ij) - 250 * 0.8 / (h_ij ** 2))

        # Task3: Move the purple ball on the purple square
        # First, decrease the radius of circle for virtual leader to cage the ball
        elif self.time < 13.5:

            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    if r_ij < 1.5:  # r_1
                        dx += - (pos[0] - m[1][0]) * ((10 / r_ij) - 10 * 0.6 / (r_ij ** 2))
                        dy += - (pos[1] - m[1][1]) * ((10 / r_ij) - 10 * 0.6 / (r_ij ** 2))
                # And follows a virtual potential field, with radius = 0.5
                h_ij = np.linalg.norm([pos[0] - 2, pos[1] - 4])
                if h_ij < 1.5:  # d_1
                    dx += - (pos[0] - 2) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))
                    dy += - (pos[1] - 4) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))

        # Second, move the center of virtual leader to move the ball
        elif self.time < 17.5:

            # Keep the cage formation
            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    dx += - (pos[0] - m[1][0]) * ((50 / r_ij) - 50 * 0.6 / (r_ij ** 2))
                    dy += - (pos[1] - m[1][1]) * ((50 / r_ij) - 50 * 0.6 / (r_ij ** 2))
                # And sets a new virtual potential field, with center at [2.5,5.5]
                h_ij = np.linalg.norm([pos[0] - 2.5, pos[1] - 5.5])
                dx += - (pos[0] - 2.5) * ((50 / h_ij) - 50 * 0.4 / (h_ij ** 2))
                dy += - (pos[1] - 5.5) * ((50 / h_ij) - 50 * 0.4 / (h_ij ** 2))
                #  Give a dissipative force if robots are far away from the new virtual potential field center
                if h_ij > 0.5:
                    dx += 1.
                    dy += 3.

        # Task4: Move the red ball on the red square
        # First, move the robots away from purple ball
        elif self.time < 30:
            # Get desired position for current robot
            id = self.id

            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    dx += - (pos[0] - m[1][0]) * ((2 / r_ij) - 2 * 0.6 / (r_ij ** 2))
                    dy += - (pos[1] - m[1][1]) * ((2 / r_ij) - 2 * 0.6 / (r_ij ** 2))

                    # robot 0 is leader robot, others follow one designated neighbour
                    if (id == 1 and m[0] == 3) or (id == 3 and m[0] == 5) or (id == 5 and m[0] == 4) \
                            or (id == 4 and m[0] == 2) or (id == 2 and m[0] == 0):
                        #r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                        dx += - (pos[0] - m[1][0]) * ((50 / r_ij) - 50 * 0.5 / (r_ij ** 2)) + 5.
                        dy += - (pos[1] - m[1][1]) * ((50 / r_ij) - 50 * 0.5 / (r_ij ** 2)) + 5.
                if id == 0:
                    # leader 0 follows a virtual potential field
                    # first go to [5,  5.5]
                    if self.time < 19.:
                        h_ij = np.linalg.norm([pos[0] - 5, pos[1] - 5.5])
                        if h_ij > 1:
                            dx += - (pos[0] - 5) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))
                            dy += - (pos[1] - 5.5) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))
                    else:
                        # then go to [5, 0.5]
                        h_ij = np.linalg.norm([pos[0] - 5, pos[1] - 0.5])
                        if h_ij > 0.5:
                            dx += - (pos[0] - 5) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))
                            dy += - (pos[1] - 0.5) * ((50 / h_ij) - 50 * 0.5 / (h_ij ** 2))
        # Second, Cage the Red ball
        elif self.time < 37:
            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.8
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    dx += - (pos[0] - m[1][0]) * ((30 / r_ij) - 30 * 0.8 / (r_ij ** 2))
                    dy += - (pos[1] - m[1][1]) * ((30 / r_ij) - 30 * 0.8 / (r_ij ** 2))
                    if self.time > 35: # Shrink the surrounding circle
                        dx += - (pos[0] - m[1][0]) * ((30 / r_ij) - 30 * 0.5 / (r_ij ** 2))
                        dy += - (pos[1] - m[1][1]) * ((30 / r_ij) - 30 * 0.5 / (r_ij ** 2))
                # And follows a virtual potential field with center at the red ball ([4,2]), with radius = 0.5
                h_ij = np.linalg.norm([pos[0] - 4, pos[1] - 2])
                dx += - (pos[0] - 4) * ((100 / h_ij) - 100 * 0.5 / (h_ij ** 2))
                dy += - (pos[1] - 2) * ((100 / h_ij) - 100 * 0.5 / (h_ij ** 2))

        # Third, Move the red ball
        elif self.time < 48:
            # Keep the cage formation
            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    dx += - (pos[0] - m[1][0]) * ((6 / r_ij) - 6 * 0.6 / (r_ij ** 2))
                    dy += - (pos[1] - m[1][1]) * ((6 / r_ij) - 6 * 0.6 / (r_ij ** 2))
                # And sets a new virtual potential field, with center at [0.5,5.5]
                h_ij = np.linalg.norm([pos[0] - 0.5, pos[1] - 5.5])
                dx += - (pos[0] - 0.5) * ((30 / h_ij) - 30 * 0.4 / (h_ij ** 2))
                dy += - (pos[1] - 5.5) * ((30 / h_ij) - 30 * 0.4 / (h_ij ** 2))

        # Task 5: Get back into the room and make a diamond formation
        # First, increase the radius
        elif self.time < 50:
            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    dx += - (pos[0] - m[1][0]) * ((30 / r_ij) - 30 * 0.6 / (r_ij ** 2))
                    dy += - (pos[1] - m[1][1]) * ((30 / r_ij) - 30 * 0.6 / (r_ij ** 2))
                # And follows a virtual potential field with center at the red ball [0.5,5.5], with radius = 0.6
                h_ij = np.linalg.norm([pos[0] - 0.5, pos[1] - 5.5])
                dx += - (pos[0] - 0.5) * ((50 / h_ij) - 50 * 0.6 / (h_ij ** 2))
                dy += - (pos[1] - 5.5) * ((50 / h_ij) - 50 * 0.6 / (h_ij ** 2))

        # Second, Go back into the room
        elif self.time < 60:

            id = self.id

            if messages:
                for m in messages:
                    # control the distance between robots, radius is 0.6
                    #r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                    # dx += - (pos[0] - m[1][0]) * ((5 / r_ij) - 5 * 0.6 / (r_ij ** 2))
                    # dy += - (pos[1] - m[1][1]) * ((5 / r_ij) - 5 * 0.6/ (r_ij ** 2))

                    # robot 4 is leader robot, others follow one designated neighbour
                    if (id == 2 and m[0] == 4) or (id == 0 and m[0] == 2) or (id == 3 and m[0] == 0) \
                            or (id == 1 and m[0] == 3) or (id == 5 and m[0] == 1):
                        r_ij = np.linalg.norm([pos[0] - m[1][0], pos[1] - m[1][1]])
                        dx += - (pos[0] - m[1][0]) * ((50 / r_ij) - 50 * 0.3 / (r_ij ** 2))
                        dy += - (pos[1] - m[1][1]) * ((50 / r_ij) - 50 * 0.3 / (r_ij ** 2))
                if id == 4:
                    # leader 4 follows a virtual potential field
                    # first go to [2.5,  5.5]
                    if self.time < 53:
                        h_ij = np.linalg.norm([pos[0] - 2.5, pos[1] - 5.5])
                        dx += - (pos[0] - 2.5) * ((30 / h_ij) - 30 * 0.05 / (h_ij ** 2)) + 3.
                        dy += - (pos[1] - 5.5) * ((30 / h_ij) - 30 * 0.05 / (h_ij ** 2)) + 3.
                    else:
                        # then go to [2.5, -1.5]
                        h_ij = np.linalg.norm([pos[0] - 2.5, pos[1] + 1.5])
                        dx += - (pos[0] - 2.5) * ((30 / h_ij) - 30 * 0.05 / (h_ij ** 2))
                        dy += - (pos[1] + 1.5) * ((30 / h_ij) - 30 * 0.05 / (h_ij ** 2))

        # Third, move up
        elif self.time < 65:
            if messages:
                for m in messages:
                    #
                    h_ij = np.linalg.norm([pos[0] - 1.5, 0])
                    dx += - (pos[0] - 1.5) * ((10 / (h_ij ** 1)) - 10 * 0.8 / (h_ij ** 2))
                    #dy += - (pos[1] - m[1][1]) * ((10 / (h_ij ** 1)) - 10 * 0.8 / (h_ij ** 2))

        # Forth, form a diamond
        elif self.time < 70:
            # Set desired position for robots, each with index corresponding to robot id
            des_pos = [[1.85, -0.25], [1.4, 0.5], [1.4, -0.5], [1.7, 0], [0.95, -0.25], [0.5, 0]]

            # Get desired position for current robot
            id = self.id
            des_pos_x = des_pos[id][0]
            des_pos_y = des_pos[id][1]

            if messages:
                for m in messages:
                    dx += m[1][0] - pos[0] + (des_pos_x - des_pos[m[0]][0])
                    dy += m[1][1] - pos[1] + (des_pos_y - des_pos[m[0]][1])

                # Set a K for the control law to make it converge faster:
                dx *= 10.
                dy *= 10.

        # compute velocity change for the wheels
        vel_norm = np.linalg.norm([dx, dy])  # norm of desired velocity
        if vel_norm < 0.01:
            vel_norm = 0.01
        des_theta = np.arctan2(dy / vel_norm, dx / vel_norm)
        right_wheel = np.sin(des_theta - rot) * vel_norm + np.cos(des_theta - rot) * vel_norm
        left_wheel = -np.sin(des_theta - rot) * vel_norm + np.cos(des_theta - rot) * vel_norm

        self.set_wheel_velocity([left_wheel, right_wheel])

        print(self.time)
        self.time += self.dt

