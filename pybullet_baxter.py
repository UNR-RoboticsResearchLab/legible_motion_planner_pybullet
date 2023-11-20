import os
import sys
import numpy as np
import argparse
from termcolor import cprint
import pybullet as p
import re
import random
import math
import copy
import time

from pybullet_planning import load_pybullet, connect, wait_for_user, set_camera, wait_for_duration
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, joints_from_names, plan_joint_motion, safe_zip, plan_cartesian_motion, get_velocity, get_joint_velocities, get_joint_state, set_joint_positions, get_max_velocity, get_joint_positions
from pybullet_planning import Pose, Point, Euler, set_pose, create_box

HERE = os.path.dirname(__file__)
BAXTER_ROBOT_URDF = os.path.join(HERE, 'data', 'baxter_common', 'baxter_description', 'urdf', 'toms_baxter.urdf')

TUTORIALS = {'Baxter1','Baxter2'}

def getJointRanges(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)

        if includeFixed or jointInfo[3] > -1:

            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(-2)
            upperLimits.append(2)
            jointRanges.append(2)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses


def setMotors(bodyId, jointPoses, arm_joints):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    jointPoses = list(jointPoses)
    # print("jointPoses:", jointPoses)
    arm_joints = list(arm_joints)
    # print("arm_joints:", arm_joints)

    max_velocities = []
    for arm_joint in arm_joints:
        max_vel = get_max_velocity(bodyId, arm_joint) 
        max_velocities.append(max_vel)      
    # print("max_velocities:", max_velocities)     

    # velocity_joint = list( get_joint_velocities(bodyId, arm_joints) )
    # # print("velocity_joint:", velocity_joint)
    # for i in range(len(velocity_joint)):
    #     # print("i:", i)
    #     velocity_joint[i] = 1.0     
    # # print("velocities mod:", vel)  
    # max_velocity_joint = velocity_joint 

    max_velocity_joint = max_velocities 

    # while loop
    iii = 0
    while 1:
        # print("max velocity_joint:", max_velocity_joint)
        ii = 0
        for arm_joint in arm_joints:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=int(arm_joint), controlMode=p.POSITION_CONTROL, targetPosition=jointPoses[ii], maxVelocity=max_velocity_joint[ii])
            ii += 1

        p.stepSimulation()
        # time.sleep(1./240.)

        positions = get_joint_positions(bodyId, arm_joints)
        # print("current positions:", positions)
        velocity_joint = get_joint_velocities(bodyId, arm_joints)
        # print("current velocity_joint:", velocity_joint)
        tol = [ abs(a - b) for a, b in zip( jointPoses, list(positions) )]
        # print("current tol:", tol)

        iii += 1
        if iii > 240:
            for joint, value in safe_zip(arm_joints, jointPoses):
                p.resetJointState(bodyId, joint, targetValue=value)
            break
        if all(x < 0.1 for x in tol):
            for joint, value in safe_zip(arm_joints, jointPoses):
                p.resetJointState(bodyId, joint, targetValue=value)
            break
        elif any(x < 0.1 for x in tol):
            ii = 0
            for i in arm_joints:
                if tol[ii] < 0.1:
                    max_velocity_joint[ii] = 0.0
                ii += 1



def create_objects(pos): 
    # create a box to be picked up
    # see: https://pybullet-planning.readthedocs.io/en/latest/reference/generated/pybullet_planning.interfaces.env_manager.create_box.html#pybullet_planning.interfaces.env_manager.create_box
    block = create_box(0.05, 0.05, 0.05)
    block_x = pos[0]
    block_y = pos[1]
    block_z = pos[2]    
    set_pose(block, Pose(Point(x=block_x, y=block_y, z=block_z)))

def get_random_points(num_points, max_x, min_x, max_y, min_y):
    to_ret = []

    for i in range(num_points):
        x = random.random() * (max_x - min_x) + min_x
        y = random.random() * (max_y - min_y) + min_y

        to_ret.append(np.array([x, y]))

    return np.transpose(np.atleast_2d(to_ret))

def KL_divergence_gaussian_uniform (cov, q):
    if np.linalg.det(cov) <= 0:
        return 0

    return -1 - math.log(2 * math.pi) - math.log( np.linalg.det(cov) ) - math.log( q )

# This is the maximum reasonable KL_divergence (otherwise objects would have to be stacked on top of one another)
def max_KL_divergence (max_x, min_x, max_y, min_y, q_const):
    dumb_points = get_random_points(20, max_x, min_x, max_y, min_y)

    point1 = [ dumb_points[0][0], dumb_points[0][1] ]

    dumb_points = np.transpose(np.atleast_2d([ [ point1[0] + .25 * random.random(), point1[1] + .25 * random.random() ] for i in range(len(dumb_points[0]))]))

    cov_mat = np.cov(dumb_points)

    return KL_divergence_gaussian_uniform(cov_mat, q_const)

# xsi is the KL_divergence of the estimated gaussian which represents the points in objects compared to a uniform distribution in the same area
def calculate_xi( objects, mins, maxs ):

    q_const = 1

    if ( maxs[0] - mins[0] ) == 0 and ( maxs[1] - mins[1] ) == 0:
        q_const = 1
    elif ( maxs[0] - mins[0] ) == 0:
        q_const =  1 /  ( maxs[1] - mins[1] )
    elif ( maxs[1] - mins[1] ) == 0:
        q_const =  1 / ( maxs[0] - mins[0] )
    else:
        q_const =  1 / ( ( maxs[0] - mins[0] ) * ( maxs[1] - mins[1] ) )

    objects_temp = np.transpose(np.atleast_2d(objects))

    max_kl = max_KL_divergence(maxs[0], mins[0], maxs[1], mins[1], q_const)

    cov_mat = np.cov(objects_temp[:2])
    kl_score = KL_divergence_gaussian_uniform(cov_mat, q_const)

    # print(f'{kl_score} / {max_kl} = {kl_score / max_kl}')

    return math.exp(- kl_score)  


def calc_psi(startPosition, targetPosition, objectPosition):
    cpa = ( np.linalg.norm( np.cross( (np.array([objectPosition[0], objectPosition[1]]) - np.array([startPosition[0], startPosition[1]])) , \
            (np.array([ targetPosition[0], targetPosition[1] ]) - np.array([startPosition[0], startPosition[1]])) ) ) ) \
            / np.linalg.norm( (np.array([ targetPosition[0], targetPosition[1] ]) - np.array([startPosition[0], startPosition[1]])) )

    psi = math.asin( cpa / math.dist([startPosition[0], startPosition[1]], [ objectPosition[0], objectPosition[1] ]))

    psi /= ( math.pi / 2 )

    return 1 - psi

def generate_pf_plan(startPosition, targetPosition, obstaclePositions, xi, oStartPosition):
    # parameter for deciding what close enough to the goal is (1cm)
    epsilon = 0.01

    # gains parameters
    k_rep = .03
    k_rep_2d = .025 * ( 1 - xi )
    k_attr = 0.45 #attr = attractive

    # final gain param (not sure if necessary/could be that this is covered by xi)
    k_update = 0.2

    # params for the size of each respective potential field
    ## we don't want to immediately push the hand up so the max height is slightly less than the height of the start position
    max_height = math.dist([ startPosition[0], startPosition[1], targetPosition[2]], startPosition) - .005
    ## the max 2d range is scaled by xsi so that more cluttered environments have less max range
    ### NOTE: this works for now but I think that I should replace this with max_range * ( 1 - xsi )
    max_range_2d = .3 *  (1 - xi)
    ## this is the ellipsoidal limit for the z direction
    ro_max = 2 * max_height

    # the max number of point is 201 and we assume to not have converged
    max_num = 300

    # output of the algorithm is stored in total_plan
    total_plan = [ list(startPosition) ]

    # either we come within range of the goal or get stuck to stop
    while np.linalg.norm( targetPosition - total_plan[-1] ) > epsilon and len(total_plan) < max_num:

        U_grad_rep_total = np.array([0.0, 0.0, 0.0])

        eePosition = np.array(total_plan[-1])

        for j in obstaclePositions:

            ##### get repulsive force for each object in the obstacles #####
            ### force in z direction calculated using the equation of an ellipsoid ###

            # we want the radius to end at the target so that the hand can always fall towards it
            radius = math.dist(j, targetPosition)

            # enforce the max x,y range
            if radius > max_height:
                radius = max_height

            ## calculation for the placement of the two focal points of the ellipsoid ##
            a = max_height * math.cos( math.asin( radius / max_height ) )

            f1 = np.array([ j[0], j[1], j[2] - a ])
            f2 = np.array([ j[0], j[1], j[2] + a ])

            ## now the force calculation ##
            # current distance from ellipsoid focal points
            ro_b = math.dist(eePosition, f1) + math.dist(eePosition, f2)

            # max distance of ellipsoid
            ro_0 = math.dist(f1, targetPosition) + math.dist(f2, targetPosition)

            # enforce the max x,y range
            if ro_0 > ro_max:
                ro_0 = ro_max

            ## now the actual force calculation ##
            force_sphere = None

            if ro_b <= ro_0 and eePosition[2] < total_plan[0][2]:
                force_sphere = k_rep * (1/ro_0 - 1/ro_b ) * ((2 * eePosition - f1 - f2)/ro_b)
            else:
                force_sphere = [ 0, 0, 0 ]

            ### x,y force calculation using the equation of a circle with no regard for height ###
            ro_b = math.dist([eePosition[0], eePosition[1]], [ j[0], j[1] ] )

            ro_0 = math.dist([j[0], j[1]], [ targetPosition[0], targetPosition[1] ] )

            psi = calc_psi(oStartPosition, targetPosition, j)

            ## 0 out psi both if the startPosition is closer to the target than the object, or if the distance from the
            #### object is greater than the distance from the startPosition to the targetPosition
            # if ro_b >= math.dist([eePosition[0], eePosition[1]], [ targetPosition[0], targetPosition[1] ]) or \
            #     ro_0 >= math.dist([oStartPosition[0], oStartPosition[1]], [ targetPosition[0], targetPosition[1] ]) :
            #     psi = 0

            # scale the max_range by psi
            # if ro_0 > max_range_2d:
            #     ro_0 = max_range_2d

            ## calculate the 2d force, this time without taking height into account ##
            force_2d = None

            if ro_b <= ro_0:
                force_2d = k_rep_2d * (1/ro_0 - 1/ro_b ) * ( ( np.array([eePosition[0], eePosition[1]]) - np.array([j[0], j[1]]) ) / ro_b )
            else:
                force_2d = np.array([ 0, 0 ])

            # scale the force by psi
            force_2d = force_2d * psi

            # negative z goes through the tabe
            # z_force = force_sphere[2]
            #
            # if z_force < 0:
            #     z_force *= -1

            ### combine the two forces ###
            U_grad_rep = [force_2d[0], force_2d[1], force_sphere[2]]

            ##### add them to the total repulsive force
            U_grad_rep_total += U_grad_rep

        ### calculate the attractive force and the total force for the update rule ###
        ## scale the total repulsive force by xsi
        U_grad_rep_total = U_grad_rep_total * ( 1 - xi )

        ## the attractive force is linear in the direction of the target
        U_grad_attr = k_attr * (eePosition - targetPosition)

        ### this -1 is taken out of earlier calculations and finally added back here
        force = -1 * (U_grad_rep_total + U_grad_attr)

        # add the new point to the plan
        new_point = total_plan[-1] + k_update * force
        total_plan.append(new_point)

    return total_plan

def clean_trajectory(waypoints):
    to_remove = [ 1 ]

    while len(to_remove) > 0:
        to_remove = []

        # min total_distance between 3 points is 3 cm
        epsilon = .03

        i = 1

        # mark for removal all the points which are too close to each other
        while i < len(waypoints) - 1:
            if math.dist(waypoints[i - 1], waypoints[i]) + math.dist(waypoints[i], waypoints[i + 1]) < epsilon:

                to_remove.append(i)
                i += 2
            else:
                i += 1

        to_remove.reverse()

        # now remove them in reverse order so that none of the indices shift
        for i in to_remove:

            del waypoints[i]

def get_smooth_legible_trajectory(startPosition, targetPosition, obstaclePositions, xi):
    total_waypoints = []

    # get pf trajectory
    waypoints = generate_pf_plan(startPosition, targetPosition, obstaclePositions, xi, startPosition)

    # find local maximums
    local_maxs = [ startPosition ]

    for i in range(1, len(waypoints) - 1):
        if waypoints[i - 1][2] <= waypoints[i][2] and waypoints[i + 1][2] <= waypoints[i][2]:
            local_maxs.append( copy.deepcopy( waypoints[i] ) )

    local_maxs.append(targetPosition)

    # plan from main point to local max
    for i in range(len(local_maxs) - 1):
        n_waypoints = generate_pf_plan(local_maxs[i], local_maxs[i + 1], obstaclePositions, xi, startPosition)

        for j in n_waypoints:
            total_waypoints.append(copy.deepcopy(j))

    clean_trajectory(total_waypoints)

    # return updated total trajectory
    return total_waypoints          


def baxter_main(viewer=True, objects=[], target=0):
    arm='right'
    connect(use_gui=viewer)
    robot = load_pybullet(BAXTER_ROBOT_URDF, fixed_base=True)
    set_camera(yaw=90, pitch=0, distance=2, target_position=(0, 0, 0))
    endEffectorId = 26 # (right gripper right finger)    

    ik_joints = get_movable_joints(robot)
    # print(ik_joints)
    ik_joint_names = get_joint_names(robot, ik_joints)
    # print(ik_joint_names)
    arm_joint_names = [x for x in ik_joint_names if re.search('right', str(x))]
    # print(arm_joint_names)
 
    for pos_i in objects:
        create_objects(pos_i)             

    arm_joints = joints_from_names(robot, arm_joint_names)
    # print(arm_joints)
    cprint('Used joints: {}'.format(get_joint_names(robot, arm_joints)), 'yellow')

    # * now let's plan a trajectory
    cprint('Randomly sample robot start/end configuration and comptue a motion plan! (no self-collision check is performed)', 'blue')
    print('Disabled collision links needs to be given (can be parsed from a SRDF via compas_fab)')
    # set joint positions
    q2 = np.zeros(len(arm_joints))
    q2[0] = -1.300942808659576
    q2[1] = -1.1064183104668466
    q2[2] = 1.1765210844176874
    q2[3] = 2.1887659552021455
    q2[4] = -0.5662488954816365
    q2[5] =  0.8754244383386937
    q2[6] = -2.508083413019744       
    cprint('Sampled end conf: {}'.format(q2), 'cyan')

    wait_for_user('Move to the home position!')

    path = plan_joint_motion(robot, arm_joints, q2, obstacles=[], self_collisions=False)
    # print("path:", path)

    # adjusting this number will adjust the simulation speed
    for conf in path:
        for joint, value in safe_zip(arm_joints, conf):
            p.resetJointState(robot, joint, targetValue=value)         

    wait_for_user('Move to the object!')        

    # set target 
    object_num = int(target)
    obstaclePositions = []
    targetPosition = None

    # xi is calculated once after the points are set up
    max_x = -1000
    max_y = -1000
    min_x = 1000
    min_y = 1000
    for i in range(len(objects)):
        # find max and min for x and y
        if max_x < objects[i][0]:
            max_x = objects[i][0]
        if max_y < objects[i][1]:
            max_y = objects[i][1]
        if min_x > objects[i][0]:
            min_x = objects[i][0]
        if min_y > objects[i][1]:
            min_y = objects[i][1]  

        # also pick the target object out of the lineup and put the rest in the obstacles list
        if i == object_num:
            targetPosition = np.array( copy.deepcopy(objects[i]) )
        else:
            obstaclePositions.append( np.array( copy.deepcopy(objects[i]) ) )    

    xi = calculate_xi(objects, [ min_x, min_y ], [ max_x, max_y ]) 

    ee_state = p.getLinkState(robot, endEffectorId, computeForwardKinematics=1)
    startPosition = ee_state[0]
    startOrientation = ee_state[1]
    # generate pf plan uses the pf eqs to generate legible motion
    total_plan = get_smooth_legible_trajectory(startPosition, targetPosition, obstaclePositions, xi)
    # print(total_plan)

    waypoint_poses = []
    for waypoint_pose in total_plan:
        waypoint_pose = list( Pose( Point(x=waypoint_pose[0],y=waypoint_pose[1],z=waypoint_pose[2]) ) )
        waypoint_pose[1] = startOrientation
        waypoint_poses.append( tuple(waypoint_pose) ) 
    # print("waypoint_poses:", waypoint_poses) 

    first_joint = 12
    target_link = 26
    path_robot = plan_cartesian_motion(robot, first_joint, target_link, waypoint_poses)
    # print("path_robot:",path_robot)

    path = []
    for path_subset in path_robot:
        path_subset = tuple([i for i in path_subset if i != 0.0]) 
        path.append(path_subset)

    # reset to home position after showing plan
    for joint, value in safe_zip(arm_joints, q2):
        p.resetJointState(robot, joint, targetValue=value)      

    # move to object
    for conf in path:
        setMotors(robot, conf, arm_joints)

    wait_for_user('Press enter to close!')         



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--noviewer', action='store_true', help='Enables the viewer during planning, default True')
    parser.add_argument('-obj', '--objects', default='Baxter1', choices=TUTORIALS, \
        help='The name of the object setup')
    parser.add_argument('-t', '--target', default=0, help='The target object number')    
    args = parser.parse_args()
    print('Arguments:', args)

    if args.objects == 'Baxter1':
        Baxter1 = [[0.58,-0.06,-0.15],[0.58,-0.26,-0.15],[0.58,-0.46,-0.15],[0.58,-0.66,-0.15],[0.58,-0.86,-0.15]]
        print("target", args.target)
        if int(args.target) > 4:
            raise ValueError('target object number must be between 0 and 4')
        baxter_main(viewer=not args.noviewer, objects=Baxter1, target=args.target)
    elif args.objects == 'Baxter2':
        Baxter2 = [[0.379,-0.65,-0.1],[0.484,-0.805,-0.1],[0.589,-0.77,-0.1],[0.579,-0.65,-0.1],[0.704,-0.55,-0.1], \
                [0.759,-0.65,-0.1],[0.799,-0.4,-0.1],[0.524,-0.51,-0.1],[0.629,-0.385,-0.1],[0.479,-0.305,-0.1]]                                   
        if int(args.target) > 9:
            raise ValueError('target object number must be between 0 and 9')                
        baxter_main(viewer=not args.noviewer, objects=Baxter2, target=args.target)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
