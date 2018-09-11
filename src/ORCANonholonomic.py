#!/usr/bin/env python
import rospy
import numpy as np
import math
import rvo2
# from RVO_Py_MAS.RVO import RVO_update, reach, compute_V_des
# from RVO_Py_MAS.vis import visualize_traj_dynamic
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
import sys, random
import matplotlib.pyplot as plt
import pylab
from tf.transformations import euler_from_quaternion
import datetime
import time

# Environment Setting
num_episodes = 201
agentRadius = 0.17
obsNumber = 0
mainRobotNumber = 4
goalPos = [[5, 5], [0, 5], [0, 0], [5, 0], [5, 2.5], [2.5, 5], [0, 2.5], [2.5, 0], [0, 1.25], [0, 3.75], [5, 1.25], [5, 3.75]] 
moveObstacles = True

# mode = 0: Const, mode = 1: Linear, mode = 2: Quad
mode = 2

def takeAction(desiredVector, robotYaw):
    linearX = 0
    angularZ = 0
    angularVelocityCalibration = 5.0
    maxSpeed = math.sqrt(math.pow(desiredVector[0], 2) + math.pow(desiredVector[1], 2))
    desiredDegree = math.atan2(desiredVector[1], desiredVector[0])

    if desiredDegree == robotYaw:
        linearX = maxSpeed
        angularZ = 0
    else:
        angularDiff = robotYaw - desiredDegree
        if angularDiff > math.pi:
            angularDiff = angularDiff - math.pi * 2
        elif angularDiff < -math.pi:
            angularDiff = angularDiff + math.pi * 2
        delimeter = 2
        if mode == 1:
            delimeter = 2
            if angularDiff >= 0:
                linearX = -2 * maxSpeed / math.pi * angularDiff + maxSpeed
            elif angularDiff < 0:
                linearX = 2 * maxSpeed / math.pi * angularDiff + maxSpeed
        elif mode == 2:
            delimeter = math.sqrt(2)
            linearX = -2 * maxSpeed / (math.pi * math.pi) * (angularDiff * angularDiff) + maxSpeed
        
        if abs(angularDiff) == math.pi:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = 0
        elif abs(angularDiff) <= math.pi / delimeter:
            if mode == 0:
                linearX = maxSpeed
            angularZ = angularDiff * angularVelocityCalibration
        elif angularDiff > math.pi / delimeter:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = (angularDiff - math.pi) * angularVelocityCalibration
        elif angularDiff < -1 * math.pi / delimeter:
            if mode == 0:
                linearX = -maxSpeed
            angularZ = (angularDiff + math.pi) * angularVelocityCalibration
    return [linearX, angularZ]

def main():
    initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5], [5, 3.75], [5, 1.25], [0, 1.25], [0, 3.75]]
    initPosObstRobot = [[1.5, 1.5], [3.5, 3.5], [3.5, 1.5],  [1.5, 3.5], [4, 2], [3, 4], [1, 3], [2, 1]]
    allRobots = []

    for i in range(0, mainRobotNumber):
        allRobots = allRobots + [initPosMainRobot[i]]
    for i in range(0, obsNumber):
        allRobots = allRobots + [initPosObstRobot[i]]
    if mode == 0:
        tmpStr = "Const"
    elif mode == 1:
        tmpStr = "Linear"
    elif mode == 2:
        tmpStr = "Quad"
    rList = []
    f = open("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/log/log" + str(mainRobotNumber) + "robots" + str(obsNumber) + "obstacles_ORCA_" + tmpStr + "_" + datetime.datetime.now().strftime('%y%m%d') + ".txt", 'w')
    
    sim = rvo2.PyRVOSimulator(1/60., agentRadius * 2, 4, 1.5, 2, agentRadius, 1)
    a0 = sim.addAgent((initPosMainRobot[0][0], initPosMainRobot[0][1]))
    a1 = sim.addAgent((initPosMainRobot[1][0], initPosMainRobot[1][1]))
    a2 = sim.addAgent((initPosMainRobot[2][0], initPosMainRobot[2][1]))
    a3 = sim.addAgent((initPosMainRobot[3][0], initPosMainRobot[3][1]))

    rospy.init_node('circler', anonymous=True)
    rate = rospy.Rate(50) #hz

    posMainRobot_pub = []
    posMainRobot_msg = []
    posObstRobot_pub = []
    posObstRobot_msg = []

    twistMainRobot_pub = []
    twistMainRobot_msg = []
    twistObstRobot_pub = []
    twistObstRobot_msg = []

    rospy.logwarn("Loading !!!")

    for i in range(0, mainRobotNumber):
        posMainRobot_pub = posMainRobot_pub + [rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)]
        twistMainRobot_pub = twistMainRobot_pub + [rospy.Publisher('mainRobot' + str(i) + '/cmd_vel', Twist, queue_size=10)]
        posMainRobot_msg.append(ModelState())
        twistMainRobot_msg.append(Twist())
        posMainRobot_msg[i].model_name = "mainRobot" + str(i)

        [posMainRobot_msg[i].pose.position.x, posMainRobot_msg[i].pose.position.y] = initPosMainRobot[i]
        posMainRobot_msg[i].pose.position.z = 0
        posMainRobot_pub[i].publish(posMainRobot_msg[i])
        twistMainRobot_msg[i].linear.x = 0
        twistMainRobot_msg[i].linear.y = 0
        twistMainRobot_msg[i].linear.z = 0
        twistMainRobot_msg[i].angular.x = 0
        twistMainRobot_msg[i].angular.y = 0
        twistMainRobot_msg[i].angular.z = 0
    for i in range(0, obsNumber):
        posObstRobot_pub = posObstRobot_pub + [rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)]
        twistObstRobot_pub = twistObstRobot_pub + [rospy.Publisher('obsRobot' + str(i) + '/cmd_vel', Twist, queue_size=10)]
        posObstRobot_msg.append(ModelState())
        twistObstRobot_msg.append(Twist())
        
        posObstRobot_msg[i].model_name = "obsRobot" + str(i)
        [posObstRobot_msg[i].pose.position.x, posObstRobot_msg[i].pose.position.y] = initPosObstRobot[i]
        posObstRobot_msg[i].pose.position.z = 0
        posObstRobot_pub[i].publish(posObstRobot_msg[i])
        twistObstRobot_msg[i].linear.x = 0
        twistObstRobot_msg[i].linear.y = 0
        twistObstRobot_msg[i].linear.z = 0
        twistObstRobot_msg[i].angular.x = 0
        twistObstRobot_msg[i].angular.y = 0
        twistObstRobot_msg[i].angular.z = 0

    time.sleep(10)
    fps = 0
    elapsed = 0
    for e in range(num_episodes):
        start = time.time()
        done = False
        rospy.logwarn("Episode %d Starts!", e)
        rospy.logwarn(datetime.datetime.now().strftime('%H:%M:%S'))
        for i in range(0, mainRobotNumber):
            [posMainRobot_msg[i].pose.position.x, posMainRobot_msg[i].pose.position.y] = initPosMainRobot[i]
            posMainRobot_msg[i].pose.position.z = 0
            posMainRobot_msg[0].pose.orientation.z = 0
            posMainRobot_msg[1].pose.orientation.z = 1
            posMainRobot_msg[2].pose.orientation.z = 1
            posMainRobot_msg[3].pose.orientation.z = 0
            posMainRobot_pub[i].publish(posMainRobot_msg[i])
            twistMainRobot_msg[i].linear.x = 0
            twistMainRobot_msg[i].angular.z = 0
            twistMainRobot_pub[i].publish(twistMainRobot_msg[i])
        for i in range(0, obsNumber):
            [posObstRobot_msg[i].pose.position.x, posObstRobot_msg[i].pose.position.y] = initPosObstRobot[i]
            posObstRobot_msg[i].pose.position.z = 0
            posObstRobot_msg[0].pose.orientation.z = 1
            posObstRobot_msg[1].pose.orientation.z = 0
            # posObstRobot_msg[2].pose.orientation.z = 0
            # posObstRobot_msg[3].pose.orientation.z = 1
            posObstRobot_pub[i].publish(posObstRobot_msg[i])
        # Initialize goalReached flag
        goalReached = []


        for i in range(0, mainRobotNumber):
            goalReached = goalReached + [False]

        frame = 0
        ckTime = 0
        while not done:
            start = time.time()
            frame += 1
            object_coordinates = []
            obst_coordinates = []
            prefVel = []
            
            # Move obstacles
            for obsRobNo in range(0, obsNumber):
                twistObstRobot_msg[obsRobNo].linear.x = 0#random.randrange(0, 2)#linearX
                twistObstRobot_msg[obsRobNo].angular.z = 0#random.randrange(-4, 5)#angularZ
                twistObstRobot_pub[obsRobNo].publish(twistObstRobot_msg[obsRobNo])

            for i in range(0, mainRobotNumber):
                model_coordinates = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
                object_coordinates = object_coordinates + [model_coordinates("mainRobot" + str(i), "")]
                
            for i in range(0, obsNumber):
                model_coordinates = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
                obst_coordinates = obst_coordinates + [model_coordinates("obsRobot" + str(i), "")]
            
            # Move Main robots
            for curRobNo in range(0, mainRobotNumber):
                quaternion = (object_coordinates[curRobNo].pose.orientation.x, object_coordinates[curRobNo].pose.orientation.y, object_coordinates[curRobNo].pose.orientation.z, object_coordinates[curRobNo].pose.orientation.w)
                euler = euler_from_quaternion(quaternion)
                yaw = euler[2]
                linearX = 0
                angularZ = 0
                sim.setAgentPosition(curRobNo, (object_coordinates[curRobNo].pose.orientation.x, object_coordinates[curRobNo].pose.orientation.y))
                tmpPref = [goalPos[curRobNo][0] - object_coordinates[curRobNo].pose.orientation.x, goalPos[curRobNo][1] - object_coordinates[curRobNo].pose.orientation.y]
                # rospy.logwarn(sim.getAgentPosition(curRobNo))
                prefVel = [tmpPref[0] / math.sqrt(tmpPref[0]**2 + tmpPref[1]**2), tmpPref[1] / math.sqrt(tmpPref[0]**2 + tmpPref[1]**2)]
                sim.setAgentPrefVelocity(curRobNo, (prefVel[0], prefVel[1]))
                rospy.logwarn(prefVel)
                rospy.logwarn(sim.getAgentVelocity(curRobNo))
                rospy.logwarn("\n")
                sim.doStep()
                [linearX, angularZ] = takeAction(sim.getAgentVelocity(curRobNo), yaw)
                twistMainRobot_msg[curRobNo].linear.x = linearX
                twistMainRobot_msg[curRobNo].angular.z = angularZ # * 0.5
                twistMainRobot_pub[curRobNo].publish(twistMainRobot_msg[curRobNo])

                collisionFlag = 0

                if(math.sqrt((object_coordinates[curRobNo].pose.position.x - goalPos[curRobNo][0])**2 + (object_coordinates[curRobNo].pose.position.y - goalPos[curRobNo][1])**2) <= 2 * agentRadius):
                    if goalReached[curRobNo] == False:
                        rospy.logwarn(str(curRobNo) + " Robot has reached to the goal!")
                    goalReached[curRobNo] = True
                for i in range(0, mainRobotNumber):
                    if i != curRobNo:
                        if math.sqrt((object_coordinates[curRobNo].pose.position.x - object_coordinates[i].pose.position.x)**2 + (object_coordinates[curRobNo].pose.position.y - object_coordinates[i].pose.position.y)**2) < 2*agentRadius:
                            rospy.logerr("Collision with a main robot")
                            collisionFlag = -1
                            done = True
                for i in range(0, obsNumber):
                    if math.sqrt((obst_coordinates[i].pose.position.x - object_coordinates[curRobNo].pose.position.x)**2 + (obst_coordinates[i].pose.position.y - object_coordinates[curRobNo].pose.position.y)**2) < 2 * agentRadius:
                        rospy.logerr("Collision with an Obstacle")
                        collisionFlag = -1
                        done = True
            rospy.logwarn("================================")

            tmpCount = 1
            for i in range(0, mainRobotNumber):
                tmpCount = tmpCount * goalReached[i]
            if tmpCount == 1:
                done = True
                rList.append(1)

            if done:
                if collisionFlag == -1:
                    rList.append(0)
                initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5], [5, 3.75], [5, 1.25], [0, 1.25], [0, 3.75]]
                for i in range(0, obsNumber):
                    [posObstRobot_msg[i].pose.position.x, posObstRobot_msg[i].pose.position.y] = initPosObstRobot[i]
                    posObstRobot_msg[i].pose.position.z = 0
                    posObstRobot_pub[i].publish(posObstRobot_msg[i])
            rate.sleep()
            final = time.time()
            ckTime = (ckTime * (frame - 1) + final - start) / frame
            
        # fps = (fps * e + frame / (final - start)) / (e + 1)
        if e != 0:
            if collisionFlag != -1 and sum(rList) != 0:
                elapsed = (elapsed * (sum(rList) - 1) + final - start) / sum(rList)
            rospy.logwarn("Percent of successful episodes: %f %%", 100.0 * sum(rList)/(e + 1))
        rospy.logwarn("Average Processing Time: %f", ckTime)
        data = "Episode_%d_%f \n" % (e, ckTime)
        f.write(data)
        # rospy.logwarn("Elapsed Time: %f s", final - start)
        # rospy.logwarn("Frame per Second: %f fps", frame / (final - start))
        # rospy.logwarn("Average Time: %f s", elapsed)
        # rospy.logwarn("Average FPS: %f fps", fps)
        rospy.logwarn("=====================================================================")
    f.close()
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass