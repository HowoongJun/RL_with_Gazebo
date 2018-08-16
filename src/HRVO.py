#!/usr/bin/env python
import rospy
import numpy as np
import math
from RVO_Py_MAS.RVO import RVO_update, reach, compute_V_des
from RVO_Py_MAS.vis import visualize_traj_dynamic
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
goalPos = [[5, 5], [0, 5], [0, 0], [5, 0], [5, 2.5], [2.5, 5], [0, 2.5], [2.5, 0]] 
moveObstacles = True

ws_model = dict()
ws_model['robot_radius'] = agentRadius
ws_model['circular_obstacles'] = []
ws_model['boundary'] = []

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
        # if angularDiff > 0:
        #     linearX = -2 * maxSpeed / math.pi * angularDiff + maxSpeed
        # else:
        #     linearX = 2 * maxSpeed / math.pi * angularDiff + maxSpeed
        linearX = -2 * maxSpeed / (math.pi * math.pi) * (angularDiff * angularDiff) + maxSpeed
        if abs(angularDiff) == math.pi:
            angularZ = 0
        elif abs(angularDiff) <= math.pi / math.sqrt(2):
            angularZ = angularDiff * angularVelocityCalibration
        elif angularDiff > math.pi / math.sqrt(2):
            angularZ = (angularDiff - math.pi) * angularVelocityCalibration
        elif angularDiff < -1 * math.pi / math.sqrt(2):
            angularZ = (angularDiff + math.pi) * angularVelocityCalibration

    return [linearX, angularZ]

def main():
    initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5]]
    rList = []

    rospy.init_node('circler', anonymous=True)
    rate = rospy.Rate(50) #hz

    posMainRobot_pub = []
    posMainRobot_msg = []

    twistMainRobot_pub = []
    twistMainRobot_msg = []

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

    time.sleep(5)
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
            posMainRobot_pub[i].publish(posMainRobot_msg[i])

        # Initialize goalReached flag
        goalReached = []
        for i in range(0, mainRobotNumber):
            goalReached = goalReached + [False]
        V = [[0, 0] for i in xrange(len(initPosMainRobot))]
        V_max = [2.0 for i in xrange(len(initPosMainRobot))]
        frame = 0
        while not done:
            frame += 1
            object_coordinates = []
            X = []
            for i in range(0, mainRobotNumber):
                model_coordinates = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
                object_coordinates = object_coordinates + [model_coordinates("mainRobot" + str(i), "")]
                X = X + [[object_coordinates[i].pose.position.x, object_coordinates[i].pose.position.y]]
            V_des = compute_V_des(X, goalPos, V_max)
            V = RVO_update(X, V_des, V, ws_model)
            for curRobNo in range(0, mainRobotNumber):
                quaternion = (object_coordinates[curRobNo].pose.orientation.x, object_coordinates[curRobNo].pose.orientation.y, object_coordinates[curRobNo].pose.orientation.z, object_coordinates[curRobNo].pose.orientation.w)
                euler = euler_from_quaternion(quaternion)
                yaw = euler[2]
                # rospy.logwarn(str(curRobNo) + ":" + str(V[curRobNo]))
                linearX = 0
                angularZ = 0
                
                [linearX, angularZ] = takeAction(V[curRobNo], yaw)
                # rospy.logwarn(str(curRobNo) + ": " + str(action))
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
                            rospy.logerr("Collision !")
                            collisionFlag = -1
                            done = True
            tmpCount = 1
            for i in range(0, mainRobotNumber):
                tmpCount = tmpCount * goalReached[i]
            if tmpCount == 1:
                done = True
                rList.append(1)

            if done:
                if collisionFlag == -1:
                    rList.append(0)
                initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5]]
            rate.sleep()
        final = time.time()
        fps = (fps * e + frame / (final - start)) / (e + 1)
        if collisionFlag != -1:
            elapsed = (elapsed * (sum(rList) - 1) + final - start) / sum(rList)
        rospy.logwarn("Percent of successful episodes: %f %%", 100.0 * sum(rList)/(e + 1))
        rospy.logwarn("Elapsed Time: %f s", final - start)
        rospy.logwarn("Frame per Second: %f fps", frame / (final - start))
        rospy.logwarn("Average Time: %f s", elapsed)
        rospy.logwarn("Average FPS: %f fps", fps)
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass