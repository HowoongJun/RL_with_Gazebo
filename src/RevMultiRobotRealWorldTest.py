#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from epuck.ePuck import ePuck
import numpy as np
import sys, random
import matplotlib.pyplot as plt
import pylab
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.optimizers import Adam
# from tf.transformations import euler_from_quaternion
import math
import datetime
import time
import socket

TCP_IP = '147.46.67.123'
TCP_PORT = 6666
BUFFER_SIZE = 1024
WHEEL_DIAMETER = 4
WHEEL_SEPARATION = 5.3
WHEEL_DISTANCE = 0.053
WHEEL_CIRCUMFERENCE = ((WHEEL_DIAMETER * math.pi) / 100.0)
MOT_STEP_DIST = (WHEEL_CIRCUMFERENCE / 1000.0)

# Environment Setting
num_episodes = 201
obstacleRadius = 0.18
agentRadius = 0.18
obsNumber = 6
mainRobotNumber = 4
state_size = 2
action_size = 9
boundaryRadius = 0.85
goalPos = [[5, 5], [0, 5], [0, 0], [5, 0], [5, 2.5], [2.5, 5], [0, 2.5], [2.5, 0], [0, 1.25], [0, 3.75], [5, 1.25], [5, 3.75]] 
moveObstacles = True

# mode = 0: Const, mode = 1: Linear, mode = 2: Quad
mode = 2

# A2C(Advantage Actor-Critic) agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.load_model1 = True
        # self.load_model2 = True
        
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 0.00002
        self.critic_lr = 0.00005

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model1:
            # self.actor.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Actor_Rev.h5")
            # self.critic.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Critic_Rev.h5")
            self.actor.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/backup/Actor_Rev_180112.h5")
            self.critic.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/backup/Critic_Rev_180112.h5")

        # if self.load_model2:
            # self.actor.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Actor_Macro.h5")
            # self.critic.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Critic_Macro.h5")
            
    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        actor.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_normal'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        critic.add(Dense(self.value_size, activation='linear', kernel_initializer='glorot_normal'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return policy

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

def stateGenerator(obsPosition, agtPosition, idx):
    returnSum = []
    if idx != -1:
        returnSum = returnSum + [agtPosition[0] - obsPosition[idx].pose.position.x, agtPosition[1] - obsPosition[idx].pose.position.y]
    else:
        returnSum = returnSum + [agtPosition[0] - obsPosition[0], agtPosition[1] - obsPosition[1]]
    returnSum = np.reshape(returnSum, [1, state_size])
    return returnSum

def rangeFinder(allObsPos, rangeCenter):
    countObs = 0
    rangeObstacle = [[0,0] for _ in range(obsNumber + mainRobotNumber - 1)]
    allObsAgtDistance = [0 for _ in range(obsNumber + mainRobotNumber - 1)]
    for i in range(0, obsNumber + mainRobotNumber - 1):
        allObsAgtDistance[i] = math.sqrt((allObsPos[i].pose.position.x - rangeCenter[0])**2 + (allObsPos[i].pose.position.y - rangeCenter[1])**2)
        if math.sqrt((rangeCenter[0] - allObsPos[i].pose.position.x)**2 + (rangeCenter[1] - allObsPos[i].pose.position.y)**2) < boundaryRadius:
            rangeObstacle[countObs] = allObsPos[i]
            countObs += 1

    index = np.argmin(allObsAgtDistance)
    return [countObs, rangeObstacle, index]

def goalFinder(idx, agtPos):
    goalAngle = 0
    if goalPos[idx][0] == agtPos[0]:
        if goalPos[idx][1] > agtPos[1]:
            goalAngle = 90 * math.pi / 180
        else:
            goalAngle = -90 * math.pi / 180
    else:
        goalAngle = math.atan(1.0*(goalPos[idx][1]-agtPos[1])/(goalPos[idx][0]-agtPos[0]))
    if goalPos[idx][0] < agtPos[0]:
        goalAngle += math.pi
        
    tmpGoal = [0,0]
    tmpGoal[0] = agtPos[0] + boundaryRadius * math.cos(goalAngle)
    tmpGoal[1] = agtPos[1] + boundaryRadius * math.sin(goalAngle)
    return tmpGoal

def action2degree(action):
    degree = 0
    if action == 0:
        degree = 0
    elif action == 1:
        degree = math.pi / 4
    elif action == 2:
        degree = math.pi / 2
    elif action == 3:
        degree = math.pi * 3 / 4
    elif action == 4:
        degree = math.pi
    elif action == 5:
        degree  = -3 * math.pi / 4
    elif action == 6:
        degree = -1 * math.pi / 2
    elif action == 7:
        degree = -1 * math.pi / 4

    return degree

def takeAction(desiredHeading, robotYaw):
    linearX = 0
    angularZ = 0
    angularVelocityCalibration = 5.0
    maxSpeed = 2.0
    if desiredHeading == 2:
        desiredHeading = 7
    elif desiredHeading == 7:
        desiredHeading = 2

    desiredDegree = action2degree(desiredHeading)

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
        if desiredHeading == 8:
            linearX = 0
            angularZ = 0
    return [linearX, angularZ]

def main():
    agent = A2CAgent(state_size, action_size)
    
    rospy.on_shutdown(self.disconnect)
    initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5], [5, 3.75], [5, 1.25], [0, 1.25], [0, 3.75]]
    initPosObstRobot = [[1.5, 1.5], [3.5, 3.5], [3.5, 1.5],  [1.5, 3.5], [0, 2.5], [5, 2.5], [2.5, 0], [2.5, 5]]
    rList = []
    rospy.logwarn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    rate = rospy.Rate(50) #hz
    tmpStr = ""
    if mode == 0:
        tmpStr = "Const"
    elif mode == 1:
        tmpStr = "Linear"
    elif mode == 2:
        tmpStr = "Quad"
    f = open("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/log/log" + str(mainRobotNumber) + "robots" + str(obsNumber) + "obstacles_RRL_RealWorld_" + tmpStr + "_" + datetime.datetime.now().strftime('%y%m%d') + ".txt", 'w')

    posMainRobot_pub = []
    posMainRobot_msg = []

    for i in range(0, mainRobotNumber):
        posMainRobot_pub = posMainRobot_pub + [rospy.Publisher('/epuck_robot_' + str(i) + '/mobile_base/cmd_vel', Twist, queue_size = 10)]
        posMainRobot_msg.append(Twist())
        posMainRobot_pub[i].publish(posMainRobot_msg[i])
    # AvgTime = 0
    for e in range(num_episodes):
        done = False
        frame = 0
        ckTime = 0
        elapsed = 0
        sendMainAngularZ = 0
        sendMainLinearX = 0
        sendObstacleLinearX = 0
        sendObstacleAngularZ = 0

        rospy.logwarn("Episode %d Starts!", e)
        rospy.logwarn(datetime.datetime.now().strftime('%H:%M:%S'))

        epStart = time.time()
        # Initialize goalReached flag
        goalReached = []
        for i in range(0, mainRobotNumber):
            goalReached = goalReached + [False]

        while not done:
            start = time.time()
            frame += 1
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((TCP_IP, TCP_PORT))

            data = s.recv(BUFFER_SIZE)
            s.close()
            
            # Main Robot Coordinates
            mainRobot_coordinates = []
            mainRobotYaw = []

            dataList = data.split(" ")
            for i in range(0, mainRobotNumber):
                mainRobot_coordinates = [] + [[dataList[4 * i], dataList[4 * i + 1]]]
                mainRobotYaw = [] + [[dataList[4 * i + 2]]]

            # Obstacle Robot Coordinates
            interferingRobot_coordinates = []
            obst_coordinates = []
            
            for curRobNo in range(0, mainRobotNumber):
                tmpAction = []

                # Receive Interfering Robot Data
                obst_coordinates = interferingRobot_coordinates

                # Receive Main Robot Data
                # mainRobot_coordinates = mainRobot_coordinates + []
                
                for i in range(0, mainRobotNumber):
                    if i != curRobNo:
                        obst_coordinates = obst_coordinates + [mainRobot_coordinates[i]]

                initPosMainRobot[curRobNo] = mainRobot_coordinates[curRobNo]
                [rangeObsNumber, rangeObsPos, _] = rangeFinder(obst_coordinates, initPosMainRobot[curRobNo])
                for i in range(0, rangeObsNumber):
                    state = stateGenerator(rangeObsPos, mainRobot_coordinates[curRobNo], i)
                    policyArr = agent.get_action(state)
                    if i == 0:
                        tmpAction = (1 - policyArr)
                    else:
                        tmpAction = tmpAction * (1 - policyArr)
                # rospy.logwarn(tmpAction)
                if tmpAction != []:
                    for j in range(0,action_size):
                        if tmpAction[j] > 0.999:
                            tmpAction[j] = 1
                        elif tmpAction[j] > 0.995:
                            tmpAction[j] = 0.1
                        else:
                            tmpAction[j] = 0
                    tmpArgMax = np.argmax(tmpAction)
                # rospy.logwarn(tmpAction)
                if rangeObsNumber == 0:
                    tmpAction = [1.0/action_size for _ in range(0,action_size)]
                tmpGoalPos = goalFinder(curRobNo, mainRobot_coordinates[curRobNo])

                state = stateGenerator(tmpGoalPos, mainRobot_coordinates[curRobNo], -1)
                policyArr = agent.get_action(state)
                # rospy.logwarn("Goal: %s", policyArr)
                if np.mean(tmpAction) == 0:
                    rospy.logwarn("No Action Selected! Random Action")
                    # tmpAction[random.randrange(0, action_size)] = 1
                    tmpAction[tmpArgMax] = 1

                tmpAction = tmpAction * np.asarray(policyArr)

                tmpAction = tmpAction / np.sum(tmpAction)
                action = np.random.choice(action_size, 1, p = tmpAction)[0]

                linearX = 0
                angularZ = 0
                [linearX, angularZ] = takeAction(action, mainRobotYaw[curRobNo])

                sendMainLinearX = linearX
                sendMainAngularZ = angularZ

                collisionFlag = 0

                if(math.sqrt((mainRobot_coordinates[curRobNo][0] - goalPos[curRobNo][0])**2 + (mainRobot_coordinates[curRobNo][1] - goalPos[curRobNo][1])**2) <= 2 * agentRadius):
                    if goalReached[curRobNo] == False:
                        rospy.logwarn(str(curRobNo) + " Robot has reached to the goal!")
                    goalReached[curRobNo] = True
                for i in range(0, obsNumber + mainRobotNumber - 1):
                    if i < obsNumber:
                        if moveObstacles:
                            sendObstacleLinearX = random.randrange(0, 2)
                            sendObstacleAngularZ = random.randrange(-4, 5)

                    if math.sqrt((mainRobot_coordinates[curRobNo][0] - obst_coordinates[i][0])**2 + (mainRobot_coordinates[curRobNo][1] - obst_coordinates[i][1])**2) < obstacleRadius + agentRadius:
                        if i < obsNumber:
                            rospy.logerr("Collision with an obstacle!")
                        else:
                            rospy.logerr("Collision with a main robot!")
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

                initPosMainRobot = [[0, 0], [5, 0], [5, 5], [0, 5], [0, 2.5], [2.5, 0], [5, 2.5], [2.5, 5], [5, 3.75], [5, 1.25], [0, 1.25], [0, 3.75]]

            final = time.time()
            ckTime = (ckTime * (frame - 1) + final - start) / frame

        # fps = (fps * e + frame / (final - start)) / (e + 1)
        epTerminate = time.time()
        if e != 0:
            if collisionFlag != -1 and sum(rList) != 0:
                elapsed = (elapsed * (sum(rList) - 1) + (epTerminate - epStart)) / sum(rList)
        rospy.logwarn("Percent of successful episodes: %f %%", 100.0 * sum(rList)/(e + 1))
        rospy.logwarn("Average Processing Time: %f s", ckTime)
        data = "Episode_%d_%f_avgTime_%f_%d \n" % (e, ckTime, epTerminate - epStart, collisionFlag)
        f.write(data)
        rospy.logwarn("Elapsed Time: %f s", epTerminate - epStart)
        rospy.logwarn("Data Saved")
        AvgTime = (AvgTime * e + epTerminate - epStart) / (e + 1)
        rospy.logwarn("Average Elapsed Time: %f s", AvgTime)
        # rospy.logwarn("Frame per Second: %f fps", frame / (final - start))
        rospy.logwarn("Average Time: %f s", elapsed)
        # rospy.logwarn("Average FPS: %f fps", fps)
        rospy.logwarn("=====================================================================")
    f.write(str(100.0*sum(rList)/(e+1)) + "%")
    f.close()
if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass