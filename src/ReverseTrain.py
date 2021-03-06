#!/usr/bin/env python
import rospy
# from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
import numpy as np
import sys, random
import matplotlib.pyplot as plt
import pylab
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import math
import datetime
import time

# Environment Setting
obsNumber = 1
state_size = obsNumber * 2
action_size = 8
num_episodes = 1801
boundaryRadius = 0.85
obstacleRadius = 0.2
agentRadius = 0.17
movingUnit = 0.017

# A2C(Advantage Actor-Critic) agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.load_model = False
        
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
        if self.load_model:
            self.actor.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Actor_Rev.h5")
            self.critic.load_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Critic_Rev.h5")
        self.actor.get_weights()
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
        return np.random.choice(self.action_size, 1, p=policy)[0]

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
        ActorHistory = self.actor.fit(state, advantages, epochs=1, verbose=0)
        CriticHistory = self.critic.fit(state, target, epochs=1, verbose=0)
        return [ActorHistory, CriticHistory]

def stateGenerator(obsPosition, agtPosition):
    returnSum = []
    returnSum = returnSum + [agtPosition[0] - obsPosition[0], agtPosition[1] - obsPosition[1]]
    returnSum = np.reshape(returnSum, [1, state_size])
    return returnSum

def takeAction(action):
    xAction = 0
    yAction = 0
    if action == 0:
        xAction = movingUnit
    elif action == 1:
        xAction = movingUnit
        yAction = movingUnit
    elif action == 2:
        yAction = movingUnit
    elif action == 3:
        xAction = -movingUnit
        yAction = movingUnit
    elif action == 4:
        xAction = -movingUnit
    elif action == 5:
        xAction = -movingUnit
        yAction = -movingUnit
    elif action == 6:
        yAction = -movingUnit
    elif action == 7:
        xAction = movingUnit
        yAction = -movingUnit
    # elif action == 8:
    #     xAction = 0
    #     yAction = 0
        
    return [xAction, yAction]

def main():
    start = time.time()
    agent = A2CAgent(state_size, action_size)

    episodeNo = []
    scorePlot = []
    # ActLossPlot = []
    # CritLossPlot = []
    # iteration = 0

    obsAngleIdx= 0
    circleFlag = False
    initRandom = random.randrange(0, 360)
    initPosMainRobot = [0, 0]

    posMainRobot_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
    posObstRobot_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)

    posMainRobot_msg = ModelState()
    posObstRobot_msg = ModelState()

    posMainRobot_msg.model_name = "simple_create"
    posObstRobot_msg.model_name = "simple_create2"
    posObstRobot_msg.pose.position.x = 1
    posObstRobot_msg.pose.position.y = 0
    posObstRobot_msg.pose.position.z = 0
    
    rospy.init_node('circler', anonymous=True)
    rate = rospy.Rate(1000) #hz

    obsAngle = obsAngleIdx * math.pi/180
    posObstRobot_msg.pose.position.x = initPosMainRobot[0] + boundaryRadius * math.cos(obsAngle)
    posObstRobot_msg.pose.position.y = initPosMainRobot[1] + boundaryRadius * math.sin(obsAngle)


    for e in range(num_episodes):
        done = False
        score = 0
        reward = 0
        total_loss_act = 0
        total_loss_crit = 0

        posMainRobot_msg.pose.position.x = initPosMainRobot[0]
        posMainRobot_msg.pose.position.y = initPosMainRobot[1]
        posMainRobot_msg.pose.position.z = 0
        rospy.logwarn("Episode %d Starts!", e)
        rospy.logwarn(datetime.datetime.now().strftime('%H:%M:%S'))
        
        state = stateGenerator([posObstRobot_msg.pose.position.x, posObstRobot_msg.pose.position.y], [posMainRobot_msg.pose.position.x, posMainRobot_msg.pose.position.y])
        while not done:
            # iteration += 1
            action = agent.get_action(state)
            xMove = 0
            yMove = 0
            [xMove, yMove] = takeAction(action)

            posMainRobot_msg.pose.position.x += xMove
            posMainRobot_msg.pose.position.y += yMove

            collisionFlag = 0
            next_state = stateGenerator([posObstRobot_msg.pose.position.x, posObstRobot_msg.pose.position.y], [posMainRobot_msg.pose.position.x, posMainRobot_msg.pose.position.y])

            if(math.sqrt((posMainRobot_msg.pose.position.x - initPosMainRobot[0])**2 + (posMainRobot_msg.pose.position.y - initPosMainRobot[1])**2) >= boundaryRadius):
                rospy.logerr("No Collision!")
                collisionFlag = 1
                done = True
            if math.sqrt((posMainRobot_msg.pose.position.x - posObstRobot_msg.pose.position.x)**2 + (posMainRobot_msg.pose.position.y - posObstRobot_msg.pose.position.y)**2) <= obstacleRadius + agentRadius:
                rospy.logwarn("Collision!")
                collisionFlag = -1
                done = True

            if not done:
                reward = -0.1
            else:
                if collisionFlag == 1:
                    reward = -1000
                elif collisionFlag == -1:
                    reward = 1000
            [ActorHistory, CriticHistory] = agent.train_model(state, action, reward, next_state, done)
            total_loss_act += ActorHistory.history['loss'][0]
            total_loss_crit += CriticHistory.history['loss'][0]

            score += reward
            state = next_state

            if done:
                if obsAngleIdx >= 360:
                    circleFlag = True
                    initRandom = random.randrange(0, 360)
                elif obsAngleIdx < 0:
                    circleFlag = False
                    initRandom = random.randrange(0, 360)
                if circleFlag == False:
                    obsAngleIdx += 2
                else:
                    obsAngleIdx -= 2
                obsAngle = (obsAngleIdx + initRandom)*math.pi/180
                posObstRobot_msg.pose.position.x = initPosMainRobot[0] + boundaryRadius * math.cos(obsAngle)
                posObstRobot_msg.pose.position.y = initPosMainRobot[1] + boundaryRadius * math.sin(obsAngle)
                episodeNo = episodeNo + [e]
                final = time.time()
                elapsed = final - start
                rospy.logwarn("Elapsed Time: %s s", elapsed)

            # iterationNo = 1000
            # if iteration % iterationNo == 0:
            #     episodeNo = episodeNo + [iteration / iterationNo]
            #     ActLossPlot = ActLossPlot + [total_loss_act / iterationNo]
            #     CritLossPlot = CritLossPlot + [total_loss_crit / iterationNo]
            #     # rospy.logwarn(total_loss)
            #     # rospy.logwarn("iteration: %s", iteration)
            #     plt.plot(episodeNo, ActLossPlot, linewidth = 0.3)
            #     # plt.savefig("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/ActorLossPlot.png", dpi = 300)
            #     plt.plot(episodeNo, CritLossPlot, linewidth = 0.3)
            #     plt.savefig("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/LossPlot_All.png", dpi = 300)
 
            #     total_loss_act = 0
            #     total_loss_crit = 0
                scorePlot = scorePlot + [score]
            posMainRobot_pub.publish(posMainRobot_msg)
            posObstRobot_pub.publish(posObstRobot_msg)
            rate.sleep()

        if e % 60 == 0:
            agent.actor.save_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Actor_Rev.h5")
            agent.critic.save_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Critic_Rev.h5")
            plt.plot(episodeNo, scorePlot, linewidth=0.3)
            plt.savefig("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/RewardPlot.png", dpi=300)
        rospy.logwarn("Reward: %d", score)
        
    # while not rospy.is_shutdown():
    #     posMainRobot_msg.pose.position.x += 0.01
    #     posMainRobot_msg.pose.position.y += 0.01
    #     # twistMainRobot_pub.publish(twistMainRobot_msg)
    #     # twistObstRobot_pub.publish(twistObstRobot_msg)
    #     posMainRobot_pub.publish(posMainRobot_msg)
    #     posObstRobot_pub.publish(posObstRobot_msg)
    #     rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
