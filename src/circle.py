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

# Environment Setting
obsNumber = 1
state_size = obsNumber * 2
action_size = 9
num_episodes = 3601
boundaryRadius = 1
obstacleRadius = 0.2
agentRadius = 0.2
movingUnit = 0.02


# trainMode = True: Train Mode / trainMode = False: Test Mode
trainMode = True

# Train Mode
if trainMode:
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

            self.actor.fit(state, advantages, epochs=1, verbose=0)
            self.critic.fit(state, target, epochs=1, verbose=0)

    def stateGenerator(obsPosition, agtPosition):
        returnSum = []
        for i in range(0,obsNumber):
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
            xAction = movingUnit
            yAction = -movingUnit       
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
            yAction = movingUnit
        elif action == 8:
            xAction = 0
            yAction = 0
            
        return [xAction, yAction]

    def main():
        agent = A2CAgent(state_size, action_size)
        
        obsAngleIdx= 0
        circleFlag = False
        initRandom = random.randrange(0, 360)
        initPosMainRobot = [0, 0]

        # twistMainRobot_pub = rospy.Publisher('simple_create/cmd_vel', Twist, queue_size=10)
        # twistObstRobot_pub = rospy.Publisher('simple_create2/cmd_vel', Twist, queue_size=10)
        posMainRobot_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
        posObstRobot_pub = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)

        # twistMainRobot_msg = Twist()
        # twistObstRobot_msg = Twist()
        posMainRobot_msg = ModelState()
        posObstRobot_msg = ModelState()

        # twistMainRobot_msg.linear.x = 0
        # twistMainRobot_msg.linear.y = 0
        # twistMainRobot_msg.linear.z = 0
        # twistMainRobot_msg.angular.x = 0
        # twistMainRobot_msg.angular.y = 0
        # twistMainRobot_msg.angular.z = 0

        posMainRobot_msg.model_name = "simple_create"
        posObstRobot_msg.model_name = "simple_create2"
        posObstRobot_msg.pose.position.x = 1
        posObstRobot_msg.pose.position.y = 0
        posObstRobot_msg.pose.position.z = 0
        
        rospy.init_node('circler', anonymous=True)
        rate = rospy.Rate(50) #hz

        obsAngle = obsAngleIdx * math.pi/180
        posObstRobot_msg.pose.position.x = int(initPosMainRobot[0] + boundaryRadius * math.cos(obsAngle))
        posObstRobot_msg.pose.position.y = int(initPosMainRobot[1] + boundaryRadius * math.sin(obsAngle))

        for e in range(num_episodes):
            done = False
            score = 0
            reward = 0
            posMainRobot_msg.pose.position.x = initPosMainRobot[0]
            posMainRobot_msg.pose.position.y = initPosMainRobot[1]
            posMainRobot_msg.pose.position.z = 0
            rospy.logwarn("Episode %d Starts!", e)            
            state = stateGenerator([posObstRobot_msg.pose.position.x, posObstRobot_msg.pose.position.y], [posMainRobot_msg.pose.position.x, posMainRobot_msg.pose.position.y])
            while not done:
                action = agent.get_action(state)
                xMove = 0
                yMove = 0
                [xMove, yMove] = takeAction(action)

                posMainRobot_msg.pose.position.x += xMove
                posMainRobot_msg.pose.position.y += yMove

                collisionFlag = 0
                next_state = stateGenerator([posObstRobot_msg.pose.position.x, posObstRobot_msg.pose.position.y], [posMainRobot_msg.pose.position.x, posMainRobot_msg.pose.position.y])

                if(math.sqrt((posMainRobot_msg.pose.position.x - initPosMainRobot[0])**2 + (posMainRobot_msg.pose.position.y - initPosMainRobot[0])**2) >= boundaryRadius):
                    rospy.logwarn("No Collision!")
                    collisionFlag = 1
                    done = True
                if math.sqrt((posMainRobot_msg.pose.position.x - posObstRobot_msg.pose.position.x)**2 + (posMainRobot_msg.pose.position.y - posObstRobot_msg.pose.position.y)**2) <= obstacleRadius + agentRadius:
                    rospy.logwarn("Collision!")
                    collsiionFlag = -1
                    done = True

                if not done:
                    reward = 0
                else:
                    if collisionFlag == 1:
                        reward = -1000
                    elif collisionFlag == -1:
                        reward = 1000
                agent.train_model(state, action, reward, next_state, done)

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
                        obsAngleIdx += 1
                    else:
                        obsAngleIdx -= 1
                    obsAngle = (obsAngleIdx + initRandom)*math.pi/180
                    posObstRobot_msg.pose.position.x = initPosMainRobot[0] + boundaryRadius * math.cos(obsAngle)
                    posObstRobot_msg.pose.position.y = initPosMainRobot[1] + boundaryRadius * math.sin(obsAngle)
                posMainRobot_pub.publish(posMainRobot_msg)
                posObstRobot_pub.publish(posObstRobot_msg)
                rate.sleep()

            if e % 50 == 0:
                agent.actor.save_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Actor_Rev.h5")
                agent.critic.save_weights("/home/howoongjun/catkin_ws/src/simple_create/src/DataSave/Critic_Rev.h5")

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

    