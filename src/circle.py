#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
# from geometry_msgs.msg import Transform
# import numpy as np
import sys, random
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.optimizers import Adam
import math

pub = rospy.Publisher('simple_create/cmd_vel', Twist, queue_size=10)
pub2 = rospy.Publisher('simple_create_2/cmd_vel', Twist, queue_size=10)
# pubTrans = rospy.Publisher('simple_create/cmd_vel' Transform, queue_size=10)
msg = Twist()
msg2 = Twist()
msg2.angular.z = 10

# msg.translation.x = 1000
# msg.translation.y = 0
# msg.translation.z = 0
# msg.linear.x = -1
# msg.angular.z =  3

def main():

    rospy.init_node('circler', anonymous=True)

    rate = rospy.Rate(2) # 10hz
    # msg.angular.z = 3

    while not rospy.is_shutdown():
        # msg.linear.x += 1
        pub.publish(msg)
        pub2.publish(msg2)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
