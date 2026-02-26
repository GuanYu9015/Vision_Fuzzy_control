#!/usr/bin/python3
"""
鍵盤遙控節點
使用 WASD 控制機器人移動，按 's' 停止
"""
from __future__ import print_function

import roslib
roslib.load_manifest('teleop_twist_keyboard')
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import sys
import select
import termios
import tty

# 全域速度變數
RoV = 0.0  # 線速度
RoW = 0.0  # 角速度

msg = """
遙控鍵盤控制
---------------------------
移動控制:
        w    
   a    s    d
        x    
   
w/x : 增加/減少線速度
a/d : 增加/減少角速度
s/空格 : 停止

CTRL-C 離開
"""

# 鍵盤按鍵與機器人運動指令的映射
moveBindings = {
    'w': (1, 0, 0, 0),
    'x': (-1, 0, 0, 0),
    'a': (0, 0, 0, 1),
    'd': (0, 0, 0, -1),
    's': (0, 0, 0, 0),
    ' ': (0, 0, 0, 0),
}


def getKey():
    """讀取一個鍵盤按鍵的輸入並返回"""
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, turn):
    """返回當前速度和旋轉速度的字符串表示"""
    return f"速度: {speed:.2f} m/s | 角速度: {turn:.2f} rad/s"


if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    avoidance_pub = rospy.Publisher('/avoidance_enabled', String, queue_size=10)
    rospy.init_node('teleop_twist_keyboard')

    speed = rospy.get_param("~speed", 0.5)
    turn = rospy.get_param("~turn", 1.0)
    avoidance_enabled = False

    try:
        print(msg)
        print(vels(speed, turn))
        
        while not rospy.is_shutdown():
            key = getKey()
            
            if key in moveBindings.keys():
                if key == 'w':
                    if RoV >= 0.9:
                        RoV = 0.9
                    else:
                        RoV = RoV + 0.05
                elif key == 'x':
                    if RoV <= -0.9:
                        RoV = -0.9
                    else:
                        RoV = RoV - 0.05
                elif key == 'd':
                    if RoW <= -0.64:
                        RoW = -0.64
                    else:
                        RoW = RoW - 0.08
                elif key == 'a':
                    if RoW >= 0.64:
                        RoW = 0.64
                    else:
                        RoW = RoW + 0.08
                elif key == 's' or key == ' ':
                    RoW = 0.0
                    RoV = 0.0
            elif key == '\x03':  # CTRL-C
                break
            else:
                # 其他按鍵不改變速度
                pass

            # 創建並發布 Twist 消息
            twist = Twist()
            twist.linear.x = RoV
            twist.angular.z = RoW
            speed = RoV
            turn = RoW
            print(vels(speed, turn))
            pub.publish(twist)

    except Exception as e:
        print(e)

    finally:
        # 停止機器人
        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        pub.publish(twist)

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)