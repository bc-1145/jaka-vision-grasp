# -*- coding: utf-8 -*-
import time
import jkrc

# 运动模式
ABS  = 0  # 绝对TCP位置
INCR = 1  # 增量位置

# 目标TCP绝对坐标 [X,Y,Z,Rx,Ry,Rz]，姿态单位弧度
tcp_pos = [300, -100, 160, 3.14, 0, -3.14]

# 创建机器人对象，建立连接
robot = jkrc.RC("192.168.3.113")

ret = robot.login()
print("登录返回：", ret[0])

ret = robot.power_on()
print("上电返回：", ret[0])

ret = robot.enable_robot()
print("使能返回：", ret[0])


# linear_move(目标位姿, ABS绝对模式, 阻塞, 速度mm/s)
ret = robot.linear_move(tcp_pos, ABS, True, 10)
print("运动返回码：", ret[0])

if ret[0] == 0:
    print("运动完成，到达目标点位")
else:
    print(f"运动失败，错误码: {ret[0]}")

