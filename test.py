import time
from PythonSDKmain.mistyPy.Robot import Robot

"""
resets head to 0, arms to 0"""
def reset(robot):
    robot.move_arms(0, 0)
    robot.move_head(0, 0, 0)

if __name__ == "__main__":
    ip_address = "172.20.10.4"
    # Create an instance of a robot
    misty = Robot(ip_address)
    reset(misty)
    time.sleep(1)
    misty.move_arms(-29, -29)
    misty.start_conversation("greetings")
    