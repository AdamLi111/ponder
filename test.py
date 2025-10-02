from PythonSDKmain.mistyPy.Robot import Robot

if __name__ == "__main__":
    ip_address = "172.20.10.4"
    # Create an instance of a robot
    misty = Robot(ip_address)
    misty.move_arms(-29, -29, 100, 100)