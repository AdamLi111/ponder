from PythonSDKmain.mistyPy.Robot import Robot
import json
import requests
import time
import os

misty = Robot("172.20.10.2")

# Modify the name for differnet output
saved_name = "test"

# move head to face forward
misty.move_head(0, 0, 0)
time.sleep(2)

# Take a photo and save it on Misty
result = misty.take_picture(base64=True, fileName="myportrait", width=3200,height= 2400,displayOnScreen=False,overwriteExisting=True)
print(f"Photo saved as: {result}")

# Extract the filename from the response
try:
    # Parse the response to get the actual data
    if hasattr(result, 'json'):
        response_data = result.json()
    elif hasattr(result, 'text'):
        response_data = json.loads(result.text)
    else:
        response_data = result
    
    #print(f"Response data: {response_data}")
    
    # The filename is typically in result['result']['name']
    if 'result' in response_data:
        filename = response_data['result']['name']
    else:
        # Sometimes it might be directly in the response
        filename = response_data.get('name')
    
    print(f"Image filename: {filename}")
    
    # create output directory
    os.makedirs('images', exist_ok=True)

    # Download the image
    image_url = f"http://172.20.10.2/api/images?FileName={filename}"
    print(f"Downloading from: {image_url}")
    
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(f'images/{saved_name}.jpg', 'wb') as f:
            f.write(response.content)
        print(f'Photo downloaded successfully to images/{saved_name}.jpg!')
    else:
        print(f"Failed to download image: {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()