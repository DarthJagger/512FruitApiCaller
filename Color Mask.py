import requests
import cv2
import threading
import time
import queue
import io


def api_Call(photo, start_time):
    # Write the frame to a file and send the API request
    success, encoded_image = cv2.imencode('.jpg', photo)

    if not success:
        print("Failed to encode image.")
        api_queue.put((None, 0))  # In case of error, put a dummy value in the queue
        return

    # Convert the encoded image to a byte stream
    image_bytes = io.BytesIO(encoded_image.tobytes())

    # Prepare the files dictionary with a byte stream
    files = {'img_file': ('frame.jpg', image_bytes, 'image/jpeg')}

    try:
        # Send the request to the API
        response = requests.get(
            'https://food-grader-a9gydgcffbg7f9dv.eastus2-01.azurewebsites.net/file',
            files=files
        )

        # After receiving the response, calculate the time from the initial API call
        end_time = time.time()
        total_time = end_time - start_time

        # Extract the prediction from the response
        prediction = response.json().get('prediction', 0)

        # Put the response and the total time in the queue
        api_queue.put((prediction, total_time))

    except Exception as e:
        print(f"Error making API call: {e}")
        api_queue.put((None, 0))  # In case of error, put a dummy value in the queue


# Initialize the webcam capture
cap = cv2.VideoCapture(0)
status, photo = cap.read()

# Queue for API responses
api_queue = queue.Queue()

# Lock for ensuring only one API call at a time
api_lock = threading.Lock()

# Variable to store the last prediction
last_prediction = None


def periodic_api_call():
    while True:
        # Sleep for 5 seconds (or modify as needed)
        time.sleep(3)

        # Only make an API call if the lock is available
        with api_lock:
            if status:
                start_time = time.time()  # Capture the time when the API call is initiated
                api_Call(photo, start_time)
                #threading.Thread(target=api_Call, args=(photo, start_time)).start()


# Start the periodic API call in a separate thread
threading.Thread(target=periodic_api_call, daemon=True).start()

while True:
    # Read the next frame from the webcam
    status, photo = cap.read()

    if status:
        # Check if there's any new API response to process
        if not api_queue.empty():
            prediction, total_time = api_queue.get()

            if prediction is not None:
                print(f"Prediction: {prediction}")
                print(f"Time from API call to response: {total_time:.2f} seconds")

                # Update the last prediction
                last_prediction = prediction

        # Apply the appropriate mask based on the last prediction
        if last_prediction == 1:
            mask_color = (0, 0, 255)  # Red in BGR
        elif last_prediction == 0:
            mask_color = (0, 255, 0)  # Green in BGR
        else:
            mask_color = (0, 0, 0)  # No mask if prediction is something else

        if last_prediction is not None:
            # Apply a translucent mask over the image
            overlay = photo.copy()
            cv2.rectangle(overlay, (0, 0), (photo.shape[1], photo.shape[0]), mask_color,
                          thickness=-1)  # Full screen overlay
            photo = cv2.addWeighted(overlay, 0.5, photo, 1 - 0.5,
                                    0)  # Blend the image with the overlay (translucent effect)

        # Display the image
        cv2.imshow("Color Mask", photo)

    # Exit the loop on pressing Enter
    if cv2.waitKey(50) == 13:
        break

cap.release()
cv2.destroyAllWindows()
