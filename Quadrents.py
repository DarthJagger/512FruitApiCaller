import requests
import cv2
import threading
import time
import queue
import io


def api_Call(photo, quadrant, start_time, result_queue):
    # Crop the photo based on the quadrant
    h, w = photo.shape[:2]
    if quadrant == 'top_left':
        cropped_photo = photo[0:h // 2, 0:w // 2]
    elif quadrant == 'top_right':
        cropped_photo = photo[0:h // 2, w // 2:w]
    elif quadrant == 'bottom_left':
        cropped_photo = photo[h // 2:h, 0:w // 2]
    elif quadrant == 'bottom_right':
        cropped_photo = photo[h // 2:h, w // 2:w]

    # Save the cropped quadrant to a file
    success, encoded_image = cv2.imencode('.jpg', photo)

    if not success:
        print("Failed to encode image.")
        return

    # Convert the encoded image to a byte stream
    image_bytes = io.BytesIO(encoded_image.tobytes())

    # Prepare the files dictionary with a byte stream
    files = {'img_file': ('frame.jpg', image_bytes, 'image/jpeg')}


     # Send the request to the API
    try:
       #update where is says link to your API link created from uploading the created docker
        response = requests.get(
            'link',
            files=files
        )

        # After receiving the response, calculate the time from the initial API call
        end_time = time.time()
        total_time = end_time - start_time

        # Extract the prediction from the response
        prediction = response.json().get('prediction', 0)

        # Put the response and the quadrant info in the queue
        result_queue.put((prediction, quadrant, total_time))

    except Exception as e:
        print(f"Error making API call for {quadrant}: {e}")
        result_queue.put((None, quadrant, 0))  # In case of error, put a dummy value in the queue


def quadrant_api_thread(photo, quadrant, initWait, result_queue):
    time.sleep(initWait)
    last_called = 0
    while True:
        current_time = time.time()

        # Make an API call if the interval has passed
        time.sleep(12)
        start_time = time.time()  # Capture the time when the API call is initiated
        #threading.Thread(target=api_Call, args=(photo, quadrant, start_time, result_queue)).start()
        api_Call(photo, quadrant, start_time, result_queue)
        last_called = current_time


def apply_mask_to_quadrant(photo, quadrant, prediction, latest_updated_quadrant):
    h, w = photo.shape[:2]

    if quadrant == 'top_left':
        mask_region = (0, 0, w // 2, h // 2)
    elif quadrant == 'top_right':
        mask_region = (w // 2, 0, w, h // 2)
    elif quadrant == 'bottom_left':
        mask_region = (0, h // 2, w // 2, h)
    elif quadrant == 'bottom_right':
        mask_region = (w // 2, h // 2, w, h)
    else:
        return photo

    # Apply the mask based on the prediction (Red for 1, Green for 0)
    if prediction == 1:
        mask_color = (0, 0, 255)  # Red in BGR
    elif prediction == 0:
        mask_color = (0, 255, 0)  # Green in BGR
    else:
        mask_color = (0, 0, 0)  # No mask

    # Create a translucent mask on the region
    overlay = photo.copy()
    cv2.rectangle(overlay, (mask_region[0], mask_region[1]), (mask_region[2], mask_region[3]), mask_color, thickness=-1)

    # Add the "plus" effect only to the most recently updated quadrant
    if quadrant == latest_updated_quadrant:
        center_x = (mask_region[0] + mask_region[2]) // 2
        center_y = (mask_region[1] + mask_region[3]) // 2
        cross_size = 30  # Size of the cross

        # Draw horizontal and vertical lines for the "plus" mark
        cv2.line(overlay, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (255, 255, 255), 2)  # Horizontal
        cv2.line(overlay, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (255, 255, 255), 2)  # Vertical

    # Combine the overlay with the original image
    photo = cv2.addWeighted(overlay, 0.5, photo, 1 - 0.5, 0)

    return photo


# Initialize the webcam capture
cap = cv2.VideoCapture(0)
status, photo = cap.read()

# Queue for API responses
result_queue = queue.Queue()

# Variable to store predictions for each quadrant
predictions = {'top_left': None, 'top_right': None, 'bottom_left': None, 'bottom_right': None}

# Variable to track the most recently updated quadrant
latest_updated_quadrant = None

# Start threads for each quadrant with different intervals
threading.Thread(target=quadrant_api_thread, args=(photo, 'top_left', 0, result_queue), daemon=True).start()
threading.Thread(target=quadrant_api_thread, args=(photo, 'top_right', 3, result_queue), daemon=True).start()
threading.Thread(target=quadrant_api_thread, args=(photo, 'bottom_left', 6, result_queue), daemon=True).start()
threading.Thread(target=quadrant_api_thread, args=(photo, 'bottom_right', 9, result_queue), daemon=True).start()

while True:
    # Read the next frame from the webcam
    status, photo = cap.read()

    if status:
        # Check if there's any new API response to process
        if not result_queue.empty():
            prediction, quadrant, total_time = result_queue.get()

            if prediction is not None:
                print(f"Prediction for {quadrant}: {prediction}")
                print(f"Time from API call to response for {quadrant}: {total_time:.2f} seconds")

                # Update the prediction for the corresponding quadrant
                predictions[quadrant] = prediction
                # Update the latest updated quadrant
                latest_updated_quadrant = quadrant

        # Apply the appropriate mask based on the prediction for each quadrant
        for quadrant in predictions:
            prediction = predictions[quadrant]
            photo = apply_mask_to_quadrant(photo, quadrant, prediction, latest_updated_quadrant)

        # Display the image with masks
        cv2.imshow("ThreadContainsAPICall", photo)

    # Exit the loop on pressing Enter
    if cv2.waitKey(50) == 13:
        break

cap.release()
cv2.destroyAllWindows()
