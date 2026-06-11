import os
import os.path
import cv2 as cv
import glob
import imutils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTCHA_IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "generated_captcha_images")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "extracted_letter_images")

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, '*'))
counts = {}

# loop over image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):

    # Since the filename contains the captcha text
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Load the image and convert it to grayscale
    image = cv.imread(captcha_image_file)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    gray = cv.copyMakeBorder(gray, 8, 8, 8, 8, cv.BORDER_REPLICATE)

    # threshold the image
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]

    # find the contours in the image
    contours_raw = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = contours_raw[0] if len(contours_raw) == 2 else contours_raw[1]

    letter_image_regions = []

    # Loop through contours and extract letter regions, filtering out noise
    for contour in contours:
        if cv.contourArea(contour) < 20:
            continue
            
        (x, y, w, h) = cv.boundingRect(contour)

        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))

    # If we found more or less than 4 letters in the captcha, the letter
    # extraction didn't work correctly. Skip the image instead
    # of saving bad training data
    if len(letter_image_regions) != 4:
        continue

    # Sort the detected letter images based on the x coordinate to
    # make sure we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        y_start, y_end = max(0, y - 2), min(gray.shape[0], y + h + 2)
        x_start, x_end = max(0, x - 2), min(gray.shape[1], x + w + 2)
        letter_image = gray[y_start:y_end, x_start:x_end]

        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1