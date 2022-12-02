import cv2
import numpy as np
import pyrealsense2 as rs
import skimage.exposure

# 1) identify and segment out the contours (generate a mask that only  leaves each individual block in the image)
# 2) identify the color label of each block
# 3) identify the shape label of each block
# 4) determine the object COM
# 5) estimate the object's dimensions (maybe dependent on block shape????, or approximate all as rectangles????)

# code up the stacking heuristic algorithm
# NOTE: perhaps mask out points past certain depth threshold???? (to mask out noisy background)

# create loop that shows the color image and the depth image and the masked background color and depth images

W = 848
H = 480

# ----- Camera 3 (static) -----
REALSENSE_INTRINSICS_CAM_3 = "calib/realsense_intrinsics_camera2.intr"
REALSENSE_TF_CAM_3 = "calib/realsense_camera2.tf"
# NOTE: can only see the last 16 corners for calibration
pipeline = rs.pipeline()
config = rs.config()
config.enable_device('151322069488')
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# start streaming
pipeline.start(config)

# align stream
aligned_stream = rs.align(rs.stream.color)

# # ------ save baseline image -----
# frames = pipeline.wait_for_frames()
# frames = aligned_stream.process(frames)
# color_frame = frames.get_color_frame()
# depth_frame = frames.get_depth_frame().as_depth_frame()
# color_image = np.asanyarray(color_frame.get_data())
# depth_image = np.asanyarray(depth_frame.get_data())

# # skip empty frames
# if not np.any(depth_image):
# 	print("no depth")

# # edit the depth image to make it easier to visualize the depth
# stretch = skimage.exposure.rescale_intensity(depth_image, in_range='image', out_range=(0,255)).astype(np.uint8)
# stretch = cv2.merge([stretch, stretch, stretch])

# background_color = color_image
# background_depth = stretch

# # # Show the images
# # cv2.imshow("Color Image No Masking", color_image)
# # cv2.imshow("Augmented Depth Image: ", stretch)
# # cv2.waitKey(1)

# color_filename = "scripts/Images/color_background.jpg"
# depth_filename = "scripts/Images/depth_background.jpg"
# cv2.imwrite(color_filename, color_image)
# cv2.imwrite(depth_filename, depth_image)


# import baseline background images
background_color = cv2.imread("scripts/Images/color_background.jpg")
background_depth = cv2.imread("scripts/Images/depth_background.jpg")



while True:
# for i in range(2):
	frames = pipeline.wait_for_frames()
	frames = aligned_stream.process(frames)
	color_frame = frames.get_color_frame()
	depth_frame = frames.get_depth_frame().as_depth_frame()
	color_image = np.asanyarray(color_frame.get_data())
	depth_image = np.asanyarray(depth_frame.get_data())

	# skip empty frames
	if not np.any(depth_image):
		print("no depth")
		# continue

	# points = point_cloud.calculate(depth_frame)
	# verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)

	# edit the depth image to make it easier to visualize the depth
	stretch = skimage.exposure.rescale_intensity(depth_image, in_range='image', out_range=(0,255)).astype(np.uint8)
	stretch = cv2.merge([stretch, stretch, stretch])

	# eliminate the noisy background
	color_diff = abs(color_image - background_color)
	grey_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
	ret, foreground = cv2.threshold(grey_diff, 190, 255, cv2.THRESH_BINARY_INV)	# NOTE: maybe make this adaptive thresholding????

	# mask the color image
	color_masked = cv2.bitwise_and(color_image, color_image, mask=foreground)


	# create contours
	grey_masked = cv2.cvtColor(color_masked, cv2.COLOR_BGR2GRAY)
	contours, hierarchy = cv2.findContours(grey_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cont_mask = np.zeros(grey_masked.shape, dtype='uint8')
	# cv2.drawContours(cont_mask, contours, -1, color=(255,255,255), thickness = cv2.FILLED)
	# cv2.imshow("Contours", cont_mask)

	# detect edges
	edges = cv2.Canny(grey_masked, 100,100)
	# cv2.imshow("Edges", edges)	

	# draw/calculate the centers of objects
	clean_contours = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			# cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			clean_contours.append(cnt)

	clean_contours = tuple(clean_contours)

	# convert clean contours into a new mask before color thresholding
	clean_mask = np.zeros((H,W), dtype='uint8')
	cv2.fillPoly(clean_mask, pts=clean_contours, color=(255,255,255))
	clean_color_masked = cv2.bitwise_and(color_image, color_image, mask=clean_mask)

	# define dictionary to keep track of blocks (key is the block number, value is [color, contour])
	blocks = {}
	i = 0

	# plot the final contours distinguished by color
	# split into different color masks (red, blue, green, beige)
	# convert to LAB color space
	lab_image = cv2.cvtColor(clean_color_masked, cv2.COLOR_BGR2LAB)
	l_channel = lab_image[:,:,0]
	a_channel = lab_image[:,:,1]	# spectrum from green to red
	b_channel = lab_image[:,:,2]	# spectrum from yellow to blue
	# print("Min A:", np.min(a_channel))
	# print("Max A: ", np.max(a_channel))
	
	# green threshold (GOOD!)
	green_a_thresh = cv2.threshold(a_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	green_b_thresh = cv2.threshold(b_channel, 150, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	green_thresh = np.zeros((H,W), dtype='uint8')
	green_idx = np.where(np.equal(green_a_thresh, green_b_thresh))
	green_thresh[green_idx] = green_b_thresh[green_idx]
	green_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = green_thresh)
	green_cnts, green_hierarchy = cv2.findContours(cv2.cvtColor(green_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("Green Masked", green_masked)
	for cnt in green_cnts:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			cv2.putText(color_image, "green", (cX - 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			blocks[i] = [cnt, "green"]
			i+=1

	# red threshold (GOOD!)
	red_a_thresh = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	red_b_thresh = cv2.threshold(b_channel, 150, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	red_thresh = np.zeros((H,W), dtype='uint8')
	red_idx = np.where(np.equal(red_a_thresh, red_b_thresh))
	red_thresh[red_idx] = red_a_thresh[red_idx]
	red_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = red_thresh)
	red_cnts, red_hierarchy = cv2.findContours(cv2.cvtColor(red_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("Red Masked", red_masked)
	for cnt in red_cnts:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			cv2.putText(color_image, "red", (cX - 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			blocks[i] = [cnt, "red"]
			i+=1

	# blue threshold (GOOD ENOUGH! Eliminate beige with contour area constraints)
	blue_b_thresh = cv2.threshold(b_channel, 254, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	blue_l_thresh = cv2.threshold(l_channel, 90, 255, cv2.THRESH_BINARY_INV - cv2.THRESH_OTSU)[1]
	blue_thresh = np.zeros((H,W), dtype='uint8')
	blue_idx = np.where(np.equal(blue_b_thresh, blue_l_thresh))
	blue_thresh[blue_idx] = blue_b_thresh[blue_idx]
	blue_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = blue_thresh)
	blue_cnts, blue_hierarchy = cv2.findContours(cv2.cvtColor(blue_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("Blue Masked", blue_masked)
	for cnt in blue_cnts:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			cv2.putText(color_image, "blue", (cX - 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			blocks[i] = [cnt, "blue"]
			i+=1

	# beige threshold (GOOD!)
	beige_l_thresh = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	beige_a_thresh = cv2.threshold(a_channel, 120, 255, cv2.THRESH_BINARY - cv2.THRESH_OTSU)[1]
	beige_thresh = np.zeros((H,W), dtype='uint8')
	beige_idx = np.where(np.equal(beige_l_thresh, beige_a_thresh))
	beige_thresh[beige_idx] = beige_a_thresh[beige_idx]
	beige_masked = cv2.bitwise_and(clean_color_masked, clean_color_masked, mask = beige_thresh)
	beige_cnts, beige_hierarchy = cv2.findContours(cv2.cvtColor(beige_masked, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.imshow("Beige Masked", beige_masked)
	for cnt in beige_cnts:
		area = cv2.contourArea(cnt)
		if area > 600:
			# compute the center of the contour
			M = cv2.moments(cnt)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv2.drawContours(color_image, [cnt], 0, (0,0,255), 2)
			cv2.putText(color_image, "beige", (cX - 30, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			blocks[i] = [cnt, "beige"]
			i+=1

	# determine the number of individual blocks in the scene
	print("\nTotal Blocks: ", len(blocks))

	# determine the individual blocks' COM


	# TODOS: 
		# 1) calculate the COM along with drawing the contours and add that as an element of the dictionary value
		# 2) code up the heuristic based stacking function to plan the order to select the blocks
				# this function should return an ordered list of the block numbers (dictionary key) to pick up in that order
		# 3) classify each block's shape (use cv2.approxPolyDP to do this!)


	# Show the images
	cv2.imshow("Color Image No Masking", color_image)
	cv2.imshow("Color Image Masked", clean_color_masked)
	cv2.waitKey(1)