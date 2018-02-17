import numpy as np
import cv2

class PanoStitcher:
	def __init__(self, path):
		self.imgs = []
		self.count = 0;
		self.load_images(path)
		self.fill_lists()
		self.surf = cv2.xfeatures2d.SURF_create()
		
	def load_images(self, path):
		f = open(path, 'r')
		img_names = []
		for i in f:
			img_names.append(i.strip())
		for i in img_names:
			#self.imgs.append(cv2.imread(i))
			self.imgs.append(cv2.resize(cv2.imread(i),(480, 320)))

	def fill_lists(self):
		self.left_list, self.right_list = [], []
		centerIdx = len(self.imgs)/2 
		for i in range(len(self.imgs)):
			if(i<=centerIdx):
				self.left_list.append(self.imgs[i])
			else:
				self.right_list.append(self.imgs[i])
		self.right_list.insert(0,self.imgs[int(centerIdx)])

	def warpImages(self, img1, img2, H, direction):
		rows1, cols1 = img1.shape[:2]
		rows2, cols2 = img2.shape[:2]

		list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
		temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
		list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
		list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

		[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
		[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
		translation_dist = [-x_min, -y_min]
		H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

		output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
		if self.blend:
			op_dir = 1
			if direction == 'left':
				op_dir = 0
			for y in range(translation_dist[1],rows1+translation_dist[1]):
				for x in range(translation_dist[0],cols1+translation_dist[0]):
					img1_pixel = img1[y-translation_dist[1],x-translation_dist[0]]
					op = abs(op_dir-(x-translation_dist[0])/cols1)
					inv_op = 1-op
					if (output_img[y,x] == [0,0,0]).all():
						output_img[y,x] = img1_pixel
					else:
						output_img[y,x] = output_img[y,x]*inv_op + img1_pixel*op

			cv2.imshow('o', output_img)
			cv2.waitKey()
		else:
			output_img[translation_dist[1]:rows1 + translation_dist[1],translation_dist[0]:cols1 + translation_dist[0]] = img1
		return output_img

	def l_stitch(self):
		left = self.left_list[0]
		for right in self.left_list[1:]:
			H = self.match(left, right)
			invH = np.linalg.inv(H)
			left = self.warpImages(right, left, invH, 'left')
			self.count += 1
			print('stitched {}/{} imgs'.format(self.count, len(self.imgs)))
		self.right_list.insert(0,left[:,0:int(left.shape[1] - right.shape[1]/2)])

	def r_stitch(self):
		self.right_list.reverse()
		right = self.right_list[0]
		for left in self.right_list[1:]:
			H = self.match(left, right)
			right = self.warpImages(left, right, H, 'right')
			self.count += 1
			print('stitched {}/{} imgs'.format(self.count, len(self.imgs)))
		self.final_img = self.crop(right)

	def crop(self, image):
		imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		_,th2 = cv2.threshold(imgray,8,255,cv2.THRESH_BINARY)
		_, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(contour) for contour in contours]
		max_index = np.argmax(areas)
		cnt = contours[max_index]
		x,y,w,h = cv2.boundingRect(cnt)
		# Crop with the largest rectangle
		crop = image[y:y+h,x:x+w]
		return crop

	def match(self, i1, i2):
		imageSet1 = self.SURF_compute(i1)
		imageSet2 = self.SURF_compute(i2)
		flann = cv2.FlannBasedMatcher({'algorithm' : 0, 'trees' : 5}, {'checks' : 50})
		matches = flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
		good = []
		for _ , (i, j) in enumerate(matches):
			if i.distance < 0.7*j.distance:
				good.append((i.trainIdx, i.queryIdx))

		if len(good) > 4:
			matchedPoints2 = np.float32(
				[imageSet2['kp'][i].pt for (__, i) in good]
			)
			matchedPoints1 = np.float32(
				[imageSet1['kp'][i].pt for (i, __) in good]
				)

			H, s = cv2.findHomography(matchedPoints2, matchedPoints1, cv2.RANSAC, 4)
			return H
		return None

	def SURF_compute(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		keypoings, descriptors = self.surf.detectAndCompute(gray, None)
		return {'kp':keypoings, 'des':descriptors}

	def inscribe(self, img):
		imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_,th2 = cv2.threshold(imgray,8,255,cv2.THRESH_BINARY)
		im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		print('Found contures', len(contours))

		max_size = 0;
		id = 0;
		for i in range(0,len(contours)):
			if len(contours[i]) > max_size:
				max_size = len(contours[i])
				id = i
		print('Chosen id', id)
		print('Max size', max_size)

		c_sorted_x = sorted(contours[id], key=lambda point: point[0][0])
		c_sorted_y = sorted(contours[id], key=lambda point: point[0][1])
		print('c sorted x', c_sorted_x)
		print('c sorted y', c_sorted_y)
		min_x_id = 0
		min_y_id = 0
		max_x_id = len(c_sorted_x)-1
		print('max x', c_sorted_x[len(c_sorted_x)-1])
		max_y_id = len(c_sorted_y)-1
		contour_mask = np.zeros(img.shape, dtype=np.uint8)
		cv2.drawContours(contour_mask, contours, -1, (0, 255, 0), 1)
		cv2.fillPoly(contour_mask, pts=contours, color=(255, 255, 255))
		while min_x_id < max_x_id and min_y_id < max_y_id:
			min = (c_sorted_x[min_x_id][0][0], c_sorted_y[min_y_id][0][1])
			max = (c_sorted_x[max_x_id][0][0] - c_sorted_x[min_x_id][0][0], c_sorted_y[max_y_id][0][1] - c_sorted_y[min_y_id][0][1])


			finished, ocTop, ocBottom, ocLeft, ocRight = self.checkInteriorExterior(contour_mask, min, max)
			if finished:
				break

			if ocLeft:
				min_x_id += 1
			if ocRight:
				max_x_id -= 1
			if ocTop:
				min_y_id += 1
			if ocBottom:
				max_y_id -= 1
		#cv2.rectangle(contour_mask, min, max, (255,0,0), 2)
		#cv2.imshow('cm', contour_mask)
		#cv2.waitKey()
		return min, max

	def checkInteriorExterior(self, sub, min, max):
		returnVal = True

		cTop = 0
		cBottom = 0
		cLeft = 0
		cRight = 0
		y = min[1]
		for x in range(min[0], max[0]):
			if (sub[y,x] == 0).all():
				returnVal = False
				cTop += 1
		y = max[1]
		for x in range(min[0], max[0]):
			if (sub[y,x] == 0).all():
				returnVal = False
				cBottom += 1

		x = min[0]
		for y in range(min[1],max[1]):
			if (sub[y,x] == 0).all():
				returnVal = False
				cLeft += 1

		x = max[0]
		for y in range(min[1],max[1]):
			if (sub[y,x] == 0).all():
				returnVal = False
				cRight += 1
		if cTop >= cBottom and cTop >= cLeft and cTop >= cRight:
			return returnVal, True, False, False, False
		elif cBottom >= cLeft and cBottom >= cRight and cBottom >= cTop:
			return returnVal, False, True, False, False
		elif cLeft >= cRight and cLeft >= cBottom and cLeft >= cTop:
			return returnVal, False, False, True, False
		else:
			return returnVal, False, False, False, True





if __name__ == '__main__':
	try:
		args = sys.argv[1]
	except:
		args = "input/filesm.txt"
	print("Parameters : ", args)
	s = PanoStitcher(args)
	s.blend = False
	s.l_stitch()
	s.r_stitch()
	print('image size', s.final_img.shape)
	cv2.imwrite("output/sup_" + args.split('/')[1].split('.')[0] + ".jpg", s.final_img)
	print("image written to output/sup_" + args.split('/')[1].split('.')[0] + ".jpg")
	ins = s.inscribe(s.final_img)
	cv2.imwrite("output/ins_" + args.split('/')[1].split('.')[0] + ".jpg", s.final_img[ins[0][1]:ins[1][1],ins[0][0]:ins[1][0]])
	print("image written to output/ins_" + args.split('/')[1].split('.')[0] + ".jpg")
	cv2.destroyAllWindows()
	