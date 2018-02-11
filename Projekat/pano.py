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
			self.imgs.append(cv2.imread(i))
			#self.imgs.append(cv2.resize(cv2.imread(i),(480, 320)))

	def fill_lists(self):
		self.left_list, self.right_list = [], []
		centerIdx = len(self.imgs)/2 
		for i in range(len(self.imgs)):
			if(i<=centerIdx):
				self.left_list.append(self.imgs[i])
			else:
				self.right_list.append(self.imgs[i])
		self.right_list.insert(0,self.imgs[int(centerIdx)])

	def warpImages(self, img1, img2, H):
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
	    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
	    
	    return output_img

	def l_stitch(self):
		left = self.left_list[0]
		for right in self.left_list[1:]:
			H = self.match(left, right)
			invH = np.linalg.inv(H)
			left = self.warpImages(right, left, invH)
			self.count += 1
			print('stitched {}/{} imgs'.format(self.count, len(self.imgs)))
		self.right_list.insert(0,left[:,0:int(left.shape[1] - right.shape[1]/2)])

	def r_stitch(self):
		self.right_list.reverse()
		right = self.right_list[0]
		for left in self.right_list[1:]:
			H = self.match(left, right)
			right = self.warpImages(left, right, H)
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


if __name__ == '__main__':
	try:
		args = sys.argv[1]
	except:
		args = "input/files1.txt"
	print("Parameters : ", args)
	s = PanoStitcher(args)
	s.l_stitch()
	s.r_stitch()
	cv2.imwrite("output/" + args.split('/')[1].split('.')[0] + ".jpg", s.final_img)
	print("image written to output/" + args.split('/')[1].split('.')[0] + ".jpg")
	cv2.destroyAllWindows()
	