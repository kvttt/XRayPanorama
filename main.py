import cv2
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import SimpleITK as sitk

from utils import Evaluate2DTranslationCorrelation, create_images_in_shared_coordinate_system, \
	composite_images_alpha_blending, final_registration, \
	calculate_angle

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--hip', '-p', required=True, help='path to hip DICOM image')
parser.add_argument('--knee', '-k', required=True, help='path to knee DICOM image')
parser.add_argument('--ankle', '-a', required=True, help='path to ankle DICOM image')
parser.add_argument('--output', '-o', required=True, help='path to output folder')
args = parser.parse_args()

# Read in images
print('Reading in images...')
h_file = dicom.dcmread(args.hip)
h = h_file.pixel_array
k_file = dicom.dcmread(args.knee)
k = k_file.pixel_array
a_file = dicom.dcmread(args.ankle)
a = a_file.pixel_array
files = [h_file, k_file, a_file]
pixel_arrays = [h, k, a]
print('Done.')

"""
Step 1 
Display 3 images side by side
"""

f, arr = plt.subplots(1, 3, figsize=(24, 8))
arr[0].imshow(h, cmap='bone')
arr[0].set_title('Hip')
arr[1].imshow(k, cmap='bone')
arr[1].set_title('Knee')
arr[2].imshow(a, cmap='bone')
arr[2].set_title('Ankle')

f.tight_layout()
print('Close the window to proceed.')
plt.show()

DO_CROP = input('Do you wish to crop image to get rid of artifacts? (y/N) ')
if DO_CROP.lower() == 'y':

	"""
	Step 2
	Crop images to get rid of artifacts
	"""

	names = ['Hip', 'Knee', 'Ankle']

	f, arr = plt.subplots(1, 3, figsize=(24, 8))

	for i, img in enumerate([h, k, a]):
		cv2.destroyAllWindows()
		top_y, bottom_y = None, None

		# normalize img to [0, 1]
		# img = (img - np.min(img)) / (np.max(img) - np.min(img))
		normed_img = (img - img.min()) / (img.max() - img.min())

		# Mouse event
		def select_level(event, x, y, flags, param):
			global ix, iy, top_y, bottom_y

			if event == cv2.EVENT_LBUTTONDBLCLK:
				if top_y is not None and bottom_y is not None:
					print('Top and bottom boundaries already specified. Press Enter to confirm...')
				else:
					ix, iy = x, y
					cv2.line(normed_img, (0, iy), (img.shape[1], iy), (255, 0, 0), 5)
					if top_y is None:
						top_y = iy
						print(f'Top boundary specified at {top_y}.')
					elif bottom_y is None:
						bottom_y = iy
						print(f'Bottom boundary specified at {bottom_y}. Press Enter to confirm...')

		# initialize window
		cv2.namedWindow('HKA', cv2.WINDOW_NORMAL)
		cv2.setMouseCallback('HKA', select_level)
		print('Please double click twice to specify the top and bottom boundaries to crop out the artifacts. '
		      'Press Enter to confirm...')

		while True:
			temp = np.copy(normed_img)
			cv2.resizeWindow('HKA', 800, 800)
			cv2.imshow('HKA', temp)
			if cv2.waitKey(20) == ord('\r'):
				break
		cv2.destroyAllWindows()

		# Crop original image
		print(f'Cropping image {i + 1}...')
		cropped_img = img[top_y:bottom_y, :]
		# cropped_img_sitk = sitk.GetImageFromArray(cropped_img)
		pixel_arrays[i] = cropped_img
		print(f'Done. Cropped image {i + 1} shape: {cropped_img.shape}.')

		# Create a new DICOM image
		print(f'Saving as new DICOM image...')
		files[i].PixelData = cropped_img.tobytes()
		files[i].Rows, files[i].Columns = cropped_img.shape
		if not os.path.exists(args.output):
			os.makedirs(args.output)
		files[i].save_as(os.path.join(args.output, f'{names[i]}_cropped.dcm'))
		print(f'Done. Saved as {names[i]}_cropped.dcm.')

		# Display using matplotlib.pyplot
		arr[i].imshow(cropped_img, cmap='bone')
		arr[i].set_title(names[i])

	f.tight_layout()
	print('Close the window to proceed.')
	plt.show()

"""
Step 3
Composite 3 images into 1
"""

# Load cropped images
hip_image = sitk.GetImageFromArray(pixel_arrays[0])
knee_image = sitk.GetImageFromArray(pixel_arrays[1])
ankle_image = sitk.GetImageFromArray(pixel_arrays[2])

metric_sampling_percentage = 0.2
min_row_overlap = 20
max_row_overlap = 0.5 * hip_image.GetHeight()
column_overlap = 0.2 * hip_image.GetWidth()
dx_step_num = 4
dy_step_num = 10

print('Performing exploration step...')

initializer = Evaluate2DTranslationCorrelation(
	metric_sampling_percentage,
	min_row_overlap,
	max_row_overlap,
	column_overlap,
	dx_step_num,
	dy_step_num,
)

# Get potential starting points for the knee-hip images.
initializer.evaluate(
	fixed_image=sitk.Cast(knee_image, sitk.sitkFloat32),
	moving_image=sitk.Cast(hip_image, sitk.sitkFloat32),
)
plotting_data = [("knee 2 hip", initializer.get_raw_data())]
k2h_candidates = initializer.get_candidates(num_candidates=4, correlation_threshold=0.5)

# Get potential starting points for the ankle-knee images.
initializer.evaluate(
	fixed_image=sitk.Cast(ankle_image, sitk.sitkFloat32),
	moving_image=sitk.Cast(knee_image, sitk.sitkFloat32),
)
plotting_data.append(("ankle 2 knee", initializer.get_raw_data()))
a2k_candidates = initializer.get_candidates(num_candidates=4, correlation_threshold=0.5)

# We will use the hip image coordinate system as the common coordinate system
# and visualize the results with the transformations corresponding to the best
# similarity metric values.
knee2hip_transform = k2h_candidates[0][0]
ankle2knee_transform = a2k_candidates[0][0]
ankle2hip_transform = sitk.CompositeTransform(
	[knee2hip_transform, ankle2knee_transform]
)
image_transform_list = [
	(hip_image, sitk.TranslationTransform(2)),
	(knee_image, knee2hip_transform),
	(ankle_image, ankle2hip_transform),
]
composite_image = composite_images_alpha_blending(
	create_images_in_shared_coordinate_system(image_transform_list)
)

# gui.multi_image_display2D([composite_image], figure_size=(4, 8))
print("knee2hip_correlation: {0:.2f}".format(k2h_candidates[0][1]))
print("ankle2hip_correlation: {0:.2f}".format(a2k_candidates[0][1]))

print('Performing exploitation step...')

# Copy the initial transformations for use in the final registration
initial_transformation_list_k2h = [
	sitk.TranslationTransform(t) for t, corr in k2h_candidates
]
initial_transformation_list_a2k = [
	sitk.TranslationTransform(t) for t, corr in a2k_candidates
]

# Perform the final registration
k2h_final = final_registration(
	fixed_image=sitk.Cast(knee_image, sitk.sitkFloat32),
	moving_image=sitk.Cast(hip_image, sitk.sitkFloat32),
	initial_mutable_transformations=initial_transformation_list_k2h,
)
a2k_final = final_registration(
	fixed_image=sitk.Cast(ankle_image, sitk.sitkFloat32),
	moving_image=sitk.Cast(knee_image, sitk.sitkFloat32),
	initial_mutable_transformations=initial_transformation_list_a2k,
)

knee2hip = min(k2h_final, key=lambda x: x[1])
knee2hip_transform = knee2hip[0]

ankle2knee = min(a2k_final, key=lambda x: x[1])
ankle2hip_transform = sitk.CompositeTransform([knee2hip_transform, ankle2knee[0]])

image_transform_list = [
	(hip_image, sitk.TranslationTransform(2)),
	(knee_image, knee2hip_transform),
	(ankle_image, ankle2hip_transform),
]
composite_image = composite_images_alpha_blending(
	create_images_in_shared_coordinate_system(image_transform_list)
)

size = composite_image.GetSize()

composite_image = sitk.RegionOfInterest(composite_image, [size[0] - 20, size[1]], [0, 0])
size_new = composite_image.GetSize()

# gui.multi_image_display2D([composite_image], figure_size=(4, 8))
print("knee2hip_correlation: {0:.2f}".format(knee2hip[1]))
print("ankle2hip_correlation: {0:.2f}".format(ankle2knee[1]))

composite_image = sitk.Cast(composite_image, sitk.sitkUInt16)

composite_array = sitk.GetArrayFromImage(composite_image)

# Display using matplotlib.pyplot
plt.imshow(composite_array, cmap='bone')
plt.title('Composite Image')
print('Close the window to proceed.')
plt.show()

DO_NORMALIZE = input('Do you wish to normalize each image? (y/N) ')
if DO_NORMALIZE.lower() == 'y':

	"""
	Step 4
	Optional normalization
	"""

	print('Normalizing images...')

	composite_image = composite_images_alpha_blending(
		create_images_in_shared_coordinate_system(image_transform_list),
		normalize=True
	)

	composite_image = sitk.RegionOfInterest(composite_image, [size[0] - 20, size[1]], [0, 0])

	composite_image = sitk.Cast(composite_image, sitk.sitkUInt16)

	composite_array = sitk.GetArrayFromImage(composite_image)

	print('Done.')

	# Display using matplotlib.pyplot
	plt.imshow(composite_array, cmap='bone')
	plt.title('Composite Image')
	print('Close the window to proceed.')
	plt.show()

ds_dst = h_file

new_study_uid = dicom.uid.generate_uid()
ds_dst.StudyInstanceUID = new_study_uid

ds_dst.SeriesDescription = 'Composite Image'
ds_dst.SeriesNumber = 1004
ds_dst.Rows, ds_dst.Columns = composite_array.shape

# Set the pixel data
ds_dst.PixelData = composite_array.tobytes()

# Save the DICOM file
ds_dst.save_as(os.path.join(args.output, 'composite.dcm'))

print('Done. Saved composite image as composite.dcm.')

CALC_ANGLE = input('Do you wish to calculate the hip-knee-ankle (HKA) angle? (y/N) ')
if CALC_ANGLE.lower() == 'y':

	"""
	Step 5
	Angle calculation
	"""

	cv2.destroyAllWindows()

	composite_array = (composite_array - composite_array.min()) / (composite_array.max() - composite_array.min())

	# Select three points
	points = []

	def select_point(event, x, y, flags, param):
		global points
		if event == cv2.EVENT_LBUTTONUP:
			if len(points) <= 2:
				points.append((x, y))
				print(f'Point {len(points)} selected at ({x}, {y}).')
				cv2.circle(composite_array, (x, y), 30, (0, 0, 255), -1)
				cv2.imshow('Composite', composite_array)
				if len(points) == 3:
					cv2.line(composite_array, points[0], points[1], (0, 255, 0), 20, cv2.LINE_AA)
					cv2.line(composite_array, points[1], points[2], (0, 255, 0), 20, cv2.LINE_AA)
					cv2.imshow('Composite', composite_array)
					print('HKA angle specified. Calculating the HKA angle...')
					angle = calculate_angle(*points)
					print(f'The HKA angle is: {180 - angle}. Press Enter to exit.')
					if cv2.waitKey(20) == ord('\r'):
						cv2.destroyAllWindows()

	# initialize window
	cv2.namedWindow('Composite', cv2.WINDOW_NORMAL)
	cv2.imshow('Composite', composite_array)
	cv2.resizeWindow('Composite',
	                 int(composite_array.shape[1] / 10),
	                 int(composite_array.shape[0] / 10),
	                 )
	cv2.setMouseCallback('Composite', select_point)
	print('Please click three times to specify three points. Press Enter to confirm...')
	cv2.waitKey()
	cv2.destroyAllWindows()
