import SimpleITK as sitk
import numpy as np


class Evaluate2DTranslationCorrelation:
	"""
    Class for evaluating the correlation value for a given set of
    2D translations between two images. The general relationship between
    the fixed and moving images is assumed (fixed is "below" the moving).
    We use the Exhaustive optimizer to sample the possible set of translations
    and an observer to tabulate the results.

    In this class we abuse the Python dictionary by using a float
    value as the key. This is a unique situation in which the floating
    values are fixed (not resulting from various computations) so that we
    can compare them for exact equality. This means they have the
    same hash value in the dictionary.
    """

	def __init__(
			self,
			metric_sampling_percentage,
			min_row_overlap,
			max_row_overlap,
			column_overlap,
			dx_step_num,
			dy_step_num,
	):
		"""
        Args:
            metric_sampling_percentage: Percentage of samples to use
                                        when computing correlation.
            min_row_overlap: Minimal number of rows that overlap between
                             the two images.
            max_row_overlap: Maximal number of rows that overlap between
                             the two images.
            column_overlap: Maximal translation in columns either in positive
                            and negative direction.
            dx_step_num: Number of samples in parameter space for translation along
                         the x axis is 2*dx_step_num+1.
            dy_step_num: Number of samples in parameter space for translation along
                         the y axis is 2*dy_step_num+1.

        """
		self._registration_values_dict = {}
		self.X = None
		self.Y = None
		self.C = None
		self._metric_sampling_percentage = metric_sampling_percentage
		self._min_row_overlap = min_row_overlap
		self._max_row_overlap = max_row_overlap
		self._column_overlap = column_overlap
		self._dx_step_num = dx_step_num
		self._dy_step_num = dy_step_num

	def _start_observer(self):
		self._registration_values_dict = {}
		self.X = None
		self.Y = None
		self.C = None

	def _iteration_observer(self, registration_method):
		x, y = registration_method.GetOptimizerPosition()
		if y in self._registration_values_dict.keys():
			self._registration_values_dict[y].append(
				(x, registration_method.GetMetricValue())
			)
		else:
			self._registration_values_dict[y] = [
				(x, registration_method.GetMetricValue())
			]

	def evaluate(self, fixed_image, moving_image):
		"""
        Assume the fixed image is lower than the moving image (e.g. fixed=knee, moving=hip).
        The transformations map points in the fixed_image to the moving_image.
        Args:
            fixed_image: Image to use as fixed image in the registration.
            moving_image: Image to use as moving image in the registration.
        """
		minimal_overlap = np.array(
			moving_image.TransformContinuousIndexToPhysicalPoint(
				(
					-self._column_overlap,
					moving_image.GetHeight() - self._min_row_overlap,
				)
			)
		) - np.array(fixed_image.GetOrigin())
		maximal_overlap = np.array(
			moving_image.TransformContinuousIndexToPhysicalPoint(
				(self._column_overlap, moving_image.GetHeight() - self._max_row_overlap)
			)
		) - np.array(fixed_image.GetOrigin())
		transform = sitk.TranslationTransform(
			2,
			(
				(maximal_overlap[0] + minimal_overlap[0]) / 2.0,
				(maximal_overlap[1] + minimal_overlap[1]) / 2.0,
			),
		)

		# Total number of evaluations, translations along the y axis in both directions around the initial
		# value is 2*dy_step_num+1.
		dy_step_length = (maximal_overlap[1] - minimal_overlap[1]) / (
				2 * self._dy_step_num
		)
		dx_step_length = (maximal_overlap[0] - minimal_overlap[0]) / (
				2 * self._dx_step_num
		)
		step_length = dx_step_length
		parameter_scales = [1, dy_step_length / dx_step_length]

		registration_method = sitk.ImageRegistrationMethod()
		registration_method.SetMetricAsCorrelation()
		registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
		registration_method.SetMetricSamplingPercentage(
			self._metric_sampling_percentage
		)
		registration_method.SetInitialTransform(transform, inPlace=True)
		registration_method.SetOptimizerAsExhaustive(
			numberOfSteps=[self._dx_step_num, self._dy_step_num], stepLength=step_length
		)
		registration_method.SetOptimizerScales(parameter_scales)

		registration_method.AddCommand(
			sitk.sitkIterationEvent,
			lambda: self._iteration_observer(registration_method),
		)
		registration_method.AddCommand(sitk.sitkStartEvent, self._start_observer)
		registration_method.Execute(fixed_image, moving_image)

		# Convert the data obtained by the observer to three numpy arrays X,Y,C
		x_lists = []
		val_lists = []
		for k in self._registration_values_dict.keys():
			x_list, val_list = zip(*(sorted(self._registration_values_dict[k])))
			x_lists.append(x_list)
			val_lists.append(val_list)

		self.X = np.array(x_lists)
		self.C = np.array(val_lists)
		self.Y = np.array(
			[
				list(self._registration_values_dict.keys()),
			]
			* self.X.shape[1]
		).transpose()

	def get_raw_data(self):
		"""
        Get the raw data, the translations and corresponding correlation values.
        Returns:
            A tuple of three numpy arrays (X,Y,C) where (X[i], Y[i]) are the translation
            and C[i] is the correlation value for that translation.
        """
		return (np.copy(self.X), np.copy(self.Y), np.copy(self.C))

	def get_candidates(self, num_candidates, correlation_threshold, nms_radius=2):
		"""
        Get the best (most correlated, minimal correlation value) transformations
        from the sample set.
        Args:
            num_candidates: Maximal number of candidates to return.
            correlation_threshold: Minimal correlation value required for returning
                                   a candidate.
            nms_radius: Non-Minima (the optimizer is negating the correlation) suppression
                        region around the local minimum.
        Returns:
            List of tuples containing (transform, correlation). The order of the transformations
            in the list is based on the correlation value (best correlation is entry zero).
        """
		candidates = []
		_C = np.copy(self.C)
		done = num_candidates - len(candidates) <= 0
		while not done:
			min_index = np.unravel_index(_C.argmin(), _C.shape)
			if -_C[min_index] < correlation_threshold:
				done = True
			else:
				candidates.append(
					(
						sitk.TranslationTransform(
							2, (self.X[min_index], self.Y[min_index])
						),
						self.C[min_index],
					)
				)
				# None-minima suppression in the region around our minimum
				start_nms = np.maximum(
					np.array(min_index) - nms_radius, np.array([0, 0])
				)
				# for the end coordinate we add nms_radius+1 because the slicing operator _C[],
				# excludes the end
				end_nms = np.minimum(
					np.array(min_index) + nms_radius + 1, np.array(_C.shape)
				)
				_C[start_nms[0]: end_nms[0], start_nms[1]: end_nms[1]] = 0
				done = num_candidates - len(candidates) <= 0
		return candidates


def create_images_in_shared_coordinate_system(image_transform_list):
	"""
    Resample a set of images onto the same region in space (the bounding)
    box of all images.
    Args:
        image_transform_list: A list of tuples each containing a transformation and an image. The transformations map the
                              images to a shared coordinate system.
    Returns:
        list of images: All images are resampled into the same coordinate system and the bounding box of all images is
                        used to define the new image extent onto which the originals are resampled. The background value
                        for the resampled images is set to 0.
    """
	pnt_list = []
	for image, transform in image_transform_list:
		pnt_list.append(transform.TransformPoint(image.GetOrigin()))
		pnt_list.append(
			transform.TransformPoint(
				image.TransformIndexToPhysicalPoint(
					(image.GetWidth() - 1, image.GetHeight() - 1)
				)
			)
		)

	max_coordinates = np.max(pnt_list, axis=0)
	min_coordinates = np.min(pnt_list, axis=0)

	# We assume the spacing for all original images is the same and we keep it.
	output_spacing = image_transform_list[0][0].GetSpacing()
	# We assume the pixel type for all images is the same and we keep it.
	output_pixelID = image_transform_list[0][0].GetPixelID()
	# We assume the direction for all images is the same and we keep it.
	output_direction = image_transform_list[0][0].GetDirection()
	output_width = int(
		np.round((max_coordinates[0] - min_coordinates[0]) / output_spacing[0])
	)
	output_height = int(
		np.round((max_coordinates[1] - min_coordinates[1]) / output_spacing[1])
	)
	output_origin = (min_coordinates[0], min_coordinates[1])

	images_in_shared_coordinate_system = []
	for image, transform in image_transform_list:
		images_in_shared_coordinate_system.append(
			sitk.Resample(
				image,
				(output_width, output_height),
				transform.GetInverse(),
				sitk.sitkLinear,
				output_origin,
				output_spacing,
				output_direction,
				0.0,
				output_pixelID,
			)
		)
	return images_in_shared_coordinate_system


def composite_images_alpha_blending(images_in_shared_coordinate_system, alpha=0.5, normalize=False):
	"""
    Composite a list of images sharing the same extent (size, origin, spacing, direction cosine).
    Args:
        images_in_shared_coordinate_system: A list of images sharing the same meta-information (origin, size, spacing, direction cosine).
        We assume zero denotes background.
        alpha: Alpha blending value.
        normalize: whether to match the intensities of three images before blending.
    Returns:
        SimpleITK image with pixel type sitkFloat32: alpha blending of the images.

    """
	# Composite all of the images using alpha blending where there is overlap between two images, otherwise
	# just paste the image values into the composite image. We assume that at most two images overlap.
	# We go from bottom to top
	composite_image = sitk.Cast(images_in_shared_coordinate_system[2], sitk.sitkFloat32)
	for img in images_in_shared_coordinate_system[:2][::-1]:
		current_image = sitk.Cast(img, sitk.sitkFloat32)

		# Get overlap map
		mask_composite = sitk.Cast(composite_image != 0, sitk.sitkFloat32)
		mask_current = sitk.Cast(current_image != 0, sitk.sitkFloat32)
		intersection_mask = mask_composite * mask_current
		intersection_mask_mean = sitk.Cast(intersection_mask, sitk.sitkUInt16)

		if normalize:
			stats = sitk.LabelStatisticsImageFilter()
			stats.Execute(composite_image, intersection_mask_mean)
			overlap_composite_average = stats.GetMean(1)
			stats.Execute(current_image, intersection_mask_mean)
			overlap_current_average = stats.GetMean(1)
			delta_average = overlap_composite_average - overlap_current_average

			# Normalize the intensities of the images
			current_image = current_image + delta_average * mask_current

		composite_image = (
				(alpha) * intersection_mask * composite_image
				+ (1 - alpha) * intersection_mask * current_image
				+ 1 * (mask_composite - intersection_mask) * composite_image
				+ (mask_current - intersection_mask) * current_image
		)

	return composite_image


def final_registration(fixed_image, moving_image, initial_mutable_transformations):
	"""
    Register the two images using multiple starting transformations.
    Args:
        fixed_image (SimpleITK image): Estimated transformation maps points from this image to the
                                       moving_image.
        moving_image (SimpleITK image): Estimated transformation maps points from the fixed image to
                                        this image.
       initial_mutable_transformations (iterable, list like): Set of initial transformations, these will
                                                              be modified in place.
    """
	registration_method = sitk.ImageRegistrationMethod()
	registration_method.SetMetricAsCorrelation()
	registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
	registration_method.SetMetricSamplingPercentage(0.2)
	registration_method.SetOptimizerAsGradientDescent(
		learningRate=1.0, numberOfIterations=200
	)
	registration_method.SetOptimizerScalesFromPhysicalShift()

	def reg(transform):
		registration_method.SetInitialTransform(transform)
		registration_method.Execute(fixed_image, moving_image)
		return registration_method.GetMetricValue()

	final_values = [reg(transform) for transform in initial_mutable_transformations]
	return list(zip(initial_mutable_transformations, final_values))