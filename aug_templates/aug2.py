import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.5, iaa.color.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30), to_colorspace='YCrCb', from_colorspace='RGB', random_order=True)),
	iaa.Sometimes(0.5, iaa.contrast.GammaContrast(gamma=(0.8, 1.2), per_channel=True)),
	iaa.Sometimes(0.4, iaa.geometric.Affine(scale=(0.8, 1.2), translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, rotate=(-30, 30), shear={'x': (-15, 15), 'y': (-15, 15)}, order=1, cval=0, mode='constant', fit_output=False)),
	iaa.Sometimes(0.4, iaa.size.CropAndPad(percent=(-0.1, 0.1), pad_mode='constant', pad_cval=0, keep_size=False, sample_independently=True)),
	iaa.Sometimes(0.2, iaa.blur.GaussianBlur(sigma=(0, 3))),
	iaa.Sometimes(0.2, iaa.arithmetic.JpegCompression(compression=(0, 10))),
	iaa.Sometimes(0.2, iaa.arithmetic.AdditiveGaussianNoise(loc=0, scale=(1, 5), per_channel=False)),
	iaa.Sometimes(0.2, iaa.arithmetic.SaltAndPepper(p=(0, 0.2), per_channel=False))
], random_order=False)
