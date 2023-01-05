import imgaug.augmenters as iaa

seq = iaa.Sequential([
	iaa.Sometimes(0.5, iaa.color.MultiplyAndAddToBrightness(mul=(0.7, 1.3), add=(-50, 50), to_colorspace='YCrCb', from_colorspace='RGB', random_order=True)),
	iaa.Sometimes(0.5, iaa.contrast.GammaContrast(gamma=(0.7, 1.5), per_channel=True)),
	iaa.Sometimes(0.5, iaa.geometric.Affine(scale=(0.7, 1.4), translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)}, rotate=(-60, 60), shear={'x': (-30, 30), 'y': (-30, 30)}, order=1, cval=0, mode='constant', fit_output=False)),
	iaa.Sometimes(0.4, iaa.size.CropAndPad(percent=(-0.2, 0.2), pad_mode='constant', pad_cval=0, keep_size=False, sample_independently=True)),
	iaa.Sometimes(0.5, iaa.flip.Fliplr(p=1.0))
], random_order=False)
