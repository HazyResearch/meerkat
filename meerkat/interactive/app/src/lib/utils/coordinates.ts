export function convertImageCoordinatesToClickCoordinates(
	coordinates: { x: number; y: number },
	image: HTMLImageElement
) {
	// Larger values means the image is scaled down more.
	// The larger ratio indicates the biggest resize that was
	// applied to the image.
	const imageRect = image.getBoundingClientRect();
	const heightRatio = image.naturalHeight / imageRect.height;
	const widthRatio = image.naturalWidth / imageRect.width;
	const ratio = Math.max(heightRatio, widthRatio);

	// The shape of the displayed image.
	// We assume the image is displayed with `contain` bounds.
	// This means the image will be scaled (preserving aspect ratio) to fit in the container.
	const imageHeight = image.naturalHeight / ratio;
	const imageWidth = image.naturalWidth / ratio;
	// padding should never be less than 0.
	const padTop = (imageRect.height - imageHeight) / 2;
	const padLeft = (imageRect.width - imageWidth) / 2;

	// The coordinates of the click relative to original image shape.
	return { x: coordinates.x / ratio + padLeft, y: coordinates.y / ratio + padTop };
}

export function convertClickCoordinatesToImageCoordinates(coordinates: {x: number, y: number}, image: HTMLImageElement) {
	const imageRect = image.getBoundingClientRect();
    const {x, y} = coordinates

	// Larger values means the image is scaled down more.
	// The larger ratio indicates the biggest resize that was
	// applied to the image.
	const heightRatio = image.naturalHeight / imageRect.height;
	const widthRatio = image.naturalWidth / imageRect.width;
	const ratio = Math.max(heightRatio, widthRatio);

	// The shape of the displayed image.
	// We assume the image is displayed with `contain` bounds.
	// This means the image will be scaled (preserving aspect ratio) to fit in the container.
	const imageHeight = image.naturalHeight / ratio;
	const imageWidth = image.naturalWidth / ratio;
	// padding should never be less than 0.
	const padTop = (imageRect.height - imageHeight) / 2;
	const padLeft = (imageRect.width - imageWidth) / 2;

	// The coordinates of the click relative to original image shape.
	const xImage = (x - padLeft) * ratio;
	const yImage = (y - padTop) * ratio;

	return { x: xImage, y: yImage };
}
