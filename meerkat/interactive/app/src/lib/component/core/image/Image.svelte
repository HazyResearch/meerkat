<script lang="ts">
	// Encoded data.
	export let data: string;
	// Alt text.
	export let alt: string = '';
	export let classes: string = '';
	export let style = '';

	// Transform parameters.
	export let enableZoom: boolean = false;
	export let enablePan: boolean = false;
	export let zoom: number = 1.0;
	export let offset: Array<number> = [0, 0];
	
	// Cursor style.
	export let cursor: string = 'default';

	console.log("Image.svelte: enableZoom = ", zoom);

	const handleWheel = (event: WheelEvent) => {
		const { deltaY } = event;
		event.preventDefault();
		zoom += deltaY * 0.01;
		zoom = Math.max(zoom, 1);
	};

	const handleMouseDown = (event) => {
		event.preventDefault();
		const { clientX, clientY } = event;
		let lastX = clientX;
		let lastY = clientY;
		cursor = 'grabbing';

		const handleMouseMove = (event) => {
			const { clientX, clientY } = event;
			offset[0] += clientX - lastX;
			offset[1] += clientY - lastY;
			lastX = clientX;
			lastY = clientY;
		};

		const handleMouseUp = () => {
			window.removeEventListener('mousemove', handleMouseMove);
			window.removeEventListener('mouseup', handleMouseUp);
			cursor = 'default';
		};

		window.addEventListener('mousemove', handleMouseMove);
		window.addEventListener('mouseup', handleMouseUp);
	};

</script>

<img
	class={classes}
	src={data}
	{alt}
	style="cursor:{cursor}; transform: translate({offset[0]}px, {offset[1]}px) scale({zoom}); {style}"
	on:wheel={(e) => {enableZoom && handleWheel(e)}}
	on:mousedown={(e) => {enablePan && handleMouseDown(e)}}
/>
