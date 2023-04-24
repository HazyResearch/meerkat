<script lang="ts">
	import { convertClickCoordinatesToImageCoordinates, convertImageCoordinatesToClickCoordinates } from "$lib/utils/coordinates";

    export let points: Array<{ x: number; y: number; color?: number | string }> = [];
	export let pointSize: number = 10;
    export let image: HTMLImageElement | null;
    export let isActive: boolean = true;
    export let canvas: HTMLCanvasElement;

    let selectedPoints: Array<object> = [];
	$: displayPoints = image
		? convertToDisplayPoints(points, image)
		: [];
    
    function manageEventListeners(canvas, isActive: boolean) {
		if (!canvas) return;
		if (isActive) {
			canvas.addEventListener('click', addPoint);
		} else {
			canvas.removeEventListener('click', addPoint);
		}
	}

    function convertToDisplayPoints(points: Array<{ x: number; y: number }>, _image: HTMLImageElement) {
		const out = points.map((point) => {
			let clickCoordinates = convertImageCoordinatesToClickCoordinates(point, _image);
			clickCoordinates['point'] = point;
			return clickCoordinates;
		});
		return out;
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.code === 'Backspace' || event.code === 'Delete') {
			points = points.filter((point) => !selectedPoints.includes(point));
			selectedPoints = [];
		}

		if (event.code === 'KeyA' && event.ctrlKey) {
			selectedPoints = points;
		}
	}

    function handleSelect(point: { x: number; y: number; color?: number | string }) {
        if (selectedPoints.includes(point)) {
            selectedPoints = selectedPoints.filter((p) => p !== point);
        } else {
            selectedPoints = [...selectedPoints, point];
        }
    }

    // Handle selecting a point on the image.
	function addPoint(event: PointerEvent) {
		const imageCoordinates = convertClickCoordinatesToImageCoordinates(
			{ x: event.offsetX, y: event.offsetY },
			image
		);
		points = [...points, imageCoordinates];
	}
    
    $: manageEventListeners(canvas, isActive);
</script>

<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
<div on:keydown={handleKeydown} tabindex="0">
    {#each displayPoints as point}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <div
            style="position: absolute; left: {point.x - pointSize / 2}px; top: {point.y -
                pointSize / 2}px;"
            on:click={(e) => {
                handleSelect(point.point);
                e.stopPropagation();
            }}
        >
            <div
                style="width: {pointSize}px; height: {pointSize}px; background-color: red; border-radius: 50%; border: 2px solid {selectedPoints.includes(
                    point.point
                )
                    ? 'blue'
                    : 'black'};"
            />
        </div>
    {/each}
</div>