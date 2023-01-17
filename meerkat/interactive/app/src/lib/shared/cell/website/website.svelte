<script lang="ts">
	export let data: string;
	export let height: number;
	
    $: width = height * 1.5;
</script>

<div
	class="thumbnail rounded-md shadow-md border-black"
	style:height={`${height}vh`}
	style:width={`${width}vh`}
>
	<iframe
		src={data}
		title={data}
        class="rounded-md"
		frameborder="”0”"
		style:height={`${height / 0.5}vh`}
		style:width={`${width / 0.5}vh`}
	/>
</div>

<style>
	.thumbnail iframe {
		-webkit-transform-origin: 0 0;
		-moz-transform-origin: 0 0;
		transform-origin: 0 0;

		-webkit-transform: scale(0.5, 0.5);
		-moz-transform: scale(0.5, 0.5);
		transform: scale(0.5, 0.5);

		overflow: hidden;
	}
	.thumbnail:after {
		content: '';
		display: block;
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
	}
</style>
